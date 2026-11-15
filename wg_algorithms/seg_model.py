"""
seg_model.py — lightweight trainable U-Net for wound segmentation
==================================================

Trainable and inferable U-Net used by the segmentation interface:
  - 输入 RGB -> 输出单通道创面概率图 (sigmoid)
  - 紧凑结构（默认 base=32, depth=4），可在 Mac/CPU 上秒级推理
  - checkpoint 自带 ``config`` 元数据（base/depth/in_ch/img_size），加载时自动匹配
  - **torch 懒加载**：本文件在无 torch 环境下 import 不报错，仅在使用时才报缺包，
    从而不影响整个 wg_algorithms 库在无 GPU 机器上的导入。

训练/评估脚本见 ``wg_train/``（train.py / evaluate.py）。
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:  # pragma: no cover - fallback when torch is unavailable
    HAS_TORCH = False
    torch = None
    nn = None
    F = None


# 项目统一的模型超参默认值；train.py 写进 checkpoint，load 时回读，避免不匹配。
DEFAULT_CONFIG = {"in_ch": 3, "base": 32, "depth": 4, "img_size": 256}


class _DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x2。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class WoundUNet(nn.Module):
    """紧凑 U-Net：3 通道 RGB -> 1 通道创面概率图（sigmoid 输出，[0,1]）。

    编码器逐级 2x 下采样，解码器逐级转置卷积上采样 + 跳跃连接。
    结构随 ``depth`` 自适应，``base`` 控制首层通道数（每下一层翻倍）。
    """

    def __init__(self, in_ch: int = 3, base: int = 32, depth: int = 4):
        super().__init__()
        if depth < 2:
            raise ValueError("depth 至少为 2")
        self.depth = depth
        self.in_ch = in_ch
        self.base = base

        self.enc = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        ch = in_ch
        for i in range(depth):
            out = base * (2 ** i)
            self.enc.append(_DoubleConv(ch, out))
            ch = out

        self.upconv = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.upconv.append(nn.ConvTranspose2d(base * (2 ** i), base * (2 ** (i - 1)), 2, stride=2))
            self.up.append(_DoubleConv(base * (2 ** i), base * (2 ** (i - 1))))

        self.final = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x)
            if i < self.depth - 1:
                skips.append(x)
                x = self.pool(x)
        for i in range(self.depth - 1):
            x = self.upconv[i](x)
            s = skips[-(i + 1)]
            if x.shape[-2:] != s.shape[-2:]:  # 处理奇偶尺寸差异
                x = F.interpolate(x, size=s.shape[-2:], mode="nearest")
            x = torch.cat([s, x], dim=1)
            x = self.up[i](x)
        return torch.sigmoid(self.final(x))


def load_wound_model(path: str, device: str = "cpu"):
    """加载 checkpoint（兼容 ``state_dict`` 或 ``{'state_dict', 'config'}``）。

    自动从 checkpoint 的 ``config`` 还原 base/depth/in_ch，避免与训练时结构不一致。
    """
    if not HAS_TORCH:
        raise RuntimeError("加载真实分割模型需要 torch；请先 `pip install torch`。")
    if device in (None, "", "auto"):   # 自动选设备：MPS 优先，否则 CPU
        device = "mps" if (getattr(torch.backends, "mps", None) is not None
                           and torch.backends.mps.is_available()) else "cpu"
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        cfg = sd.get("config", DEFAULT_CONFIG)
        state = sd["state_dict"]
    else:  # 裸 state_dict
        cfg = DEFAULT_CONFIG
        state = sd
    model = WoundUNet(
        in_ch=cfg.get("in_ch", 3),
        base=cfg.get("base", 32),
        depth=cfg.get("depth", 4),
    )
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def predict_proba(model, rgb, device: str = "cpu", img_size: int = 256) -> np.ndarray:
    """RGB (H,W,3) uint8 -> 创面概率图 (H,W) float[0,1]，最近邻上采样回原尺寸。

    预处理：PIL 双线性 resize 到 (img_size,img_size) -> /255 -> (1,3,H,W)。
    后处理上采样用 PIL 最近邻，避免依赖 torch 做最终插值（也避免亚像素伪影）。
    """
    from PIL import Image

    if not HAS_TORCH:
        raise RuntimeError("推理需要 torch；请先 `pip install torch`。")
    if isinstance(rgb, np.ndarray):
        img = Image.fromarray(rgb).convert("RGB")
    else:
        img = rgb.convert("RGB")
    orig_w, orig_h = img.size
    img_r = img.resize((img_size, img_size), Image.BILINEAR)
    x = np.asarray(img_r, dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None].copy()
    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        p = model(xt)[0, 0].cpu().numpy()
    p_img = Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8))
    p_img = p_img.resize((orig_w, orig_h), Image.NEAREST)
    return np.asarray(p_img, dtype=np.float32) / 255.0
