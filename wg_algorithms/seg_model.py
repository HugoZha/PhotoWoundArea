"""
seg_model.py — lightweight trainable U-Net for wound segmentation
==================================================

Trainable and inferable U-Net used by the segmentation interface:
  - input RGB -> output a single-channel wound probability map (sigmoid)
  - compact architecture (default base=32, depth=4), enabling second-scale
    inference on Mac/CPU
  - checkpoints carry ``config`` metadata (base/depth/in_ch/img_size), matched
    automatically on load
  - **lazy torch loading**: this file imports without error when torch is absent,
    reporting the missing package only when it is actually used, so that
    importing the whole wg_algorithms library on GPU-less machines is unaffected.

Training/evaluation scripts are in ``wg_train/`` (train.py / evaluate.py).
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


# Project-wide default model hyperparameters; train.py writes them into the
# checkpoint and load reads them back to avoid mismatches.
DEFAULT_CONFIG = {"in_ch": 3, "base": 32, "depth": 4, "img_size": 256}


class _DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x2."""

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
    """Compact U-Net: 3-channel RGB -> 1-channel wound probability map (sigmoid
    output, [0,1]).

    The encoder downsamples by 2x at each level, the decoder upsamples via
    transposed convolutions with skip connections. The architecture adapts to
    ``depth``, and ``base`` controls the number of channels in the first layer
    (doubling at each subsequent level).
    """

    def __init__(self, in_ch: int = 3, base: int = 32, depth: int = 4):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
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
            if x.shape[-2:] != s.shape[-2:]:  # handle odd/even size differences
                x = F.interpolate(x, size=s.shape[-2:], mode="nearest")
            x = torch.cat([s, x], dim=1)
            x = self.up[i](x)
        return torch.sigmoid(self.final(x))


def load_wound_model(path: str, device: str = "cpu"):
    """Load a checkpoint (compatible with ``state_dict`` or
    ``{'state_dict', 'config'}``).

    Automatically restores base/depth/in_ch from the checkpoint's ``config`` to
    avoid inconsistency with the training-time architecture.
    """
    if not HAS_TORCH:
        raise RuntimeError("Loading a real segmentation model requires torch; "
                           "please run `pip install torch` first.")
    if device in (None, "", "auto"):   # automatic device selection: MPS preferred, otherwise CPU
        device = "mps" if (getattr(torch.backends, "mps", None) is not None
                           and torch.backends.mps.is_available()) else "cpu"
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        cfg = sd.get("config", DEFAULT_CONFIG)
        state = sd["state_dict"]
    else:  # bare state_dict
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
    """RGB (H,W,3) uint8 -> wound probability map (H,W) float[0,1], upsampled
    back to the original size with nearest neighbour.

    Preprocessing: PIL bilinear resize to (img_size,img_size) -> /255 -> (1,3,H,W).
    Postprocessing upsamples with PIL nearest neighbour to avoid relying on
    torch for the final interpolation (and to avoid sub-pixel artefacts).
    """
    from PIL import Image

    if not HAS_TORCH:
        raise RuntimeError("Inference requires torch; please run `pip install torch` first.")
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
