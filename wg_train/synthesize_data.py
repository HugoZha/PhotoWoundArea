"""
synthesize_data.py — synthetic skin/wound RGB-mask pairs for smoke tests.

Each image combines a skin-toned background (smooth limb-like shading plus
Gaussian noise) with one irregular elliptical wound (granulation colour plus
texture noise). Position, size, angle and skin tone are randomised.

**Note**: engineering test data, not clinical data. Real deployment requires
expert-outlined masks; see docs/protocol.md.
"""
from __future__ import annotations

import numpy as np


def _skin_tone(rng: np.random.Generator) -> np.ndarray:
    """随机肤色基色 (R,G,B)，落在常见肤色区间。"""
    return np.array([
        rng.uniform(180, 225),
        rng.uniform(140, 175),
        rng.uniform(110, 150),
    ], dtype=np.float32)


def generate_pair(img_size: int = 256, rng: np.random.Generator | None = None,
                  wound_frac: float = 0.20, irregular: bool = True):
    """生成一对 (rgb uint8 (H,W,3), mask uint8 (H,W) 0/255)。

    Args:
        img_size: 输出正方形边长。
        rng: 随机源；不传则内部新建。
        wound_frac: 伤口长轴占图宽比例（控制难度）。
        irregular: 是否在椭圆上叠加角度扰动，使边界不规则（更接近真实创面）。
    """
    if rng is None:
        rng = np.random.default_rng()
    H = W = img_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # ── 背景：肤色 + 肢体式平滑明暗 + 噪声 ──
    base = _skin_tone(rng)
    shade = 0.85 + 0.15 * np.sin(xx / W * np.pi * 0.5)
    shade = shade * (0.9 + 0.1 * np.cos(yy / H * np.pi))
    rgb = np.empty((H, W, 3), np.float32)
    for c in range(3):
        rgb[..., c] = base[c] * shade
    rgb += rng.normal(0, 4, rgb.shape).astype(np.float32)

    # ── 伤口：旋转椭圆 + 可选角度扰动 ──
    cx = rng.uniform(0.35, 0.65) * W
    cy = rng.uniform(0.35, 0.65) * H
    ax = rng.uniform(wound_frac * 0.6, wound_frac) * W
    ay = ax * rng.uniform(0.6, 1.25)
    ang = rng.uniform(0, np.pi)

    coords = np.stack([xx - cx, yy - cy], -1).reshape(-1, 2).astype(np.float32)
    ca, sa = np.cos(-ang), np.sin(-ang)
    rot = coords @ np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
    rot = rot.reshape(H, W, 2)

    if irregular:
        th = np.arctan2(rot[..., 1], rot[..., 0])
        rr = 1.0 + rng.normal(0, 0.14, (H, W))
        rr = np.clip(rr, 0.6, 1.4)
    else:
        rr = np.ones((H, W), np.float32)

    mask = (((rot[..., 0] / ax) ** 2 + (rot[..., 1] / ay) ** 2) <= rr ** 2)
    mask = (mask > 0.5).astype(np.uint8) * 255

    # ── 伤口着色：偏红肉芽 + 纹理噪声（仅在掩膜内替换）──
    wound_col = np.array([rng.uniform(150, 200), rng.uniform(20, 60), rng.uniform(20, 50)],
                        dtype=np.float32)
    wm = mask > 0
    rgb[wm] = wound_col[None, :] + rng.normal(0, 8, (int(wm.sum()), 3)).astype(np.float32)
    rgb += rng.normal(0, 2.5, rgb.shape).astype(np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, mask


def generate_dataset(n: int, img_size: int = 256, seed: int = 0, **kwargs):
    """生成 n 对样本，返回 list[(rgb, mask)]。供 SyntheticSegDataset 内存加载。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(generate_pair(img_size=img_size, rng=rng, **kwargs))
    return out
