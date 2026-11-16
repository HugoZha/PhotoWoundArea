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
    """Random base skin colour (R,G,B), falling within a common skin-tone range."""
    return np.array([
        rng.uniform(180, 225),
        rng.uniform(140, 175),
        rng.uniform(110, 150),
    ], dtype=np.float32)


def generate_pair(img_size: int = 256, rng: np.random.Generator | None = None,
                  wound_frac: float = 0.20, irregular: bool = True):
    """Generate a pair (rgb uint8 (H,W,3), mask uint8 (H,W) 0/255).

    Args:
        img_size: output square side length.
        rng: random source; a new one is created internally if not provided.
        wound_frac: ratio of the wound's long axis to the image width (controls difficulty).
        irregular: whether to add angular perturbation to the ellipse, making the boundary irregular (closer to a real wound).
    """
    if rng is None:
        rng = np.random.default_rng()
    H = W = img_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # ── background: skin tone + limb-like smooth shading + noise ──
    base = _skin_tone(rng)
    shade = 0.85 + 0.15 * np.sin(xx / W * np.pi * 0.5)
    shade = shade * (0.9 + 0.1 * np.cos(yy / H * np.pi))
    rgb = np.empty((H, W, 3), np.float32)
    for c in range(3):
        rgb[..., c] = base[c] * shade
    rgb += rng.normal(0, 4, rgb.shape).astype(np.float32)

    # ── wound: rotated ellipse + optional angular perturbation ──
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

    # ── wound colouring: reddish granulation + texture noise (replaced only within the mask) ──
    wound_col = np.array([rng.uniform(150, 200), rng.uniform(20, 60), rng.uniform(20, 50)],
                        dtype=np.float32)
    wm = mask > 0
    rgb[wm] = wound_col[None, :] + rng.normal(0, 8, (int(wm.sum()), 3)).astype(np.float32)
    rgb += rng.normal(0, 2.5, rgb.shape).astype(np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, mask


def generate_dataset(n: int, img_size: int = 256, seed: int = 0, **kwargs):
    """Generate n pairs, returning list[(rgb, mask)]. For in-memory loading by SyntheticSegDataset."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(generate_pair(img_size=img_size, rng=rng, **kwargs))
    return out
