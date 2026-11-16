"""
dataset.py — wound segmentation dataset (real files / in-memory synthetic)
==================================================

Two data sources, with a unified output ``{"image": (3,H,W) float[0,1], "mask": (1,H,W) float[0,1]}``:

  - ImageMaskDataset : real paired files of "RGB + manually outlined mask"
      Directory convention (by default, files with the same name are paired automatically):
        <image_dir>/*.png           color image (3 channels)
        <mask_dir>/*.png            single-channel mask (0/255 or 0/1)
      If the filenames in the two directories do not match exactly, they are aligned by sorted matching prefixes.
  - SyntheticSegDataset : generated in memory by synthesize_data, no on-disk annotations needed

The training/evaluation scripts pick one based on the passed arguments; the training-loop code is identical either way.
"""
from __future__ import annotations

import os
import glob
import random

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False
    Dataset = object  # type: ignore

from PIL import Image, ImageEnhance

from . import synthesize_data


def _match_pairs(image_dir: str, mask_dir: str):
    """Pair images with masks by filename (extension removed), returns [(img_path, mask_path), ...]."""
    img_files = sorted(glob.glob(os.path.join(image_dir, "*.*")))
    pairs = []
    for ip in img_files:
        name = os.path.splitext(os.path.basename(ip))[0]
        # Look for the same name with any extension in mask_dir
        cand = glob.glob(os.path.join(mask_dir, name + ".*"))
        if cand:
            pairs.append((ip, cand[0]))
    return pairs


class _Transforms:
    """Lightweight augmentation: synchronized geometry + color jitter (applied to the image only).

    Designed for mixed multi-source training: cameras, lighting, and skin tones vary widely across datasets,
    so in addition to flips/rotations we add random scale-crop and contrast/saturation jitter,
    preventing the model from memorizing "the imaging style of a particular dataset" as a feature.
    """

    def __init__(self, augment: bool = False, seed: int = 0):
        self.augment = augment
        self.rng = random.Random(seed)

    def __call__(self, img: Image.Image, mask: Image.Image):
        if not self.augment:
            return img, mask
        # ---- geometry (image + mask strictly synchronized)
        if self.rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if self.rng.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        k = self.rng.randint(0, 3)
        if k:
            img = img.rotate(90 * k, expand=True)
            mask = mask.rotate(90 * k, expand=True)
        # Random scale-crop: simulate differences in the wound's fraction of the frame at different shooting distances
        if self.rng.random() < 0.5:
            w, h = img.size
            s = self.rng.uniform(0.75, 1.0)
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            l, t = self.rng.randint(0, w - nw), self.rng.randint(0, h - nh)
            box = (l, t, l + nw, t + nh)
            img = img.crop(box)
            mask = mask.crop(box)
        # ---- color jitter (image only)
        img = _adjust_brightness(img, 1.0 + self.rng.uniform(-0.15, 0.15))
        if self.rng.random() < 0.5:
            img = _adjust_contrast(img, 1.0 + self.rng.uniform(-0.25, 0.25))
        if self.rng.random() < 0.5:
            img = _adjust_saturation(img, 1.0 + self.rng.uniform(-0.25, 0.25))
        return img, mask


def _adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    """factor=1 gives the original image. Implemented with native PIL to avoid extra dependencies."""
    return ImageEnhance.Brightness(img).enhance(factor)


def _adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def _adjust_saturation(img: Image.Image, factor: float) -> Image.Image:
    """factor=0 gives grayscale, 1 gives the original image — used to mitigate hue differences across datasets/skin tones."""
    return ImageEnhance.Color(img).enhance(factor)


def _prepare(img: Image.Image, mask: Image.Image, img_size: int):
    """Resize to a square -> convert to tensor / array."""
    img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    mask = mask.convert("L").resize((img_size, img_size), Image.NEAREST)
    x = np.asarray(img, dtype=np.float32) / 255.0
    m = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
    x = np.transpose(x, (2, 0, 1))
    return x, m[None, ...]


class ImageMaskDataset(Dataset):
    """Real-file dataset (RGB images + outlined masks)."""

    def __init__(self, image_dir: str, mask_dir: str, img_size: int = 256,
                 augment: bool = False, seed: int = 0, stems: list = None):
        if not HAS_TORCH:
            raise RuntimeError("The dataset requires torch; please run `pip install torch` first.")
        self.pairs = _match_pairs(image_dir, mask_dir)
        if stems is not None:
            keep = set(stems)
            self.pairs = [(ip, mp) for ip, mp in self.pairs
                          if os.path.splitext(os.path.basename(ip))[0] in keep]
        if not self.pairs:
            raise RuntimeError(f"No paired samples found in {image_dir} / {mask_dir}.")
        self.img_size = img_size
        self.tf = _Transforms(augment=augment, seed=seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = Image.open(ip)
        mask = Image.open(mp)
        img, mask = self.tf(img, mask)
        x, m = _prepare(img, mask, self.img_size)
        if HAS_TORCH:
            x = torch.from_numpy(x)
            m = torch.from_numpy(m)
        return {"image": x, "mask": m}


class SyntheticSegDataset(Dataset):
    """In-memory synthetic dataset: generates n pairs via synthesize_data, with optional augmentation."""

    def __init__(self, n: int = 200, img_size: int = 256, augment: bool = False,
                 seed: int = 0, wound_frac: float = 0.20):
        if not HAS_TORCH:
            raise RuntimeError("The dataset requires torch; please run `pip install torch` first.")
        self.samples = synthesize_data.generate_dataset(n, img_size=img_size,
                                                        seed=seed, wound_frac=wound_frac)
        self.img_size = img_size
        self.tf = _Transforms(augment=augment, seed=seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb, mask = self.samples[idx]
        img = Image.fromarray(rgb).convert("RGB")
        mask = Image.fromarray(mask).convert("L")
        img, mask = self.tf(img, mask)
        x, m = _prepare(img, mask, self.img_size)
        if HAS_TORCH:
            x = torch.from_numpy(x)
            m = torch.from_numpy(m)
        return {"image": x, "mask": m}
