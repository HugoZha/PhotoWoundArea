"""
dataset.py — 伤口分割数据集（真实文件 / 合成内存）
==================================================

两种数据源，统一输出 ``{"image": (3,H,W) float[0,1], "mask": (1,H,W) float[0,1]}``：

  - ImageMaskDataset : 真实「RGB + 人工勾画掩膜」配对文件
      目录约定（缺省自动配对同名文件）：
        <image_dir>/*.png           彩色图（3 通道）
        <mask_dir>/*.png            单通道掩膜（0/255 或 0/1）
      若两目录下文件名不完全一致，会按排序后的同名前缀对齐。
  - SyntheticSegDataset : 由 synthesize_data 在内存生成，无需磁盘标注

训练/评估脚本根据传入参数二选一，训练循环代码完全一致。
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
    """按文件名（去扩展名）配对图像与掩膜，返回 [(img_path, mask_path), ...]。"""
    img_files = sorted(glob.glob(os.path.join(image_dir, "*.*")))
    pairs = []
    for ip in img_files:
        name = os.path.splitext(os.path.basename(ip))[0]
        # 在 mask_dir 中查找同名任意扩展名
        cand = glob.glob(os.path.join(mask_dir, name + ".*"))
        if cand:
            pairs.append((ip, cand[0]))
    return pairs


class _Transforms:
    """轻量增广：几何同步 + 颜色抖动（仅作用于图像）。

    面向多源混合训练：不同数据集的相机、光照、肤色差异很大，
    因此除翻转/旋转外还加了随机缩放裁剪与对比度/饱和度抖动，
    让模型不要把"某个数据集的成像风格"当特征记住。
    """

    def __init__(self, augment: bool = False, seed: int = 0):
        self.augment = augment
        self.rng = random.Random(seed)

    def __call__(self, img: Image.Image, mask: Image.Image):
        if not self.augment:
            return img, mask
        # ---- 几何（图像 + 掩膜严格同步）
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
        # 随机缩放裁剪：模拟不同拍摄距离下创面占画面的比例差异
        if self.rng.random() < 0.5:
            w, h = img.size
            s = self.rng.uniform(0.75, 1.0)
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            l, t = self.rng.randint(0, w - nw), self.rng.randint(0, h - nh)
            box = (l, t, l + nw, t + nh)
            img = img.crop(box)
            mask = mask.crop(box)
        # ---- 颜色抖动（只动图像）
        img = _adjust_brightness(img, 1.0 + self.rng.uniform(-0.15, 0.15))
        if self.rng.random() < 0.5:
            img = _adjust_contrast(img, 1.0 + self.rng.uniform(-0.25, 0.25))
        if self.rng.random() < 0.5:
            img = _adjust_saturation(img, 1.0 + self.rng.uniform(-0.25, 0.25))
        return img, mask


def _adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    """factor=1 为原图。用 PIL 原生实现，避免引入额外依赖。"""
    return ImageEnhance.Brightness(img).enhance(factor)


def _adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def _adjust_saturation(img: Image.Image, factor: float) -> Image.Image:
    """factor=0 变灰度、1 为原图 —— 用来缓解数据集间/肤色间的色调差异。"""
    return ImageEnhance.Color(img).enhance(factor)


def _prepare(img: Image.Image, mask: Image.Image, img_size: int):
    """resize 到正方形 -> 转 tensor / 数组。"""
    img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    mask = mask.convert("L").resize((img_size, img_size), Image.NEAREST)
    x = np.asarray(img, dtype=np.float32) / 255.0
    m = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)
    x = np.transpose(x, (2, 0, 1))
    return x, m[None, ...]


class ImageMaskDataset(Dataset):
    """真实文件数据集（RGB 图 + 勾画掩膜）。"""

    def __init__(self, image_dir: str, mask_dir: str, img_size: int = 256,
                 augment: bool = False, seed: int = 0, stems: list = None):
        if not HAS_TORCH:
            raise RuntimeError("数据集需要 torch；请先 `pip install torch`。")
        self.pairs = _match_pairs(image_dir, mask_dir)
        if stems is not None:
            keep = set(stems)
            self.pairs = [(ip, mp) for ip, mp in self.pairs
                          if os.path.splitext(os.path.basename(ip))[0] in keep]
        if not self.pairs:
            raise RuntimeError(f"在 {image_dir} / {mask_dir} 未找到配对样本。")
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
    """合成内存数据集：由 synthesize_data 生成 n 对样本，可选增广。"""

    def __init__(self, n: int = 200, img_size: int = 256, augment: bool = False,
                 seed: int = 0, wound_frac: float = 0.20):
        if not HAS_TORCH:
            raise RuntimeError("数据集需要 torch；请先 `pip install torch`。")
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
