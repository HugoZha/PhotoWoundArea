"""
segmentation.py — pluggable wound segmentation interface
====================================================

提供可插拔分割接口：
  - BaseSegmenter           统一接口 segment(rgb) -> (mask, info)
  - ColorThresholdSegmenter 经典 CV 基线（LAB 色彩 + Otsu），用于无模型时的兜底/对照
  - SyntheticWoundSegmenter 演示用合成伤口区域（样本数据无标注时使用，明确标注为合成）
  - WoundSegModel           对接训练好的分割模型（WoundUNet 骨干，见 seg_model.py）

注意：真实临床部署应使用训练好的分割模型；此处经典基线仅作工程对照与
      pipeline 跑通，不作为精度交付口径。WoundSegModel 的权重由
      wg_train/train.py 训练产出，推理接口与本文件其他 segmenter 完全一致。
"""
from __future__ import annotations

import numpy as np


def _resolve_device(device: str | None) -> str:
    """把 "auto"/None 解析为具体设备（MPS 优先，其次 CPU），无 torch 时回落 CPU。"""
    if device in (None, "", "auto"):
        try:
            import torch
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return device


class BaseSegmenter:
    name = "base"

    def segment(self, rgb: np.ndarray):
        """返回 (mask (H,W) uint8 0/255, info dict)。"""
        raise NotImplementedError


class ColorThresholdSegmenter(BaseSegmenter):
    """经典色彩阈值基线：在 LAB 空间对伤口色（偏红/暗红）做阈值 + 形态学闭合。

    对创面（肉芽红、坏死暗、黄痂）与健康肤色差异有效，但对光照/肤色敏感，
    仅作基线。生产应替换为训练模型。
    """

    name = "color_threshold"

    def segment(self, rgb: np.ndarray):
        try:
            import cv2
        except Exception:
            # 无 opencv 时退化为中心块
            h, w = rgb.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            mask[h // 3:2 * h // 3, w // 3:2 * w // 3] = 255
            return mask, {"fallback": "no_cv2", "area_px": int(mask.sum()) // 255}

        rgb = np.ascontiguousarray(rgb)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        a = lab[:, :, 1].astype(np.float32)   # +a 偏红
        b = lab[:, :, 2].astype(np.float32)   # +b 偏黄

        # 创面通常 a 较高（红）或 b 较高（黄痂）；综合 red-yellow 指标
        score = (a - 128.0) + 0.6 * (b - 128.0)
        score = np.clip(score, 0, 255).astype(np.uint8)
        _, bin_mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)

        # 取最大连通域
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8)
        if num > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            bin_mask = (labels == largest).astype(np.uint8) * 255
        else:
            bin_mask = np.zeros_like(bin_mask)

        area_px = int((bin_mask > 0).sum())
        return bin_mask, {"method": "LAB_a+b_otsu", "area_px": area_px}


class SyntheticWoundSegmenter(BaseSegmenter):
    """演示用：在参考帧中心生成一枚椭圆伤口区域。

    仅用于「样本数据尚无人工标注伤口」时的端到端 pipeline 演示，
    输出 info['synthetic']=True 以明确区别于真实分割结果。
    """

    name = "synthetic"

    def __init__(self, ax_frac: float = 0.18, ay_frac: float = 0.12):
        self.ax_frac = ax_frac
        self.ay_frac = ay_frac

    def segment(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        cx, cy = w // 2, h // 2
        ax, ay = int(w * self.ax_frac), int(h * self.ay_frac)
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1).astype(np.uint8) * 255
        return mask, {"synthetic": True, "area_px": int(mask.sum()) // 255,
                      "center": (cx, cy), "axes": (ax, ay)}


def _keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    """保留二值掩膜中面积最大的连通域（输入 0/255，输出 0/255）。

    与 ColorThresholdSegmenter 的"取最大连通域"行为对齐，抑制分割模型
    在远离创面的皮肤瑕疵/反光处产生的零散误检。
    """
    try:
        import cv2
    except Exception:
        return mask  # 无 opencv 时原样返回
    bin_mask = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8)
    if num <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


class WoundSegModel(BaseSegmenter):
    """对接训练好的分割模型（WoundUNet 骨干）的真实接口。

    与 ColorThresholdSegmenter / SyntheticWoundSegmenter 接口完全一致：
    ``segment(rgb) -> (mask (H,W) 0/255, info dict)``，因此可直接传入
    ``pipeline.run_pipeline(segmenter=WoundSegModel(...))`` 一键替换。

    ``info`` carries ``synthetic=False`` plus ``prob_mean/prob_max``
    用于后续置信度筛选与失败预警。

    torch 懒加载：未安装 torch 或权重为空时，调用 ``segment`` 才会显式报错，
    不影响其他 segmenter 的使用。
    """

    name = "trained_model"

    def __init__(self, weights_path: str | None = None, device: str = "auto",
                 img_size: int = 256, threshold: float = 0.5, keep_largest: bool = True):
        self.weights_path = weights_path
        self.device = _resolve_device(device)
        self.img_size = img_size
        self.threshold = threshold
        self.keep_largest = keep_largest
        self._model = None
        if weights_path:
            self._load()

    def _load(self):
        from .seg_model import load_wound_model
        self._model = load_wound_model(self.weights_path, device=self.device)

    def segment(self, rgb: np.ndarray):
        if self._model is None:
            if self.weights_path:
                self._load()
            else:
                raise RuntimeError(
                    "WoundSegModel 未提供 weights_path；请先训练（wg_train/train.py）"
                    "或传入模型权重，否则改用 ColorThresholdSegmenter / SyntheticWoundSegmenter。"
                )
        from .seg_model import predict_proba
        proba = predict_proba(self._model, rgb, self.device, self.img_size)  # (H,W)
        mask = (proba >= self.threshold).astype(np.uint8) * 255
        if self.keep_largest:
            mask = _keep_largest_cc(mask)
        area_px = int((mask > 0).sum())
        return mask, {
            "synthetic": False,
            "model_path": self.weights_path,
            "threshold": self.threshold,
            "img_size": self.img_size,
            "prob_mean": float(proba.mean()),
            "prob_max": float(proba.max()),
            "area_px": area_px,
        }
