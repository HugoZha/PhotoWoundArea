"""
segmentation.py — pluggable wound segmentation interface
====================================================

Provides a pluggable segmentation interface:
  - BaseSegmenter            unified interface segment(rgb) -> (mask, info)
  - ColorThresholdSegmenter classic CV baseline (LAB colour + Otsu), used as a
                            fallback/control when no model is available
  - SyntheticWoundSegmenter synthetic wound region for demos (used when sample
                            data has no annotations, explicitly marked as synthetic)
  - WoundSegModel            wraps a trained segmentation model (WoundUNet backbone,
                            see seg_model.py)

Note: real clinical deployment should use a trained segmentation model; the classic
      baseline here is only for engineering comparison and getting the pipeline
      running, not as an accuracy delivery criterion. WoundSegModel checkpoints are
      produced by wg_train/train.py, and its inference interface is exactly the
      same as the other segmenters in this file.
"""
from __future__ import annotations

import numpy as np


def _resolve_device(device: str | None) -> str:
    """Resolve "auto"/None to a concrete device (MPS preferred, then CPU),
    falling back to CPU when torch is unavailable."""
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
        """Return (mask (H,W) uint8 0/255, info dict)."""
        raise NotImplementedError


class ColorThresholdSegmenter(BaseSegmenter):
    """Classic colour-threshold baseline: threshold wound colour (reddish/dark
    red) in LAB space plus morphological closing.

    Effective at distinguishing wounds (red granulation, dark necrosis, yellow
    eschar) from healthy skin, but sensitive to lighting/skin tone; baseline
    only. Production should replace this with a trained model.
    """

    name = "color_threshold"

    def segment(self, rgb: np.ndarray):
        try:
            import cv2
        except Exception:
            # Fall back to a central patch when opencv is unavailable
            h, w = rgb.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            mask[h // 3:2 * h // 3, w // 3:2 * w // 3] = 255
            return mask, {"fallback": "no_cv2", "area_px": int(mask.sum()) // 255}

        rgb = np.ascontiguousarray(rgb)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        a = lab[:, :, 1].astype(np.float32)   # +a reddish
        b = lab[:, :, 2].astype(np.float32)   # +b yellowish

        # Wounds usually have high a (red) or high b (yellow eschar); combine
        # into a red-yellow score
        score = (a - 128.0) + 0.6 * (b - 128.0)
        score = np.clip(score, 0, 255).astype(np.uint8)
        _, bin_mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)

        # Keep the largest connected component
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8)
        if num > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            bin_mask = (labels == largest).astype(np.uint8) * 255
        else:
            bin_mask = np.zeros_like(bin_mask)

        area_px = int((bin_mask > 0).sum())
        return bin_mask, {"method": "LAB_a+b_otsu", "area_px": area_px}


class SyntheticWoundSegmenter(BaseSegmenter):
    """For demos: generate an elliptical wound region at the centre of the
    reference frame.

    Only used for end-to-end pipeline demos when "the sample data has no manually
    annotated wounds"; outputs info['synthetic']=True to clearly distinguish it
    from real segmentation results.
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
    """Keep the largest-area connected component in a binary mask (0/255 in,
    0/255 out).

    Aligned with ColorThresholdSegmenter's "keep the largest connected
    component" behaviour, suppressing scattered false detections by the
    segmentation model at skin blemishes/reflections far from the wound.
    """
    try:
        import cv2
    except Exception:
        return mask  # return unchanged when opencv is unavailable
    bin_mask = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8)
    if num <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


class WoundSegModel(BaseSegmenter):
    """Real interface wrapping a trained segmentation model (WoundUNet backbone).

    The interface is exactly the same as ColorThresholdSegmenter /
    SyntheticWoundSegmenter: ``segment(rgb) -> (mask (H,W) 0/255, info dict)``,
    so it can be passed directly to
    ``pipeline.run_pipeline(segmenter=WoundSegModel(...))`` for a one-line
    replacement.

    ``info`` carries ``synthetic=False`` plus ``prob_mean/prob_max``
    for subsequent confidence filtering and failure warning.

    Lazy torch loading: if torch is not installed or the weights are empty, an
    explicit error is raised only when ``segment`` is called, so other segmenters
    are unaffected.
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
                    "WoundSegModel was not given a weights_path; please train first "
                    "(wg_train/train.py) or pass in model weights; otherwise use "
                    "ColorThresholdSegmenter / SyntheticWoundSegmenter instead."
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
