"""
metrics.py — segmentation quality metrics (based on numpy + scipy)
====================================================

Inputs are all (H,W) binary/grayscale masks (either 0/255 or 0/1; binarized internally at >0.5).
Provided:
  - dice, iou, precision, recall, specificity
  - hd95    : 95% Hausdorff distance (surface error, unit = pixel)
  - boundary_f1 : boundary F1 with a tolerance of tol pixels (Boundary F1 score)

All functions handle degenerate cases such as "all empty / all full" robustly, without dividing by zero or raising errors.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion


def _bin(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) > 0.5).astype(np.uint8)


def dice(a, b) -> float:
    a, b = _bin(a), _bin(b)
    inter = float((a & b).sum())
    return 2 * inter / (a.sum() + b.sum() + 1e-6)


def iou(a, b) -> float:
    a, b = _bin(a), _bin(b)
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / (union + 1e-6)


def precision_recall(a, b):
    """With a=prediction, b=ground truth, returns (precision, recall, specificity)."""
    a, b = _bin(a), _bin(b)
    tp = float((a & b).sum())
    fp = float((a & ~b).sum())
    fn = float((~a & b).sum())
    tn = float((~a & ~b).sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    return precision, recall, specificity


def _surface_distances(a: np.ndarray, b: np.ndarray):
    """Return an array of distances from a's surface points to the nearest b surface points (the reverse is merged by the caller)."""
    a, b = _bin(a), _bin(b)
    dt_b = distance_transform_edt(1 - b)   # distance from each a pixel to the nearest b
    dt_a = distance_transform_edt(1 - a)
    da = dt_b[a > 0]                        # a surface points to b
    db = dt_a[b > 0]                        # b surface points to a
    return da, db


def hd95(a, b) -> float:
    a, b = _bin(a), _bin(b)
    if a.sum() == 0 and b.sum() == 0:
        return 0.0
    if a.sum() == 0 or b.sum() == 0:
        return float("inf")
    da, db = _surface_distances(a, b)
    return max(np.percentile(da, 95) if da.size else 0.0,
               np.percentile(db, 95) if db.size else 0.0)


def boundary_f1(a, b, tol: int = 2) -> float:
    """Boundary F1: how well the two masks' boundaries match within a tol-pixel tolerance."""
    a, b = _bin(a), _bin(b)
    ba = np.logical_xor(a, binary_erosion(a)).astype(np.uint8)
    bb = np.logical_xor(b, binary_erosion(b)).astype(np.uint8)
    if ba.sum() == 0 and bb.sum() == 0:
        return 1.0
    if ba.sum() == 0 or bb.sum() == 0:
        return 0.0
    # distance from bb boundary points to the nearest ba boundary (<=tol counts as a hit)
    dt_a = distance_transform_edt(1 - ba)
    dt_b = distance_transform_edt(1 - bb)
    rec = float((dt_a[bb > 0] <= tol).mean())   # ground-truth boundary covered by the prediction
    prec = float((dt_b[ba > 0] <= tol).mean())  # predicted boundary aligned with ground truth
    return 2 * prec * rec / (prec + rec + 1e-6)


def evaluate_pairs(pred_list, gt_list) -> dict:
    """Average the metrics over a batch of (pred, gt) pairs.

    A single empty prediction or empty reference makes HD95 infinite, which
    propagates to a plain mean. Three quantities are therefore reported:
        hd95          strict mean including inf, for diagnostics only;
        hd95_finite   mean over pairs where both masks are non-empty, for reporting;
        empty_pred    number of complete misses, reported separately.
    """
    n = len(pred_list)
    if n == 0:
        return {}
    dice_v = np.mean([dice(p, g) for p, g in zip(pred_list, gt_list)])
    iou_v = np.mean([iou(p, g) for p, g in zip(pred_list, gt_list)])
    pr = np.mean([precision_recall(p, g) for p, g in zip(pred_list, gt_list)], axis=0)
    bf = np.mean([boundary_f1(p, g) for p, g in zip(pred_list, gt_list)])

    hd_all = [hd95(p, g) for p, g in zip(pred_list, gt_list)]
    hd_fin = [h for h in hd_all if np.isfinite(h)]
    empty_pred = int(sum(1 for p, _ in zip(pred_list, gt_list)
                         if _bin(p).sum() == 0))
    empty_gt = int(sum(1 for _, g in zip(pred_list, gt_list)
                       if _bin(g).sum() == 0))

    return {
        "n": n,
        "dice": float(dice_v),
        "iou": float(iou_v),
        "precision": float(pr[0]),
        "recall": float(pr[1]),
        "specificity": float(pr[2]),
        "hd95": float(np.mean(hd_all)) if hd_all else float("inf"),
        "hd95_finite": float(np.mean(hd_fin)) if hd_fin else float("nan"),
        "hd95_finite_median": float(np.median(hd_fin)) if hd_fin else float("nan"),
        "boundary_f1": float(bf),
        "empty_pred": empty_pred,
        "empty_gt": empty_gt,
        "n_hd_finite": len(hd_fin),
    }
