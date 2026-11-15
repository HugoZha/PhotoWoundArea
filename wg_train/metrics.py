"""
metrics.py — 分割质量评估指标（基于 numpy + scipy）
====================================================

输入均为 (H,W) 的二值/灰度掩膜（任意 0/255 或 0/1，内部按 >0.5 二值化）。
提供：
  - dice, iou, precision, recall, specificity
  - hd95    : 95% Hausdorff 距离（表面误差，单位=像素）
  - boundary_f1 : 容忍容差 tol 像素的边界 F1（Boundary F1 score）

所有函数对"全空 / 全满"等退化情况做了稳健处理，不会除零或抛错。
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
    """以 a=预测, b=真值，返回 (precision, recall, specificity)。"""
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
    """返回 a 表面点到最近 b 表面点的距离数组（反之亦然由调用方合并）。"""
    a, b = _bin(a), _bin(b)
    dt_b = distance_transform_edt(1 - b)   # 每个 a 像素到最近 b 的距离
    dt_a = distance_transform_edt(1 - a)
    da = dt_b[a > 0]                        # a 表面点到 b
    db = dt_a[b > 0]                        # b 表面点到 a
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
    """边界 F1：两掩膜边界在 tol 像素容差内的匹配程度。"""
    a, b = _bin(a), _bin(b)
    ba = np.logical_xor(a, binary_erosion(a)).astype(np.uint8)
    bb = np.logical_xor(b, binary_erosion(b)).astype(np.uint8)
    if ba.sum() == 0 and bb.sum() == 0:
        return 1.0
    if ba.sum() == 0 or bb.sum() == 0:
        return 0.0
    # bb 边界点到最近 ba 边界的距离（<=tol 记为命中）
    dt_a = distance_transform_edt(1 - ba)
    dt_b = distance_transform_edt(1 - bb)
    rec = float((dt_a[bb > 0] <= tol).mean())   # 真值边界被预测覆盖
    prec = float((dt_b[ba > 0] <= tol).mean())  # 预测边界对齐真值
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
