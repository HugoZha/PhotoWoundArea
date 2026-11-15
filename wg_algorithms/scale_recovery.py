"""
scale_recovery.py — metric scale recovery from AR camera poses
====================================================================

ARKit/ARCore VIO already provides metric (metre) camera poses and world-space
feature points, so no scale-factor estimation is needed here. This module:

  1) aggregates sparse world points across frames by feature identifier into
     one global metric point cloud;
  2) estimates tracking quality from the 3D jitter of a feature identifier
     observed in multiple frames (smaller jitter means a more stable scale);
  3) returns the metric cloud and its confidence for the downstream stages.

The photographic pipeline additionally uses a scale reference in the frame.
"""
from __future__ import annotations

import numpy as np


class MetricReconstructor:
    def __init__(self, jitter_thresh_m: float = 0.02):
        """
        Args:
            jitter_thresh_m: 单点跨帧位置抖动阈值（米），低于此值视为稳定跟踪。
        """
        self.jitter_thresh_m = jitter_thresh_m

    def reconstruct(self, frames):
        """输入 wg_ar_loader 加载的 frame 列表，输出聚合结果。

        Returns:
            dict: {
                "points": (N,3) 全局度量点云（米），
                "identifiers": (N,) int64，
                "observations_per_point": (N,) 每个点被多少帧观测到，
                "jitter_per_point": (N,) 每个点跨帧位置标准差（米），
                "stats": {...} 汇总指标，
            }
        """
        id_to_pts = {}      # id -> list[(frame_idx, xyz)]
        id_to_frames = {}   # id -> list[frame_idx]
        for fi, fr in enumerate(frames):
            fp = fr.get("feature_points")
            fids = fr.get("feature_point_ids")
            if fp is None or fids is None:
                continue
            fp = np.asarray(fp, dtype=np.float64)
            fids = np.asarray(fids)
            for xyz, pid in zip(fp, fids):
                id_to_pts.setdefault(int(pid), []).append(np.asarray(xyz, dtype=np.float64))
                id_to_frames.setdefault(int(pid), []).append(fi)

        identifiers, canon, obs, jitter = [], [], [], []
        for pid, pts in id_to_pts.items():
            arr = np.stack(pts, axis=0)            # (k,3)
            canon.append(arr.mean(axis=0))
            jitter.append(float(arr.std(axis=0).mean()))  # 三轴平均标准差
            obs.append(len(pts))
            identifiers.append(pid)

        if not identifiers:
            return {
                "points": np.zeros((0, 3)),
                "identifiers": np.zeros((0,), dtype=np.int64),
                "observations_per_point": np.zeros((0,), dtype=int),
                "jitter_per_point": np.zeros((0,)),
                "stats": {"n_unique": 0, "scale_confidence": 0.0},
            }

        points = np.stack(canon, axis=0)
        identifiers = np.asarray(identifiers, dtype=np.int64)
        obs = np.asarray(obs, dtype=int)
        jitter = np.asarray(jitter, dtype=float)

        multi = obs >= 2
        stable = jitter < self.jitter_thresh_m
        conf = self._scale_confidence(obs, jitter, multi, stable)

        extent = float(np.linalg.norm(points.max(0) - points.min(0))) if len(points) else 0.0

        stats = {
            "n_unique": int(len(points)),
            "n_multi_frame": int(multi.sum()),
            "n_stable": int((multi & stable).sum()),
            "mean_jitter_m": float(jitter[multi].mean()) if multi.any() else 0.0,
            "max_jitter_m": float(jitter.max()),
            "spatial_extent_m": extent,
            "scale_confidence": float(conf),
        }
        return {
            "points": points,
            "identifiers": identifiers,
            "observations_per_point": obs,
            "jitter_per_point": jitter,
            "stats": stats,
        }

    @staticmethod
    def _scale_confidence(obs, jitter, multi, stable):
        """综合置信度 0~1：多帧跟踪占比 × 稳定占比，并对抖动做软惩罚。"""
        if len(obs) == 0:
            return 0.0
        multi_ratio = multi.mean()
        if multi_ratio <= 0:
            return 0.0
        stable_ratio = (multi & stable).sum() / max(multi.sum(), 1)
        # 抖动惩罚：平均抖动越小越接近 1
        mean_j = jitter[multi].mean() if multi.any() else 1.0
        jitter_penalty = float(np.exp(-mean_j / 0.02))   # 20mm 处 ~0.37，2mm 处 ~0.95
        return float(np.clip(multi_ratio * (0.5 * stable_ratio + 0.5 * jitter_penalty), 0, 1))
