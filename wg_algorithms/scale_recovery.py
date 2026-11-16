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
            jitter_thresh_m: per-point cross-frame position jitter threshold
                (metres); below this value tracking is considered stable.
        """
        self.jitter_thresh_m = jitter_thresh_m

    def reconstruct(self, frames):
        """Take a list of frames loaded by wg_ar_loader and output the aggregated
        result.

        Returns:
            dict: {
                "points": (N,3) global metric point cloud (metres),
                "identifiers": (N,) int64,
                "observations_per_point": (N,) number of frames observing each
                    point,
                "jitter_per_point": (N,) cross-frame position standard deviation
                    of each point (metres),
                "stats": {...} summary metrics,
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
            jitter.append(float(arr.std(axis=0).mean()))  # mean standard deviation over the three axes
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
        """Aggregate confidence 0~1: multi-frame tracking ratio × stable ratio,
        with a soft penalty for jitter."""
        if len(obs) == 0:
            return 0.0
        multi_ratio = multi.mean()
        if multi_ratio <= 0:
            return 0.0
        stable_ratio = (multi & stable).sum() / max(multi.sum(), 1)
        # Jitter penalty: closer to 1 when mean jitter is smaller
        mean_j = jitter[multi].mean() if multi.any() else 1.0
        jitter_penalty = float(np.exp(-mean_j / 0.02))   # ~0.37 at 20 mm, ~0.95 at 2 mm
        return float(np.clip(multi_ratio * (0.5 * stable_ratio + 0.5 * jitter_penalty), 0, 1))
