"""
curvature_prior.py — anatomical curvature prior for sparse depth
================================================

Rationale:
  Consumer phones without LiDAR have only sparse point clouds, lacking per-pixel
  depth. This module uses "statistical curvature priors of human anatomical parts"
  to model the body surface containing the wound region as a parametric surface
  (plane / cylinder / sphere), thereby:
    (a) providing depth for any pixel (intersecting the camera ray with the
        parametric surface) — replacing LiDAR;
    (b) providing a normal for each surface point, and hence the compression
        compensation factor 1/|n·view| for "projected area -> true surface area".

Typical part presets (radius in metres):
  limb_forearm  forearm   R≈0.040
  limb_thigh    thigh     R≈0.070
  heel          heel      R≈0.030 (spherical approximation)
  sacrum        sacrum    planar approximation
"""
from __future__ import annotations

import numpy as np

# Anatomical part presets: (model_type, default_radius_m)
PART_PRESETS = {
    "flat":        ("plane", 0.0),
    "sacrum":      ("plane", 0.0),
    "limb_forearm":("cylinder", 0.040),
    "limb_thigh":  ("cylinder", 0.070),
    "limb_upper":  ("cylinder", 0.050),
    "heel":        ("sphere", 0.030),
    "face":        ("sphere", 0.080),
}


class BodyPartModel:
    """Base class for parametric body-surface models."""

    def __init__(self):
        self.surface_kind = "base"

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def intersect_ray(self, o: np.ndarray, d: np.ndarray):
        """Return (hit_world_point (3,), t>0) or (None, None)."""
        raise NotImplementedError

    def foreshorten_factor(self, p: np.ndarray, cam_pos: np.ndarray) -> float:
        """True area / projected area = 1/|n·view_hat| (reciprocal of the cosine
        of the angle).

        When the surface faces the camera, n·view≈1 (no compression); for oblique
        views or at surface edges n·view becomes smaller, the projection is
        compressed and must be compensated by enlargement.
        """
        n = self.normal_at(p)
        view = cam_pos - p
        view = view / (np.linalg.norm(view) + 1e-12)
        c = float(np.clip(np.abs(np.dot(n, view)), 1e-3, 1.0))
        return 1.0 / c


class PlaneModel(BodyPartModel):
    def __init__(self, center, normal):
        super().__init__()
        self.surface_kind = "plane"
        self.center = np.asarray(center, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal) + 1e-12

    def normal_at(self, p=None):
        return self.normal.copy()

    def intersect_ray(self, o, d):
        d = np.asarray(d, dtype=float); o = np.asarray(o, dtype=float)
        denom = np.dot(self.normal, d)
        if abs(denom) < 1e-9:
            return None, None
        t = np.dot(self.center - o, self.normal) / denom
        if t <= 0:
            return None, None
        return o + t * d, float(t)


class CylinderModel(BodyPartModel):
    def __init__(self, axis_point, axis_dir, radius):
        super().__init__()
        self.surface_kind = "cylinder"
        self.axis_point = np.asarray(axis_point, dtype=float)
        self.axis_dir = np.asarray(axis_dir, dtype=float)
        self.axis_dir /= np.linalg.norm(self.axis_dir) + 1e-12
        self.radius = float(radius)

    def _perp(self, v):
        return v - np.dot(v, self.axis_dir) * self.axis_dir

    def normal_at(self, p):
        p = np.asarray(p, dtype=float)
        radial = self._perp(p - self.axis_point)
        n = radial / (np.linalg.norm(radial) + 1e-12)
        return n

    def intersect_ray(self, o, d):
        o = np.asarray(o, dtype=float); d = np.asarray(d, dtype=float)
        W = o - self.axis_point
        Wp = self._perp(W); Dp = self._perp(d)
        a = float(np.dot(Dp, Dp))
        b = float(np.dot(Wp, Dp))
        c = float(np.dot(Wp, Wp) - self.radius ** 2)
        if a < 1e-12:
            return None, None
        disc = b * b - a * c
        if disc < 0:
            return None, None
        sq = np.sqrt(disc)
        for t in (sorted([(-b - sq) / a, (-b + sq) / a])):
            if t > 1e-6:
                hit = o + t * d
                # Only take the outer surface (radially outward)
                if np.dot(self._perp(hit - self.axis_point), self._perp(d)) < 0:
                    return hit, float(t)
        return None, None


class SphereModel(BodyPartModel):
    def __init__(self, center, radius):
        super().__init__()
        self.surface_kind = "sphere"
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def normal_at(self, p):
        p = np.asarray(p, dtype=float)
        n = p - self.center
        return n / (np.linalg.norm(n) + 1e-12)

    def intersect_ray(self, o, d):
        o = np.asarray(o, dtype=float); d = np.asarray(d, dtype=float)
        oc = o - self.center
        b = float(np.dot(oc, d))
        c = float(np.dot(oc, oc) - self.radius ** 2)
        disc = b * b - c
        if disc < 0:
            return None, None
        sq = np.sqrt(disc)
        for t in (sorted([(-b - sq), (-b + sq)])):
            if t > 1e-6:
                return o + t * d, float(t)
        return None, None


def fit_model_to_cloud(points: np.ndarray, planes=None, part_type: str = "limb_forearm"):
    """Fit an anatomical parametric model from a sparse metric point cloud
    (+ARKit planes).

    Args:
        points: (N,3) global metric point cloud (metres).
        planes: optional, list of ARKit planes (for planar parts).
        part_type: part-type key (see PART_PRESETS).
    Returns:
        a BodyPartModel instance.
    """
    model_type, default_R = PART_PRESETS.get(part_type, ("cylinder", 0.04))
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        # Degenerate case: build a default model centred at the origin using
        # the preset
        if model_type == "plane":
            return PlaneModel([0, 0, 0], [0, 0, 1])
        if model_type == "sphere":
            return SphereModel([0, 0, 0], default_R)
        return CylinderModel([0, 0, 0], [0, 1, 0], default_R)

    centroid = pts.mean(axis=0)
    # PCA principal direction
    cov = np.cov(pts.T)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    axis = eigvec[:, order[0]]  # direction of maximum variance

    if model_type == "plane":
        normal = eigvec[:, order[-1]]  # direction of minimum variance = normal
        return PlaneModel(centroid, normal)

    if model_type == "sphere":
        R = float(np.median(np.linalg.norm(pts - centroid, axis=1)))
        R = R if 0.005 < R < 0.3 else default_R
        return SphereModel(centroid, R)

    # cylinder (default)
    # Use the median perpendicular distance from the points to the axis
    # (through the centroid along axis) as the radius
    perp = pts - centroid
    perp_dist = np.linalg.norm(perp - np.outer(perp @ axis, axis), axis=1)
    R = float(np.median(perp_dist))
    R = R if 0.01 < R < 0.2 else default_R
    return CylinderModel(centroid, axis, R)
