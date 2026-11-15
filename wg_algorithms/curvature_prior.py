"""
curvature_prior.py — anatomical curvature prior for sparse depth
================================================

Rationale:
  消费级手机在无 LiDAR 时只有稀疏点云，缺乏逐像素深度。本模块用「人体
  解剖部位统计曲率先验」把伤口区域所在的体表建模为参数化曲面
  （平面 / 圆柱 / 球面），从而：
    (a) 为任意像素提供深度（相机射线与参数曲面求交）—— 替代 LiDAR；
    (b) 为每个表面点提供法向，进而给出「投影面积 -> 真实曲面面积」的
        压缩补偿因子 1/|n·view|。

典型部位预设（半径单位：米）：
  limb_forearm  前臂   R≈0.040
  limb_thigh    大腿   R≈0.070
  heel          足跟   R≈0.030 (球面近似)
  sacrum        骶尾   平面近似
"""
from __future__ import annotations

import numpy as np

# 解剖部位预设： (model_type, default_radius_m)
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
    """参数化体表曲面基类。"""

    def __init__(self):
        self.surface_kind = "base"

    def normal_at(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def intersect_ray(self, o: np.ndarray, d: np.ndarray):
        """返回 (hit_world_point (3,), t>0) 或 (None, None)。"""
        raise NotImplementedError

    def foreshorten_factor(self, p: np.ndarray, cam_pos: np.ndarray) -> float:
        """真实面积 / 投影面积 = 1/|n·view_hat|（夹角的余弦倒数）。

        当表面正对相机时 n·view≈1（无压缩）；斜视或曲面边缘 n·view 变小，
        投影被压缩，需补偿放大。
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
                # 仅取外表面（径向向外）
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
    """由稀疏度量点云（+ARKit 平面）拟合部位参数化模型。

    Args:
        points: (N,3) 全局度量点云（米）。
        planes: 可选，ARKit 平面列表（用于平面类部位）。
        part_type: 部位类型键（见 PART_PRESETS）。
    Returns:
        BodyPartModel 实例。
    """
    model_type, default_R = PART_PRESETS.get(part_type, ("cylinder", 0.04))
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        # 退化：用预设构造一个以原点为中心的默认模型
        if model_type == "plane":
            return PlaneModel([0, 0, 0], [0, 0, 1])
        if model_type == "sphere":
            return SphereModel([0, 0, 0], default_R)
        return CylinderModel([0, 0, 0], [0, 1, 0], default_R)

    centroid = pts.mean(axis=0)
    # PCA 主方向
    cov = np.cov(pts.T)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    axis = eigvec[:, order[0]]  # 最大方差方向

    if model_type == "plane":
        normal = eigvec[:, order[-1]]  # 最小方差方向 = 法向
        return PlaneModel(centroid, normal)

    if model_type == "sphere":
        R = float(np.median(np.linalg.norm(pts - centroid, axis=1)))
        R = R if 0.005 < R < 0.3 else default_R
        return SphereModel(centroid, R)

    # cylinder（默认）
    # 点到轴线（过 centroid 沿 axis）的垂直距离中位数作为半径
    perp = pts - centroid
    perp_dist = np.linalg.norm(perp - np.outer(perp @ axis, axis), axis=1)
    R = float(np.median(perp_dist))
    R = R if 0.01 < R < 0.2 else default_R
    return CylinderModel(centroid, axis, R)
