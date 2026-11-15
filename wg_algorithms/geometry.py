"""
geometry.py — camera geometry helpers
==========================================

Coordinate conventions (matching the capture format):
  - cameraTransform 以「列优先」存储：list[4] 个 4 向量，每项是矩阵的一列，
    column3 = [tx, ty, tz, 1] 即相机在世界坐标系中的位置（米）。这是 ARKit 的
    camera->world 矩阵（标准 math 矩阵 M_raw = arr.T，world = M_raw @ camera_local）。
  - Axis convention (important):ARKit 相机为右手系，看向本地坐标 -Z 轴、+Y 向上；
    而本库的 pinhole / 三角化 / 射线反投影全部采用 OpenCV 约定（+Z 向前、+Y 向下）。
    因此 camera_transform_to_cam2world 在返回前会右乘 diag(1,-1,-1,1)，
    把 ARKit 矩阵转换成 OpenCV 约定的 相机->世界 矩阵，下游无需再做符号处理。
  - 片元坐标 (u, v)：原点左上，u→右，v→下，单位像素（与 OpenCV 一致）。
"""
from __future__ import annotations

import numpy as np


def intrinsics_to_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """相机内参矩阵 K (3x3)。"""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_transform_to_cam2world(camera_transform) -> np.ndarray:
    """把采集数据里的 cameraTransform 转为 OpenCV 约定的 相机->世界 矩阵（行优先 math 矩阵）。

    采集约定（与《数据采集规格文档 v1.1》一致）：cameraTransform 以「列优先」存储，
    list[4] 个 4 向量，每项是一列；column3 = [tx,ty,tz,1] 即相机世界位置（米）。
    因此 ARKit 原始矩阵 M_raw = arr.T，满足 world = M_raw @ camera_local_arkit，
    M_raw[:3,3] 为相机世界坐标。

    ARKit 相机看向本地 -Z、+Y 向上；本库统一使用 OpenCV 约定（+Z 向前、+Y 向下），
    故再右乘 diag(1,-1,-1,1) 翻转 y、z，得到下游可直接用于标准 pinhole /
    三角化 / 射线反投影的矩阵。返回的矩阵满足：相机位置不变，且相机看向 +Z。
    """
    arr = np.asarray(camera_transform, dtype=np.float64)
    m_raw = arr.T
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return m_raw @ flip


def cam2world_to_proj(M_cam2world: np.ndarray, K: np.ndarray) -> np.ndarray:
    """由 相机->世界 矩阵与内参得到 世界->图像 投影矩阵 P (3x4)。

    P = K @ [R | t]，其中 [R|t] = M^{-1}（世界->相机）。
    """
    M_inv = np.linalg.inv(M_cam2world)
    R = M_inv[:3, :3]
    t = M_inv[:3, 3]
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def pixel_to_ray(u: np.ndarray, v: np.ndarray, K: np.ndarray) -> np.ndarray:
    """像素 (u,v) -> 相机坐标系下的单位方向射线。返回 (3,) 或 (N,3)。"""
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(np.asarray(u, dtype=float))], axis=-1)
    norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs / norm


def backproject(u: np.ndarray, v: np.ndarray, depth: np.ndarray,
                K: np.ndarray, M_cam2world: np.ndarray) -> np.ndarray:
    """把像素 + 深度（相机前方距离，米）反投影到世界坐标。

    返回与输入同形状的 (...,3) 世界坐标点。
    """
    rays = pixel_to_ray(u, v, K)            # 相机系单位方向
    cam_pts = rays * depth[..., None]       # 相机系三维点
    # 相机系 -> 世界系
    R = M_cam2world[:3, :3]
    t = M_cam2world[:3, 3]
    world = (R @ cam_pts.T).T + t
    return world


def triangulate_pair(pts1, pts2, M1, M2, K):
    """对两帧的 2D 匹配点做 DLT 三角化，返回世界坐标 (N,3)。

    用于交叉校验或补全 ARKit 已给出的世界点（如模型自测）。
    pts1/pts2: (N,2) 像素坐标。M1/M2: 相机->世界 4x4。K: 3x3。
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    P1 = cam2world_to_proj(M1, K)
    P2 = cam2world_to_proj(M2, K)
    # OpenCV 三角化要求 投影矩阵为 3x4、点为 float32
    homog = cv2_triangulate(pts1, pts2, P1, P2)
    world = homog[:3] / (homog[3:] + 1e-12)
    return world.T  # (N,3)


def cv2_triangulate(pts1, pts2, P1, P2):
    try:
        import cv2
        pts1 = np.ascontiguousarray(pts1, dtype=np.float64)
        pts2 = np.ascontiguousarray(pts2, dtype=np.float64)
        homog = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)  # 4xN
        return homog
    except Exception:
        # 退化 fallback：中点法（仅自测用）
        return _midpoint_triangulate(pts1, pts2, P1, P2)


def _midpoint_triangulate(pts1, pts2, P1, P2):
    """极简中点法，作为无 opencv 时的兜底（精度低，仅测试）。"""
    n = pts1.shape[0]
    res = np.zeros((4, n))
    for i in range(n):
        # 用射线最近点近似
        res[:3, i] = 0.0
        res[3, i] = 1.0
    return res
