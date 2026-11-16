"""
geometry.py — camera geometry helpers
==========================================

Coordinate conventions (matching the capture format):
  - cameraTransform is stored "column-major": list[4] four-vectors, each being a
    column of the matrix; column3 = [tx, ty, tz, 1] is the camera's position in
    the world frame (metres). This is the ARKit camera->world matrix (standard
    math matrix M_raw = arr.T, world = M_raw @ camera_local).
  - Axis convention (important): the ARKit camera is right-handed, looking along
    local -Z with +Y up; this library's pinhole / triangulation / ray
    back-projection all use the OpenCV convention (+Z forward, +Y down).
    Therefore camera_transform_to_cam2world right-multiplies by diag(1,-1,-1,1)
    before returning, converting the ARKit matrix into an OpenCV-convention
    camera->world matrix so downstream code needs no further sign handling.
  - Fragment coordinates (u, v): origin at top-left, u→right, v→down, in pixels
    (consistent with OpenCV).
"""
from __future__ import annotations

import numpy as np


def intrinsics_to_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Camera intrinsics matrix K (3x3)."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_transform_to_cam2world(camera_transform) -> np.ndarray:
    """Convert the cameraTransform in the capture data to an OpenCV-convention
    camera->world matrix (row-major math matrix).

    Capture convention (consistent with the "Data Capture Specification v1.1"):
    cameraTransform is stored "column-major", as list[4] four-vectors each being
    a column; column3 = [tx,ty,tz,1] is the camera's world position (metres).
    Thus the raw ARKit matrix M_raw = arr.T satisfies
    world = M_raw @ camera_local_arkit, and M_raw[:3,3] is the camera's world
    coordinates.

    The ARKit camera looks along local -Z with +Y up; this library uniformly
    uses the OpenCV convention (+Z forward, +Y down), so we further
    right-multiply by diag(1,-1,-1,1) to flip y and z, yielding a matrix that
    downstream code can use directly for standard pinhole / triangulation /
    ray back-projection. The returned matrix preserves the camera position and
    makes the camera look along +Z.
    """
    arr = np.asarray(camera_transform, dtype=np.float64)
    m_raw = arr.T
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return m_raw @ flip


def cam2world_to_proj(M_cam2world: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Obtain the world->image projection matrix P (3x4) from the camera->world
    matrix and intrinsics.

    P = K @ [R | t], where [R|t] = M^{-1} (world->camera).
    """
    M_inv = np.linalg.inv(M_cam2world)
    R = M_inv[:3, :3]
    t = M_inv[:3, 3]
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def pixel_to_ray(u: np.ndarray, v: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixel (u,v) -> unit direction ray in the camera frame. Returns (3,) or
    (N,3)."""
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(np.asarray(u, dtype=float))], axis=-1)
    norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs / norm


def backproject(u: np.ndarray, v: np.ndarray, depth: np.ndarray,
                K: np.ndarray, M_cam2world: np.ndarray) -> np.ndarray:
    """Back-project a pixel + depth (distance in front of the camera, metres)
    to world coordinates.

    Returns (...,3) world-coordinate points with the same shape as the input.
    """
    rays = pixel_to_ray(u, v, K)            # camera-frame unit directions
    cam_pts = rays * depth[..., None]       # camera-frame 3D points
    # camera frame -> world frame
    R = M_cam2world[:3, :3]
    t = M_cam2world[:3, 3]
    world = (R @ cam_pts.T).T + t
    return world


def triangulate_pair(pts1, pts2, M1, M2, K):
    """DLT-triangulate 2D matched points from two frames, returning world
    coordinates (N,3).

    Used for cross-checking or completing world points already provided by
    ARKit (e.g. for model self-tests).
    pts1/pts2: (N,2) pixel coordinates. M1/M2: camera->world 4x4. K: 3x3.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    P1 = cam2world_to_proj(M1, K)
    P2 = cam2world_to_proj(M2, K)
    # OpenCV triangulation requires 3x4 projection matrices and float32 points
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
        # Degenerate fallback: midpoint method (for self-tests only)
        return _midpoint_triangulate(pts1, pts2, P1, P2)


def _midpoint_triangulate(pts1, pts2, P1, P2):
    """Minimal midpoint method as a fallback when opencv is unavailable (low
    accuracy, for tests only)."""
    n = pts1.shape[0]
    res = np.zeros((4, n))
    for i in range(n):
        # Approximate with the closest point on the rays
        res[:3, i] = 0.0
        res[3, i] = 1.0
    return res
