"""
test_core.py — cross-checks for the core geometry and area-integration maths
============================================
Cross-checks the core maths against analytic geometry:

  T1 Planar area: analytic area vs integrator result (error should be < 2%)
  T2 Curvature compensation: cylinder surface true area / frontal projection area > 1 (curvature makes the true area larger)
  T3 VIO triangulation: a synthetic world point is projected through two cameras and then triangulated, recovery error < 1mm
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wg_algorithms.geometry import (  # noqa: E402
    intrinsics_to_K, camera_transform_to_cam2world, cam2world_to_proj, triangulate_pair,
)
from wg_algorithms.curvature_prior import PlaneModel, CylinderModel  # noqa: E402
from wg_algorithms.surface_area import SurfaceAreaIntegrator  # noqa: E402


def _make_mask(H, W, cx, cy, r_px):
    yy, xx = np.mgrid[0:H, 0:W]
    return (((xx - cx) / r_px) ** 2 + ((yy - cy) / r_px) ** 2 <= 1).astype(np.uint8) * 255


def test_plane_area():
    H, W = 1000, 1000
    fx = fy = 1500.0
    cx, cy = W / 2, H / 2
    K = intrinsics_to_K(fx, fy, cx, cy)
    D = 0.30
    r_px = 200
    mask = _make_mask(H, W, int(cx), int(cy), r_px)
    M = np.eye(4)
    plane = PlaneModel([0, 0, D], [0, 0, 1])
    integ = SurfaceAreaIntegrator()
    res = integ.integrate(np.zeros((H, W, 3), np.uint8), mask, K, M, ("model", plane))
    # Analytic: frontal circular spot radius = r_px * D / fx
    r_phys = r_px * D / fx
    analytic = np.pi * r_phys ** 2
    err = abs(res["area_cm2"] - analytic * 1e4) / (analytic * 1e4)
    print(f"[T1] Planar area  integrated={res['area_cm2']:.3f}cm²  analytic={analytic*1e4:.3f}cm²  "
          f"error={err*100:.3f}%  -> {'PASS' if err < 0.02 else 'FAIL'}")
    assert err < 0.02, 'planar area error exceeds 2%'


def test_curvature_compensation():
    H, W = 1500, 1500
    fx = fy = 1500.0
    cx, cy = W / 2, H / 2
    K = intrinsics_to_K(fx, fy, cx, cy)
    D, R = 0.30, 0.05
    r_px = 120
    mask = _make_mask(H, W, int(cx), int(cy), r_px)
    M = np.eye(4)
    plane = PlaneModel([0, 0, D], [0, 0, 1])
    cyl = CylinderModel([0, 0, D + R], [1, 0, 0], R)
    integ = SurfaceAreaIntegrator()
    flat = integ.integrate(np.zeros((H, W, 3), np.uint8), mask, K, M, ("model", plane))
    curved = integ.integrate(np.zeros((H, W, 3), np.uint8), mask, K, M, ("model", cyl))
    ratio = curved["area_cm2"] / flat["area_cm2"]
    ok = 1.0 < ratio < 1.5
    print(f"[T2] Curvature compensation  planar={flat['area_cm2']:.3f}cm²  surface={curved['area_cm2']:.3f}cm²  "
          f"compensation ratio={ratio:.3f}×  -> {'PASS' if ok else 'FAIL'}")
    assert ok, 'curvature compensation ratio outside (1.0, 1.5)'


def test_triangulation():
    fx = fy = 1500.0
    K = intrinsics_to_K(fx, fy, 500, 400)
    # Two cameras: the origin and one translated 0.1m along X
    M1 = np.eye(4)
    M2 = np.eye(4); M2[0, 3] = 0.10
    P1 = cam2world_to_proj(M1, K); P2 = cam2world_to_proj(M2, K)
    # Synthetic world point
    X = np.array([0.02, -0.03, 0.50])
    x1 = (P1 @ np.append(X, 1))[:2] / (P1 @ np.append(X, 1))[2]
    x2 = (P2 @ np.append(X, 1))[:2] / (P2 @ np.append(X, 1))[2]
    rec = triangulate_pair(x1.reshape(1, 2), x2.reshape(1, 2), M1, M2, K)[0]
    err = np.linalg.norm(rec - X)
    print(f"[T3] VIO triangulation  synthetic point={X}  recovered={rec}  error={err*1000:.3f}mm  "
          f"-> {'PASS' if err < 0.001 else 'FAIL'}")
    assert err < 0.001, 'triangulation error exceeds 1 mm'


def test_lidar_depth_cylinder():
    """T4 LiDAR depth path verification: synthesize a scene of "camera looking along +Z + a cylinder (limb) whose axis is along X",
    generate a low-resolution depth map + a rectangular wound patch mask on the cylinder, analytic ground truth area = R·Δθ·Δx.
    Verify that the depth path (back-projection + depth normal estimation + resampling) can recover the true surface area."""
    H, W = 480, 640
    fx = fy = 800.0
    cx, cy = W / 2, H / 2
    K = intrinsics_to_K(fx, fy, cx, cy)
    R = 0.04            # limb radius 4cm
    d = 0.30           # distance from the cylinder's nearest point to the camera
    cc = d + R         # cylinder center z
    M = np.eye(4)

    # Wound patch (a rectangular patch on the cylinder), analytic ground truth area = R·Δθ·Δx (m²)
    th1, th2 = -0.30, 0.30
    x1, x2 = -0.04, 0.04
    gt_area_m2 = R * (th2 - th1) * (x2 - x1)

    # ── Generate a low-resolution depth map (to test the resampling path), depth grid (hd, wd) ──
    hd, wd = H // 2, W // 2
    vv, uu = np.mgrid[0:hd, 0:wd]
    u_c = uu * (W - 1) / (wd - 1)     # depth pixel -> color pixel coordinates
    v_c = vv * (H - 1) / (hd - 1)
    dx = (u_c - cx) / fx
    dy = (v_c - cy) / fy
    dz = np.ones_like(dx)
    norm = np.sqrt(dx**2 + dy**2 + 1.0)
    dx, dy, dz = dx / norm, dy / norm, dz / norm   # unit viewing direction
    a = dy**2 + dz**2
    b = -2 * dz * cc
    c = cc**2 - R**2
    disc = b**2 - 4 * a * c
    valid = disc >= 0
    sq = np.sqrt(np.where(valid, disc, 0.0))
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    t = np.where((t1 > 0) & (t1 < t2), t1, t2)   # near surface (front side)
    depth = np.where(valid, t, np.nan)            # (hd, wd) distance along the viewing ray (meters)

    # ── Generate a color-resolution wound patch mask + verify hits ──
    vv2, uu2 = np.mgrid[0:H, 0:W]
    dx2 = (uu2 - cx) / fx
    dy2 = (vv2 - cy) / fy
    dz2 = np.ones_like(dx2)
    n2 = np.sqrt(dx2**2 + dy2**2 + 1.0)
    dx2, dy2, dz2 = dx2 / n2, dy2 / n2, dz2 / n2
    a2 = dy2**2 + dz2**2
    b2 = -2 * dz2 * cc
    c2 = cc**2 - R**2
    disc2 = np.maximum(b2**2 - 4 * a2 * c2, 0.0)
    sq2 = np.sqrt(disc2)
    t1b = (-b2 - sq2) / (2 * a2)
    t2b = (-b2 + sq2) / (2 * a2)
    tb = np.where((t1b > 0) & (t1b < t2b), t1b, t2b)
    Px = tb * dx2; Py = tb * dy2; Pz = tb * dz2
    theta = np.arctan2(Py, (cc - Pz))     # polar angle of the hit point
    mask = np.zeros((H, W), np.uint8)
    inside = (disc2 > 0) & (theta >= th1) & (theta <= th2) & (Px >= x1) & (Px <= x2)
    mask[inside] = 255

    integ = SurfaceAreaIntegrator()
    res = integ.integrate(np.zeros((H, W, 3), np.uint8), mask, K, M,
                          ("lidar", depth), confidence=None)
    got = res["area_cm2"] * 1e-4     # m²
    err = abs(got - gt_area_m2) / gt_area_m2
    print(f"[T4] LiDAR depth path  integrated={res['area_cm2']:.3f}cm²  analytic ground truth={gt_area_m2*1e4:.3f}cm²  "
          f"error={err*100:.3f}%  valid pixels={res['n_valid']}  "
          f"-> {'PASS' if err < 0.05 else 'FAIL'}")
    assert err < 0.05, 'LiDAR depth path error exceeds 5%'


if __name__ == "__main__":
    test_plane_area()
    test_curvature_compensation()
    test_triangulation()
    test_lidar_depth_cylinder()
    print("\nResult: ALL PASS")
    sys.exit(0)
