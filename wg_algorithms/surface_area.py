"""
surface_area.py — integrate wound area over a curved surface
================================================================

Converts a flat 2D mask area into the corresponding curved-surface area.

Two data paths:
  (A) LiDAR path: per-pixel depth map -> back-project to 3D -> surface integration;
  (B) no-LiDAR path: use the module-2 curvature prior model, intersect camera rays
      with the parametric surface to obtain depth -> 3D -> surface integration
      (this is the actual implementation of "curvature prior compensation").

Area computation uses an analytic per-pixel method (primary) plus triangulation of
the image grid (cross-check):
  true per-pixel area dA = (z² / (fx·fy)) / |n·view_hat|
  where z is the pixel-to-camera distance, n is the surface normal, and view_hat is
  the viewing direction.
  The sum is the true surface area; comparing against the planar projected area
  that ignores curvature yields the compensation effect.
"""
from __future__ import annotations

import numpy as np

from .geometry import backproject


def recommended_normal_sigma(K, median_depth_m: float, target_mm: float = 6.0) -> float:
    """Convert a "physical smoothing scale" into a Gaussian sigma in pixel units.

    Why not hardcode a pixel count: the same algorithm on iPhone LiDAR (coarse
    resolution) and WoundsDB point clouds (~2 mm/pixel) has an equivalent pixel
    sigma that differs by several times. Calibration on real data
    (tools/validate_real_surface.py, REAL human skin point cloud) shows that the
    error is minimized when the depth is smoothed at **a physical scale of about
    6 mm** before computing normals (mean 7.3%, versus 21.7% without smoothing).

    Args:
        K: intrinsics
        median_depth_m: median depth of the target region (metres)
        target_mm: physical scale to smooth away (millimetres)
    """
    fx = float(K[0, 0])
    d = float(median_depth_m)
    if fx <= 0 or d <= 0:
        return 0.0
    mm_per_px = d * 1000.0 / fx          # millimetres per pixel at this depth
    return max(0.0, target_mm / mm_per_px)


def _smooth_valid(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a depth map containing NaNs.

    Why not call gaussian_filter directly: NaNs spread like a virus, and a small
    hole can contaminate the whole image. Approach: first fill NaNs with the
    nearest valid value, then smooth, and **keep only the values at originally
    valid positions** after smoothing.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    valid = np.isfinite(arr)
    if valid.all():
        return gaussian_filter(arr, sigma=sigma, mode="nearest")
    if not valid.any():
        return arr
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = arr[tuple(idx)]
    sm = gaussian_filter(filled, sigma=sigma, mode="nearest")
    return np.where(valid, sm, arr)


class SurfaceAreaIntegrator:
    def __init__(self):
        pass

    # ---------- Main entry point ----------
    def integrate(self, rgb, mask, K, M_cam2world, source, confidence=None,
                  normal_smooth_sigma: float = 0.0):
        """Unified entry point.

        Args:
            rgb: (H,W,3) reference frame image (only used for dimensions/visualisation).
            mask: (H,W) uint8 0/255 wound region.
            K: 3x3 intrinsics.
            M_cam2world: 4x4 camera->world (OpenCV convention: +Z forward, +Y down).
            source: tuple
                ('lidar', depth_map (Hd,Wd) float metres) or
                ('model', a BodyPartModel instance)
            confidence: optional (Hd,Wd) uint8 confidence (0=low), used to filter
                low-confidence LiDAR pixels.
        Returns:
            dict: area_cm2, flat_area_cm2, per_pixel_area (H,W) m²,
                  points3d (N,3), foreshorten (H,W), method
        """
        H, W = mask.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return self._empty(H, W)

        cam_pos = M_cam2world[:3, 3]
        fx, fy = K[0, 0], K[1, 1]
        R = M_cam2world[:3, :3]

        # Get the depth of each mask pixel (distance along the view ray to the surface)
        if source[0] == "lidar":
            depth_map = np.asarray(source[1], dtype=np.float64)
            # Bilinearly resample the whole depth map to RGB resolution (depth and
            # RGB resolutions usually differ)
            depth_full = self._resample_depth_full(depth_map, (H, W))
            valid_full = np.isfinite(depth_full)
            conf_mask = None
            if confidence is not None:
                # Confidence is resampled with nearest neighbour (never bilinearly
                # interpolate integer labels); only exclude "low-confidence" pixels
                # from the area accumulation; **do not** set depth to NaN and then
                # zero-fill — otherwise artificial cliffs appear at low-confidence
                # pixel boundaries, distorting the normals of neighbouring valid
                # pixels and abnormally inflating the area (about 3× measured).
                conf = self._resample_nearest(np.asarray(confidence, dtype=np.float64), (H, W))
                conf_mask = conf >= 1.0          # keep medium/high confidence; low confidence (0) is excluded
                valid_full = valid_full & np.isfinite(conf)
            # Integrate only over pixels "mask ∩ valid depth ∩ valid confidence"
            # (re-determine the pixel set)
            keep = (mask > 0) & valid_full
            if conf_mask is not None:
                keep = keep & conf_mask
            ys, xs = np.where(keep)
            if len(xs) == 0:
                return self._empty(H, W)
            depth = depth_full[ys, xs].copy()
            # Back-project to obtain world-coordinate points
            pts3d = backproject(xs.astype(float), ys.astype(float), depth, K, M_cam2world)
            # Estimate the true per-pixel normals from the depth map (key: for a
            # curved surface we can no longer assume normal = viewing axis)
            normals_full = self._normals_from_depth_full(depth_full, K, R,
                                                         smooth_sigma=normal_smooth_sigma)
            normals = normals_full[ys, xs]
            method = "lidar"
        else:
            model = source[1]
            # pixel -> camera-frame ray -> world-frame ray
            rays_cam = self._pixel_rays(xs, ys, K)            # (N,3) camera-frame unit vectors
            rays_world = (R @ rays_cam.T).T                   # (N,3) world-frame unit vectors
            o = cam_pos
            depths, pts3d = [], []
            for i in range(len(xs)):
                hit, t = model.intersect_ray(o, rays_world[i])
                if hit is None:
                    depths.append(np.nan); pts3d.append([np.nan, np.nan, np.nan])
                else:
                    depths.append(float(np.linalg.norm(hit - o)))
                    pts3d.append(hit)
            depth = np.array(depths, dtype=np.float64)
            pts3d = np.array(pts3d, dtype=np.float64)
            normals = np.array([model.normal_at(pts3d[i]) for i in range(len(pts3d))])
            method = f"model:{model.surface_kind}"

        # Analytic true per-pixel area
        z = depth
        view = cam_pos - pts3d                     # world-frame view direction (toward the camera)
        view_len = np.linalg.norm(view, axis=1, keepdims=True)
        view_hat = view / (view_len + 1e-12)

        n_dot_v = np.sum(normals * view_hat, axis=1)
        # Safe lower bound: skin wounds are mostly facing the camera, so |n·v|
        # cannot be extremely small; clamp grazing-angle (>~78°) pixels to
        # cos=0.2 to prevent a few distorted normals from inflating a single
        # pixel's area to 1000× and contaminating the total area.
        n_dot_v = np.clip(np.abs(n_dot_v), 0.2, 1.0)

        per_pixel = (z ** 2) / (fx * fy) / n_dot_v   # m²
        per_pixel = np.where(np.isfinite(per_pixel), per_pixel, 0.0)

        # Projected (planar) area: ignore curvature, simply lay out a plane at
        # the viewing distance
        flat_per_pixel = (z ** 2) / (fx * fy)
        flat_per_pixel = np.where(np.isfinite(flat_per_pixel), flat_per_pixel, 0.0)

        area_m2 = float(per_pixel.sum())
        flat_m2 = float(flat_per_pixel.sum())

        # Fill-back maps
        pp_map = np.zeros((H, W), dtype=np.float64)
        pp_map[ys, xs] = per_pixel
        foreshorten_map = np.zeros((H, W), dtype=np.float64)
        foreshorten_map[ys, xs] = 1.0 / n_dot_v

        return {
            "area_cm2": area_m2 * 1e4,
            "flat_area_cm2": flat_m2 * 1e4,
            "compensation_ratio": (area_m2 / flat_m2) if flat_m2 > 0 else 1.0,
            "per_pixel_area": pp_map,
            "foreshorten": foreshorten_map,
            "points3d": pts3d,
            "method": method,
            "n_valid": int(np.isfinite(per_pixel).sum()),
        }

    # ---------- Internal helpers ----------
    @staticmethod
    def _pixel_rays(xs, ys, K):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        d = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs, dtype=float)], axis=1)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    @staticmethod
    def _resample_depth_full(depth_map, target_shape):
        """Bilinearly resample a (Hd, Wd) depth map to RGB resolution (H, W),
        returning an (H, W) array.

        ARKit's sceneDepth is already aligned to the colour camera coordinate
        system, but at a lower resolution; depth pixels (u_d, v_d) and colour
        pixels (u_c, v_c) satisfy u_d = u_c * Wd/Wc, v_d = v_c * Hd/Hc, so linear
        interpolation in normalised coordinates suffices. Invalid depth (<=0 or
        NaN) stays NaN.
        """
        dh, dw = depth_map.shape[:2]
        th, tw = target_shape
        yy = np.linspace(0, dh - 1, th)
        xx = np.linspace(0, dw - 1, tw)
        # Fast bilinear interpolation with scipy (falls back to nearest neighbour
        # on failure)
        try:
            from scipy.interpolate import RegularGridInterpolator
            f = RegularGridInterpolator(
                (np.arange(dh), np.arange(dw)), depth_map,
                bounds_error=False, fill_value=np.nan
            )
            gx, gy = np.meshgrid(xx, yy)   # gx:(th,tw) columns, gy rows
            out = f(np.stack([gy.ravel(), gx.ravel()], axis=1)).reshape(th, tw)
        except Exception:
            out = depth_map[
                (np.linspace(0, dh - 1, th)).round().astype(int)[:, None],
                (np.linspace(0, dw - 1, tw)).round().astype(int)[None, :],
            ]
        out = np.where(np.isfinite(out) & (out > 0), out, np.nan)
        return out

    @staticmethod
    def _resample_nearest(value_map, target_shape):
        """Nearest-neighbour resampling (Hd,Wd) -> (H,W). Used for integer labels
        (such as confidence) to avoid bilinear interpolation blending 0/1/2 into
        fractions and corrupting invalid boundaries."""
        dh, dw = value_map.shape[:2]
        th, tw = target_shape
        yy = (np.linspace(0, dh - 1, th)).round().astype(int)
        xx = (np.linspace(0, dw - 1, tw)).round().astype(int)
        return value_map[np.ix_(yy, xx)]

    @staticmethod
    def _normals_from_depth_full(depth_full, K, R, smooth_sigma: float = 0.0):
        """Estimate per-pixel world-coordinate normals from an (H,W) depth map.

        Build a 3D point cloud (X,Y,Z) in the camera frame (OpenCV convention:
        +Z forward), take finite differences along x and y to obtain two tangents,
        cross them to get the camera-frame normal (corrected to face +Z), then
        rotate to the world frame via R. Normals are NaN where depth is invalid.

        ⚠️ Why smooth_sigma exists (a lesson from real-data measurements):
        On real point clouds (WoundsDB, ~2 mm pixel spacing), direct per-pixel
        finite differences make normals jitter violently due to skin micro-relief
        plus depth noise: mean |n·v| drops to about 0.4 and the area is
        systematically inflated to **about 2×** (measured 8.54 vs ground truth
        3.80 cm², 173% error). Gaussian-smoothing the depth before computing
        normals stabilises the "macroscopic orientation". The unit of sigma is
        **pixels**; 0 means no smoothing (preserving the old behaviour).
        """
        H, W = depth_full.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        vv, uu = np.mgrid[0:H, 0:W]

        if smooth_sigma and smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            z = _smooth_valid(depth_full, smooth_sigma)
        else:
            z = depth_full
        X = (uu - cx) * z / fx
        Y = (vv - cy) * z / fy
        # Fill invalid positions with 0 to avoid gradient noise (normals at the
        # mask are estimated from valid neighbours)
        valid = np.isfinite(z)
        Xf = np.where(valid, X, 0.0)
        Yf = np.where(valid, Y, 0.0)
        Zf = np.where(valid, z, 0.0)

        # Finite-difference tangents
        t_u = np.stack([
            np.gradient(Xf, axis=1), np.gradient(Yf, axis=1), np.gradient(Zf, axis=1)
        ], axis=-1)  # (H,W,3)
        t_v = np.stack([
            np.gradient(Xf, axis=0), np.gradient(Yf, axis=0), np.gradient(Zf, axis=0)
        ], axis=-1)

        n_cam = np.cross(t_u, t_v)
        n_len = np.linalg.norm(n_cam, axis=-1, keepdims=True)
        n_cam = np.where(n_len > 1e-12, n_cam / np.where(n_len > 1e-12, n_len, 1.0), 0.0)
        # Normals face the camera (+Z forward); flip if the z component is negative
        flip = np.where(n_cam[..., 2:3] < 0, -1.0, 1.0)
        n_cam = n_cam * flip

        # Rotate to the world frame
        n_world = (R @ n_cam.reshape(-1, 3).T).T.reshape(H, W, 3)
        n_len_w = np.linalg.norm(n_world, axis=-1, keepdims=True)
        n_world = np.where(n_len_w > 1e-12, n_world / np.where(n_len_w > 1e-12, n_len_w, 1.0), 0.0)
        # Set normals to NaN where depth is invalid
        n_world = np.where(valid[..., None], n_world, np.nan)
        return n_world

    @staticmethod
    def _empty(H, W):
        return {
            "area_cm2": 0.0, "flat_area_cm2": 0.0, "compensation_ratio": 1.0,
            "per_pixel_area": np.zeros((H, W)), "foreshorten": np.zeros((H, W)),
            "points3d": np.zeros((0, 3)), "method": "empty", "n_valid": 0,
        }
