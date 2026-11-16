"""
pipeline.py — end-to-end measurement pipeline and controlled curvature experiment
=================================================

Step 1: capture data loading (wg_ar_loader)
Step 2: wound segmentation (first half of module 3, pluggable)
Step 3: scale recovery (module 1 -> metric point cloud + confidence)
Step 4: curvature prior modelling (module 2 -> anatomical parametric surface)
Step 5: 3D surface integration (second half of module 3 -> true area)
Step 6: multi-frame fusion (auxiliary -> median/mean over multiple reference frames)
"""
from __future__ import annotations

import os
import sys
import numpy as np

# Allow running directly as a script
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from wg_algorithms import (  # noqa: E402
    camera_transform_to_cam2world, intrinsics_to_K,
    MetricReconstructor, fit_model_to_cloud,
    ColorThresholdSegmenter, SyntheticWoundSegmenter, WoundSegModel,
    SurfaceAreaIntegrator,
)
from wg_algorithms.surface_area import recommended_normal_sigma  # noqa: E402


def _auto_normal_sigma(K, depth, mask, target_mm: float = 6.0) -> float:
    """Automatically compute the normal-smoothing sigma from the current frame's
    actual depth.

    Hardcoded pixel values distort results across devices: the point spacing of
    iPhone LiDAR and WoundsDB point clouds differs by several times. Here we
    uniformly convert to "a physical scale of about target_mm millimetres".
    """
    if depth is None or mask is None:
        return 0.0
    d = np.asarray(depth, dtype=np.float64)
    m = np.asarray(mask) > 0
    if d.ndim == 2 and m.shape[:2] == d.shape:
        vals = d[m & np.isfinite(d)]
    else:
        vals = d[np.isfinite(d)]
    if len(vals) == 0:
        return 0.0
    return recommended_normal_sigma(K, float(np.median(vals)), target_mm)


def _load_frames(session_dir, loader_mod, max_frames=None):
    loader = loader_mod.ARCaptureLoader(session_dir)
    n = loader.frame_count()
    if max_frames:
        n = min(n, max_frames)
    return [loader.load_frame(i) for i in range(n)], loader.meta


def _n_points(fr):
    fp = fr.get("feature_points")
    return 0 if fp is None else int(len(fp))


def _best_reference_frame(frames):
    """Select the frame with the most feature points as the reference frame
    (best tracking quality)."""
    best_i, best_n = 0, -1
    for i, fr in enumerate(frames):
        n = _n_points(fr)
        if n > best_n:
            best_n, best_i = n, i
    return best_i


def run_pipeline(session_dir, part_type: str = "auto", segmenter=None,
                 max_frames: int = 60, lidar: bool = True):
    """Run the full chain on one capture session. Returns a result dict.

    lidar: True (default) means automatically use the depth primary path if the
           session contains LiDAR depth; set to False to force the model
           (no-depth) path, for ablation comparisons.
    """
    sys.path.insert(0, os.path.join(_PARENT, "tools"))
    import wg_ar_loader

    frames, meta = _load_frames(session_dir, wg_ar_loader, max_frames)
    if not frames:
        raise RuntimeError("No frames were loaded")

    # Step 2: segmentation
    seg = segmenter or SyntheticWoundSegmenter()
    ref_i = _best_reference_frame(frames)
    ref = frames[ref_i]
    rgb = ref.get("rgb")
    if rgb is None:
        # Without PIL, fall back to the declared resolution
        res = ref["metadata"]["imageResolution"]
        rgb = np.zeros((res["height"], res["width"], 3), np.uint8)
    mask, seg_info = seg.segment(rgb)

    # Step 3: scale recovery
    recon = MetricReconstructor().reconstruct(frames)

    # Step 4: curvature prior modelling
    if part_type == "auto":
        # Prefer ARKit plane classification to decide "flat", otherwise default
        # to limb
        cls = _dominant_plane_class(frames)
        part_type = "flat" if cls in ("table", "floor", "wall") else "limb_forearm"
    # Fit the surface using the reference frame's own feature points (ensures it
    # lies in front of the camera and matches the surface actually seen in this
    # frame); prefer local fitting with "wound-neighbourhood feature points"
    # (more stable), falling back to global if insufficient
    model, fit_mode = _fit_model_localized(ref, part_type, mask)

    # Step 5: 3D surface integration
    K = intrinsics_to_K(*ref["intrinsics"])
    M = camera_transform_to_cam2world(ref["camera_transform"])
    integrator = SurfaceAreaIntegrator()

    has_depth = (ref.get("depth") is not None) and lidar
    area_detail = {"primary_method": "model" if not has_depth else "lidar"}
    if has_depth:
        # LiDAR primary path: per-pixel depth -> back-projection -> surface
        # integration using true depth-derived normals
        # normal_smooth_sigma: smooth depth with a sigma converted from a
        # "physical scale" before computing normals. Calibration on a real human
        # point cloud (tools/validate_real_surface.py) shows this step reduces
        # the area error from 21.7% to 7.3% — without smoothing, micro-noise
        # tilts normals and systematically inflates area.
        conf = ref.get("confidence")
        sig = _auto_normal_sigma(K, ref["depth"], mask)
        area_lidar = integrator.integrate(
            rgb, mask, K, M, ("lidar", ref["depth"]), confidence=conf,
            normal_smooth_sigma=sig,
        )
        # Also run the model path (the no-depth fallback) as a control to
        # directly quantify the difference made by LiDAR (bottleneck #3)
        area_model = integrator.integrate(rgb, mask, K, M, ("model", model))
        area = area_lidar
        area_detail.update({
            "lidar": _area_summary(area_lidar),
            "model_baseline": _area_summary(area_model),
            "delta_cm2": area_lidar["area_cm2"] - area_model["area_cm2"],
            "delta_pct": (area_lidar["area_cm2"] - area_model["area_cm2"])
                        / max(area_model["area_cm2"], 1e-6) * 100.0,
        })
    else:
        area = integrator.integrate(rgb, mask, K, M, ("model", model))
        area_detail["model_baseline"] = _area_summary(area)

    # Step 6: multi-frame fusion (take the best-quality reference frames,
    # preferring those with depth)
    multi = _multiframe_area(frames, seg, model, K)

    return {
        "meta": meta,
        "reference_frame": ref_i,
        "part_type": part_type,
        "model_kind": model.surface_kind,
        "model_fit_mode": fit_mode,
        "model_params": _model_params(model),
        "segmentation": {"name": seg.name, **seg_info},
        "scale_recovery": recon["stats"],
        "has_depth": has_depth,
        "area": _area_summary(area),
        "_area_full": area,   # includes points3d (for visualisation only, not in the report)
        "area_detail": area_detail,
        "multiframe_median_cm2": multi["median_cm2"],
        "multiframe_std_cm2": multi["std_cm2"],
    }


def _area_summary(a):
    return {
        "method": a["method"],
        "area_cm2": a["area_cm2"],
        "flat_area_cm2": a["flat_area_cm2"],
        "compensation_ratio": a["compensation_ratio"],
        "n_valid": a["n_valid"],
    }


def _dominant_plane_class(frames):
    from collections import Counter
    c = Counter()
    for fr in frames:
        for p in fr.get("planes", []) or []:
            c[p.get("classification", "none")] += 1
    return c.most_common(1)[0][0] if c else "none"


def _fit_model_localized(ref, part_type, mask, pad_px: int = 60):
    """Use the added 2D pixel coordinates of feature points to restrict surface
    model fitting to the wound neighbourhood (bottleneck #4 fix).

    This is especially important for the "no-LiDAR (model) path": global PCA
    jitters on noisy point clouds, whereas feature points around the wound all
    lie on the same body surface, so fitting a local surface with them is more
    stable and better matches the surface actually seen in this frame. Falls
    back to global fitting if there are too few neighbourhood points.
    """
    pts = ref.get("feature_points")
    pix = ref.get("feature_point_pixels")
    if pts is None or len(pts) < 3 or pix is None or mask is None:
        if pts is not None and len(pts) >= 3:
            return fit_model_to_cloud(np.asarray(pts, dtype=float), part_type=part_type), "global"
        return fit_model_to_cloud(np.zeros((3, 3)), part_type=part_type), "empty"
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return fit_model_to_cloud(np.asarray(pts, dtype=float), part_type=part_type), "global"
    minx, maxx = xs.min() - pad_px, xs.max() + pad_px
    miny, maxy = ys.min() - pad_px, ys.max() + pad_px
    pix = np.asarray(pix, dtype=float)
    sel = (pix[:, 0] >= minx) & (pix[:, 0] <= maxx) & (pix[:, 1] >= miny) & (pix[:, 1] <= maxy)
    sel &= (pix[:, 0] >= 0) & (pix[:, 1] >= 0)   # exclude invalid [-1,-1] points
    local = np.asarray(pts, dtype=float)[sel]
    if len(local) >= 10:
        return fit_model_to_cloud(local, part_type=part_type), f"local(n={len(local)})"
    return fit_model_to_cloud(np.asarray(pts, dtype=float), part_type=part_type), "global(few_local)"


def _model_params(model):
    if model.surface_kind == "plane":
        return {"center": model.center.tolist(), "normal": model.normal.tolist()}
    if model.surface_kind == "cylinder":
        return {"axis_point": model.axis_point.tolist(),
                "axis_dir": model.axis_dir.tolist(), "radius_m": model.radius}
    if model.surface_kind == "sphere":
        return {"center": model.center.tolist(), "radius_m": model.radius}
    return {}


def _multiframe_area(frames, seg, model, K, n_refs: int = 5):
    """Auxiliary module: integrate separately over several tracking-stable frames
    and take the median (temporal multi-frame fusion).

    Frames with depth preferentially use the LiDAR path, while frames without
    depth fall back to the model path, making the fused result more stable on
    LiDAR sessions (bottleneck #4: the unstable fitting without LiDAR is directly
    avoided by using depth).
    """
    integrator = SurfaceAreaIntegrator()
    # Select the n_refs frames with "the most feature points" and "preference for
    # those with depth"
    def _score(i):
        fr = frames[i]
        return (_n_points(fr), 1 if fr.get("depth") is not None else 0)
    idx = sorted(range(len(frames)), key=_score, reverse=True)[:n_refs]
    areas = []
    methods = []
    for i in idx:
        fr = frames[i]
        rgb = fr.get("rgb")
        if rgb is None:
            res = fr["metadata"]["imageResolution"]
            rgb = np.zeros((res["height"], res["width"], 3), np.uint8)
        mask, _ = seg.segment(rgb)
        M = camera_transform_to_cam2world(fr["camera_transform"])
        if fr.get("depth") is not None:
            a = integrator.integrate(rgb, mask, K, M, ("lidar", fr["depth"]),
                                     confidence=fr.get("confidence"),
                                     normal_smooth_sigma=_auto_normal_sigma(
                                         K, fr["depth"], mask))
            methods.append("lidar")
        else:
            a = integrator.integrate(rgb, mask, K, M, ("model", model))
            methods.append("model")
        if a["n_valid"] > 10:
            areas.append(a["area_cm2"])
    if not areas:
        return {"median_cm2": 0.0, "std_cm2": 0.0, "n": 0, "methods": methods}
    arr = np.array(areas)
    return {"median_cm2": float(np.median(arr)), "std_cm2": float(arr.std()),
            "n": len(arr), "methods": methods}


def curvature_compensation_study(session_dir, part_type: str = "limb_forearm",
                                 radius_m: float = 0.04, max_frames: int = 30,
                                 wound_radius_px: float = 120, distance_m: float = 0.30):
    """Controlled experiment: compare the area of the same synthetic wound on a
    plane versus a cylindrical (limb) surface.

    Construct a reproducible scene (camera at the origin looking toward +Z) to
    directly quantify the effect of module 2's "curvature prior compensation" on
    the result, outputting real numbers (not estimates).
      - plane: at z=distance, normal +Z (no curvature, compensation ratio ≈1)
      - cylinder (limb): axis along X, centre (0,0,distance), radius radius_m
        (the camera sees the curved side)
    """
    sys.path.insert(0, os.path.join(_PARENT, "tools"))
    import wg_ar_loader
    frames, _ = _load_frames(session_dir, wg_ar_loader, max_frames)
    ref = frames[_best_reference_frame(frames)]
    K = intrinsics_to_K(*ref["intrinsics"])
    H, W = ref["metadata"]["imageResolution"]["height"], ref["metadata"]["imageResolution"]["width"]

    # Controlled scene: camera at the origin, cam2world = I
    M = np.eye(4)
    # Synthetic circular wound mask (image centre)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W // 2, H // 2
    mask = (((xx - cx) / wound_radius_px) ** 2 + ((yy - cy) / wound_radius_px) ** 2 <= 1).astype(np.uint8) * 255
    mask_info = {"synthetic": True, "area_px": int(mask.sum()) // 255,
                 "center": (cx, cy), "radius_px": wound_radius_px}

    from wg_algorithms.curvature_prior import PlaneModel, CylinderModel
    integrator = SurfaceAreaIntegrator()
    # Planar reference surface at z=distance (normal +Z, facing the camera, no
    # curvature)
    plane = PlaneModel([0, 0, distance_m], [0, 0, 1])
    flat = integrator.integrate(np.zeros((H, W, 3), np.uint8), mask, K, M, ("model", plane))

    if part_type == "flat":
        curved = flat
    else:
        # The nearest point of the cylinder has the same depth as the plane
        # (centre z = distance + R), so the comparison reflects only curvature
        cyl = CylinderModel([0, 0, distance_m + radius_m], [1, 0, 0], radius_m)
        curved = integrator.integrate(np.zeros((H, W, 3), np.uint8), mask, K, M, ("model", cyl))

    return {
        "synthetic_wound_px": mask_info["area_px"],
        "wound_radius_px": wound_radius_px,
        "distance_m": distance_m,
        "flat_surface_cm2": flat["area_cm2"],
        "curved_limb_cm2": curved["area_cm2"],
        "compensation_ratio": curved["compensation_ratio"],
        "radius_m": radius_m,
        "part_type": part_type,
    }
