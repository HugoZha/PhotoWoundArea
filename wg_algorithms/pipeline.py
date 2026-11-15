"""
pipeline.py — end-to-end measurement pipeline and controlled curvature experiment
=================================================

Step1 采集数据加载（wg_ar_loader）
Step2 伤口分割（模块三前半段，可插拔）
Step3 尺度恢复（模块一 -> 度量点云 + 置信度）
Step4 曲率先验建模（模块二 -> 部位参数曲面）
Step5 三维曲面积分（模块三后半段 -> 真实面积）
Step6 多帧融合（辅助 -> 多参考帧取中位/平均）
"""
from __future__ import annotations

import os
import sys
import numpy as np

# 允许直接以脚本方式运行
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
    """按当前帧的实际深度，自动换算法向平滑 sigma。

    写死像素值在不同设备上会失真：iPhone LiDAR 的点间距和 WoundsDB 点云
    差好几倍。这里统一换算成"约 target_mm 毫米的物理尺度"。
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
    """选特征点最多的一帧作为参考帧（跟踪质量最好）。"""
    best_i, best_n = 0, -1
    for i, fr in enumerate(frames):
        n = _n_points(fr)
        if n > best_n:
            best_n, best_i = n, i
    return best_i


def run_pipeline(session_dir, part_type: str = "auto", segmenter=None,
                 max_frames: int = 60, lidar: bool = True):
    """在一段采集会话上跑通完整链路。返回结果 dict。

    lidar: True（默认）表示若会话含 LiDAR 深度则自动走深度主路径；
           设为 False 可强制使用 model（无深度）路径，用于消融对比。
    """
    sys.path.insert(0, os.path.join(_PARENT, "tools"))
    import wg_ar_loader

    frames, meta = _load_frames(session_dir, wg_ar_loader, max_frames)
    if not frames:
        raise RuntimeError("未加载到任何帧")

    # Step2 分割
    seg = segmenter or SyntheticWoundSegmenter()
    ref_i = _best_reference_frame(frames)
    ref = frames[ref_i]
    rgb = ref.get("rgb")
    if rgb is None:
        # Without PIL, fall back to the declared resolution
        res = ref["metadata"]["imageResolution"]
        rgb = np.zeros((res["height"], res["width"], 3), np.uint8)
    mask, seg_info = seg.segment(rgb)

    # Step3 尺度恢复
    recon = MetricReconstructor().reconstruct(frames)

    # Step4 曲率先验建模
    if part_type == "auto":
        # 优先用 ARKit 平面分类决定 flat，否则默认 limb
        cls = _dominant_plane_class(frames)
        part_type = "flat" if cls in ("table", "floor", "wall") else "limb_forearm"
    # 用参考帧自身特征点拟合曲面（保证位于相机前方、贴合本帧实际看到的表面）
    # 优先用「伤口邻域特征点」局部拟合（更稳），不足则全局
    model, fit_mode = _fit_model_localized(ref, part_type, mask)

    # Step5 三维曲面积分
    K = intrinsics_to_K(*ref["intrinsics"])
    M = camera_transform_to_cam2world(ref["camera_transform"])
    integrator = SurfaceAreaIntegrator()

    has_depth = (ref.get("depth") is not None) and lidar
    area_detail = {"primary_method": "model" if not has_depth else "lidar"}
    if has_depth:
        # LiDAR 主路径：逐像素深度 -> 反投影 -> 基于深度图真实法向的曲面积分
        # normal_smooth_sigma：先用「物理尺度」换算出的 sigma 平滑深度再求法向。
        # 真实人体点云标定（tools/validate_real_surface.py）显示这一步把
        # 面积误差从 21.7% 降到 7.3% —— 不平滑时微观噪声会把法向打歪、面积系统性虚高。
        conf = ref.get("confidence")
        sig = _auto_normal_sigma(K, ref["depth"], mask)
        area_lidar = integrator.integrate(
            rgb, mask, K, M, ("lidar", ref["depth"]), confidence=conf,
            normal_smooth_sigma=sig,
        )
        # 同时跑 model 路径（无深度退化路径）做对照，直接量化 LiDAR 带来的差异（瓶颈#3）
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

    # Step6 多帧融合（取质量最好的若干参考帧，优先用含深度帧）
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
        "_area_full": area,   # 含 points3d（仅用于可视化，不入报告）
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
    """利用新增的特征点 2D 像素坐标，把曲面模型拟合限定到伤口邻域（瓶颈#4 修复）。

    对「无 LiDAR（model）路径」尤其重要：全局 PCA 在不干净点云上会抖，而伤口周围的
    特征点都位于同一身体曲面，用它们拟合局部曲面比全局更稳、更贴合本帧实际看到的表面。
    若邻域点不足则退化为全局拟合。
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
    sel &= (pix[:, 0] >= 0) & (pix[:, 1] >= 0)   # 排除 [-1,-1] 无效点
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
    """辅助模块：在跟踪稳定的若干帧上分别积分，取中位数（多帧时序融合）。

    含深度帧优先走 LiDAR 路径，无深度帧退化为 model 路径，使融合结果在
    LiDAR 会话上更稳定（瓶颈#4：无 LiDAR 时拟合不稳的问题被深度直接规避）。
    """
    integrator = SurfaceAreaIntegrator()
    # 选「特征点最多」且「优先含深度」的 n_refs 帧
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
    """受控实验：同一枚合成伤口，在平面 vs 圆柱（肢体）曲面上对比面积。

    构造可复现场景（相机在原点看向 +Z），直接量化模块二「曲率先验补偿」
    对结果的影响，输出真实数字（非估算）。
      - 平面：位于 z=distance，法向 +Z（无曲率，补偿比≈1）
      - 圆柱（肢体）：轴沿 X，中心 (0,0,distance)，半径 radius_m（相机看到曲面侧）
    """
    sys.path.insert(0, os.path.join(_PARENT, "tools"))
    import wg_ar_loader
    frames, _ = _load_frames(session_dir, wg_ar_loader, max_frames)
    ref = frames[_best_reference_frame(frames)]
    K = intrinsics_to_K(*ref["intrinsics"])
    H, W = ref["metadata"]["imageResolution"]["height"], ref["metadata"]["imageResolution"]["width"]

    # 受控场景：相机在原点，cam2world = I
    M = np.eye(4)
    # 合成圆形伤口掩膜（图像中心）
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W // 2, H // 2
    mask = (((xx - cx) / wound_radius_px) ** 2 + ((yy - cy) / wound_radius_px) ** 2 <= 1).astype(np.uint8) * 255
    mask_info = {"synthetic": True, "area_px": int(mask.sum()) // 255,
                 "center": (cx, cy), "radius_px": wound_radius_px}

    from wg_algorithms.curvature_prior import PlaneModel, CylinderModel
    integrator = SurfaceAreaIntegrator()
    # 平面参考面位于 z=distance（法向 +Z，正对相机，无曲率）
    plane = PlaneModel([0, 0, distance_m], [0, 0, 1])
    flat = integrator.integrate(np.zeros((H, W, 3), np.uint8), mask, K, M, ("model", plane))

    if part_type == "flat":
        curved = flat
    else:
        # 圆柱最近点与平面同深度（中心 z = distance + R），使对比仅反映曲率
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
