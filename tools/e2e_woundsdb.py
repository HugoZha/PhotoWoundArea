#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e2e_woundsdb.py — end-to-end area measurement on real photographs and registered point clouds.

链路（每个样本）：
    photo.jpg (4896x3264, 专家掩膜同尺寸)
      ├─(裁黑边)→ 分割模型 → 预测掩膜 ─┐
      └─ mask.png = 专家掩膜 (GT) ────┤
                                     │ SIFT+RANSAC 相似变换 (photo→stereo 左半 1024x768)
                                     ▼
              stereo-mesh.ply 结构化点云 (1024x768，mesh[i,j]=img[j,i])
                                     │ 三角化曲面积分（σ 按物理尺度平滑）
                                     ▼
              GT 曲面面积 vs 预测曲面面积 → 相对误差

真值口径：专家掩膜映射到点云后三角化的曲面面积（真实皮肤几何 + 专家边界），
          与「模型预测掩膜在同一块几何上」的面积之比，即为端到端误差。

用法：
    python tools/e2e_woundsdb.py --limit 6            # 冒烟
    python tools/e2e_woundsdb.py                      # 全量 188
    python tools/e2e_woundsdb.py --mode two_stage     # 两阶段 ROI 精化
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cv2  # noqa: E402

from validate_real_surface import (  # noqa: E402
    nominal_spacing, smooth_xyz, triangle_area_over_region,
)
from wg_algorithms.geometry import intrinsics_to_K  # noqa: E402
from wg_algorithms.surface_area import recommended_normal_sigma  # noqa: E402
from wg_algorithms.segmentation import _keep_largest_cc  # noqa: E402

PAT_ROOT = os.path.join(ROOT, "data/raw/woundsdb/patients_partial/patients")
RES_ROOT = os.path.join(ROOT, "data/raw/woundsdb/results")
PHOTO_SCALE = 0.25          # SIFT 用 1/4 分辨率的 photo（4896px 全跑太慢）
MESH_H, MESH_W = 1024, 768  # 点云网格（行,列）；stereo 图是其转置


# ------------------------------------------------------------------ 配准
def photo_to_stereo(photo_rgb, stereo_left, ratio=0.75, thr=3.0):
    """SIFT + RANSAC 估计 photo→stereo 的相似变换。

    固定 OpenCV 随机种子，使 RANSAC 配准确定性、可复现（消融表要求
    同一批样本逐层加组件，避免随机跳过导致 N 漂移）。

    返回 (M 2x3, inliers, n_good) 或 (None, 0, 0)。
    为什么用相似变换而非单应：两台相机看同一平面创面，视差有限，
    相似变换已足够把 ~0.3% 画面占比的掩膜搬过去，且约束更强、更抗漂移。
    """
    p_small = cv2.resize(photo_rgb, None, fx=PHOTO_SCALE, fy=PHOTO_SCALE,
                         interpolation=cv2.INTER_AREA)
    cv2.setRNGSeed(0)
    sift = cv2.SIFT_create(nfeatures=3000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(p_small, cv2.COLOR_RGB2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(stereo_left, cv2.COLOR_RGB2GRAY), None)
    if d1 is None or d2 is None:
        return None, 0, 0
    raw = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in raw if a.distance < ratio * b.distance]
    if len(good) < 10:
        return None, 0, len(good)
    src = np.float32([k1[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=thr)
    if M is None:
        return None, 0, len(good)
    return M, int(inl.sum()), len(good)


def mask_photo_to_mesh(mask_photo_bin, M):
    """photo 全分辨率二值掩膜 → 点云网格空间 (MESH_H, MESH_W)。

    路径：photo →(x0.25)→ photo_small →(M)→ stereo(768,1024) →(转置)→ mesh(1024,768)。
    转置依据 tools/probe_mesh_orient.py 的颜色相关性检验（0.76 vs 其余 <0.31）。
    """
    h, w = mask_photo_bin.shape
    m_small = cv2.resize(mask_photo_bin, (int(w * PHOTO_SCALE), int(h * PHOTO_SCALE)),
                         interpolation=cv2.INTER_NEAREST)
    # dsize=(宽,高)=(1024,768)：输出为 stereo 图像空间 (768 行,1024 列)
    st = cv2.warpAffine(m_small, M, (MESH_H, MESH_W),
                        flags=cv2.INTER_NEAREST, borderValue=0)
    # 必须二值化：warp 输出是 0/255，直接当 0/1 权重会把面积放大 255 倍
    return (st.T > 127).astype(np.uint8)  # (1024,768) 点云网格空间


def read_ply_grid_any(path: str, shape):
    """Robust binary/ASCII PLY reader with bounded header parsing.

    One WoundsDB stereo-mesh.ply (case_15) is ASCII while the rest are binary.
    A byte-at-a-time loop looking for the LF-only end_header marker never
    terminates on that CRLF file, so the header is located in 4 KB blocks
    with CRLF support. Binary bodies use the fast path; ASCII bodies are
    parsed with pandas (about 1 s for 786k rows).
    """
    with open(path, "rb") as f:
        head = f.read(4096)
        if b"format ascii" in head.split(b"end_header")[0]:
            # --- ascii 分支 ---
            import pandas as pd
            f.seek(0)
            raw = f.read()
            hdr_end = raw.index(b"end_header") + len(b"end_header")
            lines = raw[hdr_end:].split(b"\n")
            # 跳过 header 尾的空行
            rows = [l for l in lines if l.strip()]
            del lines, raw
            df = pd.DataFrame(
                [r.split() for r in rows],
                columns=["x", "y", "z", "r", "g", "b"], dtype=np.float32)
            h, w = shape
            nv = len(df)
            if h * w != nv:
                raise ValueError(f"{path}: {nv} 点与 {h}x{w} 不符")
            xyz = np.stack([df.x, df.y, df.z], 1).reshape(h, w, 3)
            rgb = np.stack([df.r, df.g, df.b], 1).reshape(h, w, 3).astype(np.uint8)
            return xyz, rgb
        # --- binary 分支（逻辑同 validate_real_surface.read_ply_grid，header 有界）---
        import struct as _s
        hdr_end = head.index(b"end_header") + len(b"end_header")
        # 吃掉紧跟的行尾
        while head[hdr_end:hdr_end + 1] in (b"\r", b"\n"):
            hdr_end += 1
        nv = int([l for l in head[:hdr_end].decode().split("\n")
                  if l.startswith("element vertex")][0].split()[-1])
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
        with open(path, "rb") as f:
            f.seek(hdr_end)
            d = np.frombuffer(f.read(nv * dt.itemsize), dtype=dt, count=nv)
    h, w = shape
    if h * w != nv:
        raise ValueError(f"{path}: {nv} 点与 {h}x{w} 不符")
    return (np.stack([d["x"], d["y"], d["z"]], 1).reshape(h, w, 3),
            np.stack([d["r"], d["g"], d["b"]], 1).reshape(h, w, 3))


def align_pinhole_fast(xyz_mm, n_sample: int = 60000):
    """align_pinhole 的抽样加速版（拟合 6 万点 vs 全量 70 万点，精度不变）。"""
    h, w, _ = xyz_mm.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    best = (np.inf, None, None)
    rng = np.random.default_rng(0)
    for perm in [(1, 0, 2), (0, 1, 2), (0, 2, 1), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        X = xyz_mm[:, :, list(perm)]
        z = X[:, :, 2]
        m = np.isfinite(z) & (z > 50)
        if m.sum() < 1000:
            continue
        idx = np.flatnonzero(m.ravel())
        if len(idx) > n_sample:
            idx = rng.choice(idx, n_sample, replace=False)
        us, vs = u.ravel()[idx], v.ravel()[idx]
        zu = X[:, :, 0].ravel()[idx] / z.ravel()[idx]
        zv = X[:, :, 1].ravel()[idx] / z.ravel()[idx]
        cu = np.polyfit(us, zu, 1)
        cv = np.polyfit(vs, zv, 1)
        fx, fy = 1.0 / cu[0], 1.0 / cv[0]
        cx, cy = -cu[1] * fx, -cv[1] * fy
        uhat = fx * X[:, :, 0] / z + cx
        vhat = fy * X[:, :, 1] / z + cy
        res = float(np.median(np.abs(uhat - u)[m]) + np.median(np.abs(vhat - v)[m]))
        if res < best[0]:
            best = (res, perm, (float(fx), float(fy), float(cx), float(cy)))
    return best


# ------------------------------------------------------------------ 分割
def crop_black_borders(img: np.ndarray, thr: int = 12):
    """裁掉连续纯黑边框，返回 (crop, (top, left))。同 tools/crop_black_border.py 口径。"""
    m = (img.max(axis=2) <= thr)
    if m.all():
        return img, (0, 0)

    def run(bits):
        n = 0
        for b in bits:
            if b:
                n += 1
            else:
                break
        return n

    top = run(m.all(axis=1))
    bot = run(m.all(axis=1)[::-1])
    left = run(m.all(axis=0))
    right = run(m.all(axis=0)[::-1])
    top = top if top >= 2 else 0
    bot = bot if bot >= 2 else 0
    left = left if left >= 2 else 0
    right = right if right >= 2 else 0
    h, w = img.shape[:2]
    crop = img[top:h - bot, left:w - right]
    return (crop if crop.size else img), (top, left)


def predict_mask(model, device, photo_rgb, mode="full", thr=0.5, img_size=256,
                 mask_roi_hint=None, tta=False):
    """返回 photo 坐标系下的二值预测掩膜。

    full:       整图 resize 到 256 直接预测（模拟远拍，最坏情况）；
    two_stage:  先 full 粗预测，再以粗质心为中心裁 1/6 幅面 ROI 放大精预测
                （模拟 App 近拍场景；粗预测为空则退化 full）。
    """
    from wg_algorithms.seg_model import predict_proba

    def _tta(img):
        p = predict_proba(model, img, device, img_size)
        if not tta:
            return p
        pf = predict_proba(model, img[:, ::-1].copy(), device, img_size)[:, ::-1]
        return (p + pf) / 2.0

    H, W = photo_rgb.shape[:2]

    def _run(roi_box=None):
        if roi_box is None:
            crop, (top, left) = crop_black_borders(photo_rgb)
            proba = predict_proba(model, crop, device, img_size)
            full = np.zeros((H, W), np.float32)
            ch, cw = proba.shape
            full[top:top + ch, left:left + cw] = proba
            return full
        top, left, bh, bw = roi_box
        proba = predict_proba(model, photo_rgb[top:top + bh, left:left + bw],
                              device, img_size)
        full = np.zeros((H, W), np.float32)
        full[top:top + bh, left:left + bw] = proba
        return full

    proba = _run()
    pred = (proba >= thr).astype(np.uint8) * 255

    if mode == "tile":
        # 滑窗推理：远拍全图里创面只占 ~0.3%，直接 resize 到 256 远低于训练分布
        # （中位 6.8%）。把原图切成 ~0.9MP 的重叠小块逐块预测再均值融合，
        # 每块里创面占比被放大约 30 倍，回到训练分布；无需任何 GT 信息。
        H, W = photo_rgb.shape[:2]
        tile, stride = 896, 448
        acc = np.zeros((H, W), np.float32)
        cnt = np.zeros((H, W), np.float32)
        tops = sorted({min(t, H - tile) for t in range(0, max(H - tile, 0) + 1, stride)})
        lefts = sorted({min(l, W - tile) for l in range(0, max(W - tile, 0) + 1, stride)})
        for top in tops:
            for left in lefts:
                th, tw = min(tile, H - top), min(tile, W - left)
                p = predict_proba(model, photo_rgb[top:top + th, left:left + tw],
                                  device, img_size)
                acc[top:top + th, left:left + tw] += p
                cnt[top:top + th, left:left + tw] += 1
        proba = acc / np.maximum(cnt, 1)
        pred = (proba >= thr).astype(np.uint8) * 255

    if mode == "roi":
        # 模拟 App 近拍取景：GT 掩膜**只用于决定「用户把镜头对准创面」的画幅**
        # （真实系统里取景由用户完成），不参与分割、阈值与面积计算。
        ys, xs = np.where(mask_roi_hint > 127)
        cy, cx = float(ys.mean()), float(xs.mean())
        size = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 2.5)
        size = int(np.clip(size, 800, min(H, W)))
        top = int(np.clip(cy - size / 2, 0, H - size))
        left = int(np.clip(cx - size / 2, 0, W - size))
        proba = _tta(photo_rgb[top:top + size, left:left + size])
        full = np.zeros((H, W), np.float32)
        full[top:top + size, left:left + size] = proba
        return _keep_largest_cc((full >= thr).astype(np.uint8) * 255)

    if mode == "two_stage" and (pred > 0).sum() > 30:
        ys, xs = np.where(pred > 0)
        cy, cx = int(ys.mean()), int(xs.mean())
        bh, bw = max(H // 6, 64), max(W // 6, 64)
        top = int(np.clip(cy - bh // 2, 0, H - bh))
        left = int(np.clip(cx - bw // 2, 0, W - bw))
        proba = _run((top, left, bh, bw))
        pred = (proba >= thr).astype(np.uint8) * 255

    return _keep_largest_cc(pred)


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/publicA_cropped_ft/best.pt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", default="full", choices=["full", "tile", "roi", "two_stage"])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--smooth_mm", type=float, default=6.0,
                    help="法向/GT 共用的物理平滑尺度")
    ap.add_argument("--min_cov", type=float, default=0.8)
    ap.add_argument("--tta", action="store_true", help="水平翻转 TTA 平均")
    ap.add_argument("--only_group", default=None, choices=["train", "val", "test"],
                    help="只评估指定分组（按 case 划分）。用于排除已参与训练的样本，"
                         "避免端到端指标被训练样本污染。")
    ap.add_argument("--split_file", default="data/woundsdb_seg_ms/split.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--vis", type=int, default=8, help="保存几张可视化")
    ap.add_argument("--out", default="runs/e2e_woundsdb")
    args = ap.parse_args()

    import torch
    from wg_algorithms.seg_model import load_wound_model

    device = args.device
    if device == "auto":
        device = ("mps" if getattr(torch.backends, "mps", None)
                  and torch.backends.mps.is_available() else "cpu")
    model = load_wound_model(os.path.join(ROOT, args.model), device=device)
    print(f"▶ 模型 {args.model} | 设备 {device} | 推理模式 {args.mode}")

    # ---- 配对清单
    cases = []
    for mp in sorted(glob.glob(os.path.join(PAT_ROOT, "case_*", "day_*",
                                            "data", "scene_*", "mask.png"))):
        d = os.path.dirname(mp)
        if not os.path.exists(os.path.join(d, "stereo_l.png")):
            continue
        rel = os.path.relpath(d, PAT_ROOT)              # case_X/day_Y/data/scene_N
        res = os.path.join(RES_ROOT, rel.replace("data", "results", 1),
                           "stereo-mesh.ply")
        if os.path.exists(res):
            cases.append((d, res, rel))

    # ---- 按 case 分组过滤（防止训练样本混入端到端评测）
    if args.only_group:
        sp = json.load(open(os.path.join(ROOT, args.split_file)))
        case2grp = {}
        for s in sp["samples"]:
            case2grp[s["case"]] = s["group"]
        keep = []
        for item in cases:
            c = item[2].split(os.sep)[0]
            if case2grp.get(c) == args.only_group:
                keep.append(item)
        print(f"▶ 只评估 {args.only_group} 组：{len(keep)}/{len(cases)} 个配对样本")
        cases = keep
    print(f"▶ 配对样本 {len(cases)} 个（mask+stereo_l+mesh 三全）")
    if args.limit:
        cases = cases[: args.limit]

    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    integ_rows, skips = [], []
    vis_saved = 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import time as _time
    for idx, (pdir, ply, rel) in enumerate(cases):
        t0 = _time.time()
        try:
            photo = np.asarray(Image.open(os.path.join(pdir, "photo.jpg")).convert("RGB"))
            mask_gt = (np.asarray(Image.open(os.path.join(pdir, "mask.png"))
                                  .convert("L")) > 127).astype(np.uint8) * 255
            stereo = np.asarray(Image.open(os.path.join(pdir, "stereo_l.png"))
                                .convert("RGB"))
            left = stereo[:, : stereo.shape[1] // 2]

            M, inl, ngood = photo_to_stereo(photo, left)
            if M is None or inl < 15:
                skips.append({"rel": rel, "why": f"sift inliers={inl}/{ngood}"})
                continue

            keep_gt = mask_photo_to_mesh(mask_gt, M)
            if keep_gt.sum() < 150:
                skips.append({"rel": rel, "why": f"gt 映射后仅 {int(keep_gt.sum())}px"})
                continue

            pred = predict_mask(model, device, photo, mode=args.mode, thr=args.thr,
                                mask_roi_hint=mask_gt, tta=args.tta)
            keep_pred = mask_photo_to_mesh(pred, M)

            xyz_mm, _rgbm = read_ply_grid_any(ply, (MESH_H, MESH_W))
            resid, perm, Kc = align_pinhole_fast(xyz_mm)
            xyz_mm = xyz_mm[:, :, list(perm)]
            fx, fy, cx, cy = Kc
            depth_m = xyz_mm[:, :, 2] / 1000.0
            sigma_px = recommended_normal_sigma(
                intrinsics_to_K(fx, fy, cx, cy), float(np.nanmedian(depth_m)),
                target_mm=args.smooth_mm)
            xyz_s = smooth_xyz(xyz_mm, sigma_px)
            gap = nominal_spacing(xyz_mm) * 4.0

            gt_mm2, cov_gt = triangle_area_over_region(xyz_s, keep_gt, max_gap_mm=gap)
            pred_mm2, cov_pr = triangle_area_over_region(xyz_s, keep_pred, max_gap_mm=gap)
            if gt_mm2 <= 0 or cov_gt < args.min_cov:
                skips.append({"rel": rel, "why": f"gt 点云质量 cov={cov_gt:.2f}"})
                continue
            if pred_mm2 <= 0:
                pred_mm2 = 0.0          # 模型漏检：面积 0 是合法结果

            gt_cm2, pr_cm2 = gt_mm2 / 100.0, pred_mm2 / 100.0
            err = abs(pr_cm2 - gt_cm2) / gt_cm2 * 100.0
            inter = int((keep_gt & keep_pred).sum())
            dice2d = 2 * inter / max(int(keep_gt.sum()) + int(keep_pred.sum()), 1)
            integ_rows.append({
                "rel": rel, "inliers": inl, "resid_px": round(resid, 3),
                "sigma_px": round(float(sigma_px), 2),
                "gt_cm2": round(gt_cm2, 3), "pred_cm2": round(pr_cm2, 3),
                "rel_err_pct": round(err, 2), "dice_2d_mesh": round(dice2d, 4),
                "cov_gt": round(cov_gt, 3), "cov_pred": round(cov_pr, 3),
                "n_gt_px": int(keep_gt.sum()), "n_pred_px": int(keep_pred.sum()),
            })
            print(f"  [{idx+1}/{len(cases)}] {rel}  GT {gt_cm2:.2f} cm² "
                  f"预测 {pr_cm2:.2f} cm²  误差 {err:.1f}%  dice {dice2d:.3f}"
                  f"  ({_time.time()-t0:.1f}s)", flush=True)

            if vis_saved < args.vis:
                vis_saved += 1
                m_small = cv2.resize(mask_gt, None, fx=PHOTO_SCALE, fy=PHOTO_SCALE,
                                     interpolation=cv2.INTER_NEAREST)
                p_small = cv2.resize(photo, None, fx=PHOTO_SCALE, fy=PHOTO_SCALE,
                                     interpolation=cv2.INTER_AREA)
                pred_s = cv2.resize(pred, (p_small.shape[1], p_small.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
                ax[0].imshow(p_small)
                ax[0].contour(m_small, levels=[0.5], colors="g", linewidths=1.2)
                ax[0].contour(pred_s, levels=[0.5], colors="r", linewidths=1.2)
                ax[0].set_title(f"photo  GT(绿)/Pred(红)  {rel.split('/')[-1]}")
                ax[0].axis("off")
                # 点云深度渲染 + 两掩膜轮廓（mesh 空间，转置回图像向）
                z = xyz_s[:, :, 2].copy()
                z[~np.isfinite(z)] = np.nanmedian(z)
                zn = (z - np.nanpercentile(z, 2)) / max(np.nanpercentile(z, 98)
                                                        - np.nanpercentile(z, 2), 1e-6)
                ax[1].imshow(zn.T, cmap="gray")
                ax[1].contour(keep_gt.T.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
                ax[1].contour(keep_pred.T.astype(float), levels=[0.5], colors="r", linewidths=1.2)
                ax[1].set_title(f"点云深度  GT {gt_cm2:.2f} vs Pred {pr_cm2:.2f} cm²")
                ax[1].axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(ROOT, args.out, f"vis_{vis_saved:02d}.png"), dpi=110)
                plt.close(fig)
        except Exception as ex:
            skips.append({"rel": rel, "why": f"{type(ex).__name__}: {str(ex)[:90]}"})
            print(f"  [{idx+1}/{len(cases)}] SKIP {rel} ({_time.time()-t0:.1f}s)", flush=True)

    # ---- 汇总
    if not integ_rows:
        print(f"❌ 无有效样本。跳过 {len(skips)} 条：")
        for s in skips[:10]:
            print("   ", s)
        return 1

    errs = np.array([r["rel_err_pct"] for r in integ_rows])
    gts = np.array([r["gt_cm2"] for r in integ_rows])
    prs = np.array([r["pred_cm2"] for r in integ_rows])
    dcs = np.array([r["dice_2d_mesh"] for r in integ_rows])
    # 像素比（模型/专家）与面积比（模型/专家）——用于诊断偏置来源
    pxr = np.array([r["n_pred_px"] / max(r["n_gt_px"], 1) for r in integ_rows])
    ar = prs / gts

    # ---- 按创面大小分层（阈值 18.7cm²，与既往分析一致）
    THR = 18.7
    small = gts < THR
    large = ~small

    # ---- 系统偏置与几何保真度（MedIA 归因：偏置来自掩膜分歧还是几何口径？）
    offsets = (prs - gts) / gts * 100.0          # 带符号偏置
    faith_r = float(np.corrcoef(pxr, ar)[0, 1])  # ≈1 → 几何积分忠实于掩膜
    from collections import defaultdict
    bycase = defaultdict(list)
    for r, o in zip(integ_rows, offsets):
        bycase[r["rel"].split(os.sep)[0]].append(o)

    def _stratum(mask, name):
        if int(mask.sum()) == 0:
            return
        e = errs[mask]
        print(f"  {name} (N={int(mask.sum())}): 中位误差 {np.median(e):.1f}%  "
              f"<10%占{(e < 10).mean()*100:.0f}%  dice中位 {np.median(dcs[mask]):.3f}  "
              f"面积中位 {np.median(gts[mask]):.1f}cm²")

    print("\n" + "=" * 74)
    print(f"端到端面积验证（真实分割 + 真实点云）  N={len(integ_rows)}"
          f"  跳过 {len(skips)}")
    print("=" * 74)
    print(f"  面积相对误差: 均值 {errs.mean():.1f}%  中位 {np.median(errs):.1f}%  "
          f"P90 {np.percentile(errs, 90):.1f}%  最大 {errs.max():.1f}%")
    print(f"  误差<10% 占比 {(errs<10).mean()*100:.0f}%   <20% 占比 {(errs<20).mean()*100:.0f}%")
    print(f"  真值面积: 中位 {np.median(gts):.2f} cm² ({gts.min():.2f}~{gts.max():.2f})")
    print(f"  与真值相关 R = {np.corrcoef(gts, prs)[0,1]:.3f}")
    print(f"  mesh 上 2D dice: 中位 {np.median(dcs):.3f}")
    print("\n--- 按创面大小分层（阈值 %.1f cm²）---" % THR)
    _stratum(small, "  小创面 (<%.1f cm²)" % THR)
    _stratum(large, "  大创面 (>=%.1f cm²)" % THR)
    print("\n--- 系统偏置与几何保真度（真值口径校准诊断）---")
    print(f"  带符号面积偏置中位 {np.median(offsets):+.1f}%  "
          f"高估占比 {(offsets > 0).mean()*100:.0f}%")
    print(f"  面积比 vs 像素比 相关 R={faith_r:.3f}  "
          f"（≈1 说明几何积分忠实于分割掩膜，偏置源自掩膜分歧而非几何口径）")
    print("  逐 case 带符号偏置中位:")
    for c in sorted(bycase):
        ol = bycase[c]
        print(f"    {c}: N={len(ol)} 中位 {np.median(ol):+.1f}%  "
              f"{'⚠' if abs(np.median(ol)) > 15 else ''}")

    summary = {
        "overall": {
            "n": len(integ_rows), "err_mean": float(errs.mean()),
            "err_median": float(np.median(errs)),
            "err_p90": float(np.percentile(errs, 90)),
            "err_lt10": float((errs < 10).mean() * 100),
            "err_lt20": float((errs < 20).mean() * 100),
            "gt_median": float(np.median(gts)),
            "r_gt_pred": float(np.corrcoef(gts, prs)[0, 1]),
            "dice_median": float(np.median(dcs)),
        },
        "stratify_threshold_cm2": THR,
        "small": (None if int(small.sum()) == 0 else {
            "n": int(small.sum()), "err_median": float(np.median(errs[small])),
            "err_lt10": float((errs[small] < 10).mean() * 100),
            "dice_median": float(np.median(dcs[small])),
            "gt_median": float(np.median(gts[small]))}),
        "large": (None if int(large.sum()) == 0 else {
            "n": int(large.sum()), "err_median": float(np.median(errs[large])),
            "err_lt10": float((errs[large] < 10).mean() * 100),
            "dice_median": float(np.median(dcs[large])),
            "gt_median": float(np.median(gts[large]))}),
        "bias": {
            "offset_median": float(np.median(offsets)),
            "overratio": float((offsets > 0).mean() * 100),
            "area_vs_pixel_ratio_R": faith_r,
            "by_case": {c: {"n": len(ol), "offset_median": float(np.median(ol))}
                        for c, ol in bycase.items()},
        },
    }

    with open(os.path.join(ROOT, args.out, "results.json"), "w") as f:
        json.dump({"config": vars(args), "rows": integ_rows, "skips": skips,
                   "summary": summary}, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 明细 -> {args.out}/results.json   可视化 -> {args.out}/vis_*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
