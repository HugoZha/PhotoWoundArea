#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e2e_woundsdb.py — end-to-end area measurement on real photographs and registered point clouds.

Pipeline (per sample):
    photo.jpg (4896x3264, expert mask at the same size)
      ├─(crop black borders)→ segmentation model → predicted mask ─┐
      └─ mask.png = expert mask (GT) ──────────────────────────────┤
                                     │ SIFT+RANSAC similarity transform (photo→stereo left half 1024x768)
                                     ▼
              stereo-mesh.ply structured point cloud (1024x768, mesh[i,j]=img[j,i])
                                     │ triangulated surface integration (σ smoothed at the physical scale)
                                     ▼
              GT surface area vs predicted surface area → relative error

Ground-truth definition: the triangulated surface area after mapping the expert mask onto the point cloud
          (real skin geometry + expert boundary), divided by the area of "the model-predicted mask on the
          same piece of geometry" — this ratio is the end-to-end error.

Usage:
    python tools/e2e_woundsdb.py --limit 6            # smoke test
    python tools/e2e_woundsdb.py                      # full run, all 188
    python tools/e2e_woundsdb.py --mode two_stage     # two-stage ROI refinement
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
PHOTO_SCALE = 0.25          # SIFT uses 1/4-resolution photo (running the full 4896px is too slow)
MESH_H, MESH_W = 1024, 768  # point cloud grid (rows, cols); the stereo image is its transpose


# ------------------------------------------------------------------ registration
def photo_to_stereo(photo_rgb, stereo_left, ratio=0.75, thr=3.0):
    """SIFT + RANSAC to estimate the photo→stereo similarity transform.

    Fix the OpenCV random seed so RANSAC registration is deterministic and
    reproducible (the ablation table requires adding components layer by layer
    to the same batch of samples, avoiding random skips that make N drift).

    Returns (M 2x3, inliers, n_good), or (None, 0, 0).
    Why a similarity transform rather than a homography: both cameras view the
    same planar wound with limited parallax; a similarity transform is already
    enough to transfer the mask (which covers ~0.3% of the frame), and is more
    constrained and more robust against drift.
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
    """photo full-resolution binary mask → point cloud grid space (MESH_H, MESH_W).

    Path: photo →(x0.25)→ photo_small →(M)→ stereo(768,1024) →(transpose)→ mesh(1024,768).
    The transpose is based on the color-correlation check in tools/probe_mesh_orient.py
    (0.76 versus <0.31 for the rest).
    """
    h, w = mask_photo_bin.shape
    m_small = cv2.resize(mask_photo_bin, (int(w * PHOTO_SCALE), int(h * PHOTO_SCALE)),
                         interpolation=cv2.INTER_NEAREST)
    # dsize=(width,height)=(1024,768): the output is in stereo image space (768 rows, 1024 cols)
    st = cv2.warpAffine(m_small, M, (MESH_H, MESH_W),
                        flags=cv2.INTER_NEAREST, borderValue=0)
    # Must binarize: warp outputs 0/255; using them directly as 0/1 weights would inflate the area 255x
    return (st.T > 127).astype(np.uint8)  # (1024,768) point cloud grid space


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
            # --- ascii branch ---
            import pandas as pd
            f.seek(0)
            raw = f.read()
            hdr_end = raw.index(b"end_header") + len(b"end_header")
            lines = raw[hdr_end:].split(b"\n")
            # skip blank lines at the end of the header
            rows = [l for l in lines if l.strip()]
            del lines, raw
            df = pd.DataFrame(
                [r.split() for r in rows],
                columns=["x", "y", "z", "r", "g", "b"], dtype=np.float32)
            h, w = shape
            nv = len(df)
            if h * w != nv:
                raise ValueError(f"{path}: {nv} points do not match {h}x{w}")
            xyz = np.stack([df.x, df.y, df.z], 1).reshape(h, w, 3)
            rgb = np.stack([df.r, df.g, df.b], 1).reshape(h, w, 3).astype(np.uint8)
            return xyz, rgb
        # --- binary branch (same logic as validate_real_surface.read_ply_grid, bounded header) ---
        import struct as _s
        hdr_end = head.index(b"end_header") + len(b"end_header")
        # consume the immediately following line ending
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
        raise ValueError(f"{path}: {nv} points do not match {h}x{w}")
    return (np.stack([d["x"], d["y"], d["z"]], 1).reshape(h, w, 3),
            np.stack([d["r"], d["g"], d["b"]], 1).reshape(h, w, 3))


def align_pinhole_fast(xyz_mm, n_sample: int = 60000):
    """Sampled, accelerated version of align_pinhole (fits 60k points vs 700k in full, same accuracy)."""
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


# ------------------------------------------------------------------ segmentation
def crop_black_borders(img: np.ndarray, thr: int = 12):
    """Crop contiguous pure-black borders, returns (crop, (top, left)). Same convention as tools/crop_black_border.py."""
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
    """Return the binary predicted mask in photo coordinates.

    full:       resize the whole image to 256 and predict directly (simulates distant shooting, worst case);
    two_stage:  first a coarse full-frame prediction, then crop a 1/6-frame ROI centered on the coarse
                centroid and enlarge it for a fine prediction (simulates the app's close-up scenario;
                if the coarse prediction is empty, fall back to full).
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
        # Sliding-window inference: in a distant full-frame shot the wound covers only ~0.3%, so resizing
        # directly to 256 is far below the training distribution (median 6.8%). Cut the original into
        # overlapping ~0.9MP tiles, predict each, then average-fuse; the wound fraction in each tile is
        # enlarged about 30x, back into the training distribution; no GT information is needed.
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
        # Simulate the app's close-up framing: the GT mask is used **only** to decide the frame in which
        # "the user points the camera at the wound" (in a real system framing is done by the user); it
        # takes no part in segmentation, thresholding, or area computation.
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


# ------------------------------------------------------------------ main pipeline
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/publicA_cropped_ft/best.pt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", default="full", choices=["full", "tile", "roi", "two_stage"])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--smooth_mm", type=float, default=6.0,
                    help="physical smoothing scale shared by the normal and GT")
    ap.add_argument("--min_cov", type=float, default=0.8)
    ap.add_argument("--tta", action="store_true", help="horizontal-flip TTA averaging")
    ap.add_argument("--only_group", default=None, choices=["train", "val", "test"],
                    help="Evaluate only the specified group (split by case). Used to exclude samples "
                         "already used in training, so the end-to-end metrics are not polluted by "
                         "training samples.")
    ap.add_argument("--split_file", default="data/woundsdb_seg_ms/split.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--vis", type=int, default=8, help="how many visualizations to save")
    ap.add_argument("--out", default="runs/e2e_woundsdb")
    args = ap.parse_args()

    import torch
    from wg_algorithms.seg_model import load_wound_model

    device = args.device
    if device == "auto":
        device = ("mps" if getattr(torch.backends, "mps", None)
                  and torch.backends.mps.is_available() else "cpu")
    model = load_wound_model(os.path.join(ROOT, args.model), device=device)
    print(f"▶ model {args.model} | device {device} | inference mode {args.mode}")

    # ---- paired-case list
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

    # ---- filter by case group (prevent training samples from leaking into the end-to-end evaluation)
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
        print(f"▶ evaluating only the {args.only_group} group: {len(keep)}/{len(cases)} paired samples")
        cases = keep
    print(f"▶ {len(cases)} paired samples (all three of mask+stereo_l+mesh present)")
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
                skips.append({"rel": rel, "why": f"gt has only {int(keep_gt.sum())}px after mapping"})
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
                skips.append({"rel": rel, "why": f"gt point-cloud quality cov={cov_gt:.2f}"})
                continue
            if pred_mm2 <= 0:
                pred_mm2 = 0.0          # model missed detection: area 0 is a valid result

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
                  f"pred {pr_cm2:.2f} cm²  error {err:.1f}%  dice {dice2d:.3f}"
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
                ax[0].set_title(f"photo  GT(green)/Pred(red)  {rel.split('/')[-1]}")
                ax[0].axis("off")
                # point cloud depth rendering + both mask contours (mesh space, transposed back to image orientation)
                z = xyz_s[:, :, 2].copy()
                z[~np.isfinite(z)] = np.nanmedian(z)
                zn = (z - np.nanpercentile(z, 2)) / max(np.nanpercentile(z, 98)
                                                        - np.nanpercentile(z, 2), 1e-6)
                ax[1].imshow(zn.T, cmap="gray")
                ax[1].contour(keep_gt.T.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
                ax[1].contour(keep_pred.T.astype(float), levels=[0.5], colors="r", linewidths=1.2)
                ax[1].set_title(f"point-cloud depth  GT {gt_cm2:.2f} vs Pred {pr_cm2:.2f} cm²")
                ax[1].axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(ROOT, args.out, f"vis_{vis_saved:02d}.png"), dpi=110)
                plt.close(fig)
        except Exception as ex:
            skips.append({"rel": rel, "why": f"{type(ex).__name__}: {str(ex)[:90]}"})
            print(f"  [{idx+1}/{len(cases)}] SKIP {rel} ({_time.time()-t0:.1f}s)", flush=True)

    # ---- summary
    if not integ_rows:
        print(f"❌ No valid samples. Skipped {len(skips)}:")
        for s in skips[:10]:
            print("   ", s)
        return 1

    errs = np.array([r["rel_err_pct"] for r in integ_rows])
    gts = np.array([r["gt_cm2"] for r in integ_rows])
    prs = np.array([r["pred_cm2"] for r in integ_rows])
    dcs = np.array([r["dice_2d_mesh"] for r in integ_rows])
    # pixel ratio (model/expert) and area ratio (model/expert) — for diagnosing the source of bias
    pxr = np.array([r["n_pred_px"] / max(r["n_gt_px"], 1) for r in integ_rows])
    ar = prs / gts

    # ---- stratify by wound size (threshold 18.7cm², consistent with previous analysis)
    THR = 18.7
    small = gts < THR
    large = ~small

    # ---- systematic bias and geometric fidelity (MedIA attribution: does bias come from mask
    #      disagreement or from the geometry definition?)
    offsets = (prs - gts) / gts * 100.0          # signed bias
    faith_r = float(np.corrcoef(pxr, ar)[0, 1])  # ≈1 → geometric integration is faithful to the mask
    from collections import defaultdict
    bycase = defaultdict(list)
    for r, o in zip(integ_rows, offsets):
        bycase[r["rel"].split(os.sep)[0]].append(o)

    def _stratum(mask, name):
        if int(mask.sum()) == 0:
            return
        e = errs[mask]
        print(f"  {name} (N={int(mask.sum())}): median error {np.median(e):.1f}%  "
              f"<10% share {(e < 10).mean()*100:.0f}%  median dice {np.median(dcs[mask]):.3f}  "
              f"median area {np.median(gts[mask]):.1f}cm²")

    print("\n" + "=" * 74)
    print(f"End-to-end area validation (real segmentation + real point cloud)  N={len(integ_rows)}"
          f"  skipped {len(skips)}")
    print("=" * 74)
    print(f"  area relative error: mean {errs.mean():.1f}%  median {np.median(errs):.1f}%  "
          f"P90 {np.percentile(errs, 90):.1f}%  max {errs.max():.1f}%")
    print(f"  error<10% share {(errs<10).mean()*100:.0f}%   <20% share {(errs<20).mean()*100:.0f}%")
    print(f"  ground-truth area: median {np.median(gts):.2f} cm² ({gts.min():.2f}~{gts.max():.2f})")
    print(f"  correlation with ground truth R = {np.corrcoef(gts, prs)[0,1]:.3f}")
    print(f"  2D dice on mesh: median {np.median(dcs):.3f}")
    print("\n--- stratification by wound size (threshold %.1f cm²) ---" % THR)
    _stratum(small, "  small wounds (<%.1f cm²)" % THR)
    _stratum(large, "  large wounds (>=%.1f cm²)" % THR)
    print("\n--- systematic bias and geometric fidelity (ground-truth definition calibration diagnostics) ---")
    print(f"  median signed area bias {np.median(offsets):+.1f}%  "
          f"overestimation share {(offsets > 0).mean()*100:.0f}%")
    print(f"  area ratio vs pixel ratio correlation R={faith_r:.3f}  "
          f"(≈1 means geometric integration is faithful to the segmentation mask, so the bias "
          f"stems from mask disagreement rather than from the geometry definition)")
    print("  median signed bias per case:")
    for c in sorted(bycase):
        ol = bycase[c]
        print(f"    {c}: N={len(ol)} median {np.median(ol):+.1f}%  "
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
    print(f"\n✅ details -> {args.out}/results.json   visualizations -> {args.out}/vis_*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
