#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""validate_real_surface.py — validate the surface-area integrator on real skin point clouds.

Background (why this step matters):
    All our previous 3D area validations used **synthetic geometry** (plane/cylinder/sphere).
    Those experiments show the formula is not mathematically wrong, but cannot show that it
    works on **real skin** — real skin has creases, irregular curvature, and steep steps at
    ulcer edges. Public resources that provide both a "3D point cloud + expert mask" are
    extremely rare; WoundsDB is one of them: each of its scenes comes with stereo-mesh.ply /
    depth-mesh.ply, which is a **structured point cloud arranged on a camera grid**, with xyz
    in millimetres.

Approach (validation without needing a mask):
    1. The point cloud itself can be treated as a depth map seen by a pinhole camera: the grid
       coordinates (row, col) are the pixels, and each pixel has a real 3D point. We solve for
       the virtual camera's intrinsics fx, fy, cx, cy by least squares from x=(u-cx)·z/fx —
       this way we build a self-consistent imaging model **without relying on any external
       calibration**.
    2. Take a circular region on the point cloud (radius R millimetres, measured by 3D
       distance rather than pixel distance, so on the real surface it is ≈ a disk),
    3. True area = triangulate the 3D points of this region and sum them (Ground Truth),
    4. Our algorithm = SurfaceAreaIntegrator taking the 'lidar' branch, fed the same virtual
       camera's K and the same depth map, producing an area,
    5. Compare the relative error between the two.

What this validates is "given the same real geometry, can our formula recover its area",
**excluding segmentation error** — segmentation error is a separate matter (see
analyze_area_vs_dice.py).

Usage:
    python tools/validate_real_surface.py --root data/raw/woundsdb/results --scenes 6 \
        --patches 8 --radius_mm 15
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wg_algorithms.geometry import intrinsics_to_K
from wg_algorithms.surface_area import SurfaceAreaIntegrator


# ------------------------------------------------------------------ PLY reading
def read_ply_grid(path: str, shape):
    """Read a WoundsDB binary_little_endian PLY into xyz (mm) and rgb grids.

    The files carry vertices only: each "mesh" is a structured point cloud
    stored in row-major order as 1024 rows x 768 columns. Swapping the axes
    produces implausible point spacing.
    """
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            hdr += f.read(1)
    lines = hdr.decode().split("\n")
    nv = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
    h, w = shape
    if h * w != nv:
        raise ValueError(f"{path}: {nv} points do not match {h}x{w}")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    with open(path, "rb") as f:
        f.read(len(hdr))
        d = np.frombuffer(f.read(nv * dt.itemsize), dtype=dt, count=nv)
    return (np.stack([d["x"], d["y"], d["z"]], 1).reshape(h, w, 3),
            np.stack([d["r"], d["g"], d["b"]], 1).reshape(h, w, 3))


_PERMS = list(itertools.permutations([0, 1, 2]))


def align_pinhole(xyz_mm):
    """Align the point cloud to a pinhole camera frame and recover intrinsics.

    WoundsDB stores coordinates in (y, x, z) order, not (x, y, z). Treating x
    as the horizontal axis gives a reprojection residual of about 4000 px.
    All six axis permutations are fitted and the one with the smallest
    residual is used; the correct order gives 0.00 px, so no external
    calibration file is required.
    """
    h, w, _ = xyz_mm.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    best = (np.inf, None, None)
    for perm in _PERMS:
        X = xyz_mm[:, :, list(perm)]
        z = X[:, :, 2]
        m = np.isfinite(z) & (z > 50)
        if m.sum() < 1000:
            continue
        Au = np.stack([u[m], np.ones(int(m.sum()))], 1)
        Av = np.stack([v[m], np.ones(int(m.sum()))], 1)
        cu, *_ = np.linalg.lstsq(Au, (X[:, :, 0] / z)[m], rcond=None)
        cv, *_ = np.linalg.lstsq(Av, (X[:, :, 1] / z)[m], rcond=None)
        fx, cx = 1.0 / cu[0], None
        fx = 1.0 / cu[0]
        cx = -cu[1] * fx
        fy = 1.0 / cv[0]
        cy = -cv[1] * fy
        uhat = fx * X[:, :, 0] / z + cx
        vhat = fy * X[:, :, 1] / z + cy
        res = float(np.median(np.abs(uhat - u)[m]) + np.median(np.abs(vhat - v)[m]))
        if res < best[0]:
            best = (res, perm, (float(fx), float(fy), float(cx), float(cy)))
    return best  # (residual_px, perm, (fx,fy,cx,cy))


def triangle_area_over_region(xyz, keep, max_gap_mm: float = 0.0):
    """Triangulate the structured points inside `keep` and return the surface area.

    Each 2x2 cell is split into two triangles, weighted by the fraction of
    its four corners that lie inside the region.

    Two conventions matter:
    1. Keeping only cells whose four corners are all inside drops about 30%
       of the boundary area, so corner-coverage weighting is used instead.
    2. Isolated stereo outliers (depth jumping tens of centimetres) create
       huge triangles; cells whose corner span exceeds a multiple of the
       normal point spacing are discarded via max_gap_mm.
    """
    k = keep.astype(np.float64)[:, :, None]
    p00 = xyz[:-1, :-1]
    p10 = xyz[1:, :-1]
    p01 = xyz[:-1, 1:]
    p11 = xyz[1:, 1:]

    wgt = 0.25 * (k[:-1, :-1] + k[1:, :-1] + k[:-1, 1:] + k[1:, 1:])  # (H-1,W-1,1)
    if max_gap_mm and max_gap_mm > 0:
        corners = np.stack([p00, p10, p01, p11], axis=0)              # (4,H-1,W-1,3)
        spread = (corners.max(axis=0) - corners.min(axis=0)).max(axis=2)  # (H-1,W-1)
        wgt = wgt * (spread < max_gap_mm)[:, :, None]

    a1 = np.linalg.norm(np.cross(p10 - p00, p01 - p00), axis=2)
    a2 = np.linalg.norm(np.cross(p11 - p10, p01 - p10), axis=2)
    cell = 0.5 * (a1 + a2)                            # (H-1,W-1)
    area = float((cell * wgt[:, :, 0]).sum())
    # Valid cells / cells that should be covered — low coverage means the point cloud itself
    # is of poor quality; in that case "partial counting" makes the ground truth severely too
    # low (we have observed the geometrically impossible "ground truth < projected area"),
    # and the correct action is to discard the whole patch rather than force a value.
    expect = max(float((0.25 * (k[:-1, :-1] + k[1:, :-1] + k[:-1, 1:]
                                + k[1:, 1:])[:, :, 0]).sum()), 1e-9)
    return area, float(wgt[:, :, 0].sum() / expect)


def nominal_spacing(xyz) -> float:
    """Median of the normal point spacing of the point cloud (mm) — used to set the "outlier" threshold."""
    d = np.linalg.norm(np.diff(xyz, axis=1), axis=2)
    return float(np.median(d))


def smooth_xyz(xyz, sigma):
    """Smooth the point cloud itself at the same scale, to obtain a matching Ground Truth.

    Why the GT must also be smoothed: at different scales "surface area" is not a single
    number — the finer you look, the larger the area (microscopic skin creases contribute
    area). What clinics want to measure is the **macroscopic** area, so "computing normals
    with a smoothing sigma" must be paired with "using a surface smoothed with the same
    sigma as the ground truth" for comparison.
    """
    if not sigma:
        return xyz
    from scipy.ndimage import gaussian_filter
    out = np.zeros_like(xyz)
    for k in range(3):
        out[:, :, k] = gaussian_filter(xyz[:, :, k], sigma=sigma, mode="nearest")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/raw/woundsdb/results")
    ap.add_argument("--scenes", type=int, default=6, help="how many scenes to use in the experiment")
    ap.add_argument("--patches", type=int, default=8, help="how many patches to sample per scene")
    ap.add_argument("--radius_mm", type=float, default=15.0, help="patch radius (millimetres)")
    ap.add_argument("--cloud", default="stereo-mesh.ply",
                    choices=["stereo-mesh.ply", "depth-mesh.ply"])
    ap.add_argument("--sigmas", default="0,1,2,3,5,8",
                    help="normal smoothing strengths to compare (pixels)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="result JSON")
    args = ap.parse_args()

    shape = {"stereo-mesh.ply": (1024, 768), "depth-mesh.ply": (176, 144)}[args.cloud]
    scenes = sorted(glob.glob(os.path.join(args.root, "case_*", "day_*", "results",
                                           "scene_*")))
    scenes = [s for s in scenes if os.path.exists(os.path.join(s, args.cloud))]
    if not scenes:
        print(f"❌ no scene containing {args.cloud} found under {args.root}")
        return 1
    scenes = scenes[: args.scenes]
    sigmas = [float(x) for x in args.sigmas.split(",") if x != ""]
    print(f"▶ Real-surface validation: {len(scenes)} scenes / {args.cloud} / "
          f"{args.patches} patches per scene, R={args.radius_mm}mm\n")

    rng = np.random.default_rng(args.seed)
    integ = SurfaceAreaIntegrator()
    rows = []
    for si, sdir in enumerate(scenes):
        try:
            xyz_mm, rgb = read_ply_grid(os.path.join(sdir, args.cloud), shape)
        except Exception as e:
            print(f"  [!] {sdir}: {e}")
            continue
        resid, perm, Kc = align_pinhole(xyz_mm)
        xyz_mm = xyz_mm[:, :, list(perm)]          # reorient to the standard pinhole frame
        fx, fy, cx, cy = Kc
        K = intrinsics_to_K(fx, fy, cx, cy)
        depth_m = xyz_mm[:, :, 2] / 1000.0          # mm -> m (algorithm convention)
        z = xyz_mm[:, :, 2]
        M = np.eye(4)                               # the point cloud is already in the camera frame; the camera is at the origin
        gap_max = nominal_spacing(xyz_mm) * 4.0     # cells with spacing above 4x the normal point spacing are treated as outliers

        case = re.search(r"case_(\d+)", sdir)
        day = re.search(r"day_(\d+)", sdir)
        scene = re.search(r"scene_(\d+)", sdir)
        tag = (f"case_{case.group(1) if case else '?'}__"
               f"day_{day.group(1) if day else '?'}__"
               f"scene_{scene.group(1) if scene else '?'}")

        # ---- determine patches first (independent of sigma, so different sigmas compare the same skin)
        vn = z[np.isfinite(z)]
        valid = (np.isfinite(z) & (z > max(50.0, np.percentile(vn, 5)))
                 & (z < np.percentile(vn, 95)))
        ys_all, xs_all = np.where(valid)
        patches = []
        for _ in range(args.patches * 60):
            if len(patches) >= args.patches:
                break
            k = rng.integers(len(xs_all))
            cy0, cx0 = ys_all[k], xs_all[k]
            d3 = np.linalg.norm(xyz_mm - xyz_mm[cy0, cx0], axis=2)
            keep = valid & (d3 <= args.radius_mm)
            if keep.sum() >= 60:
                patches.append((cy0, cx0, keep))
        print(f"  [{si+1}/{len(scenes)}] {os.path.relpath(sdir, args.root)}  "
              f"fx={fx:.0f} cx={cx:.0f} cy={cy:.0f} axis order {perm} "
              f"residual {resid:.3f}px median depth {np.median(depth_m)*1000:.0f}mm "
              f"({len(patches)} patches)")

        for (cy0, cx0, keep) in patches:
            # First use the **unsmoothed** point cloud to judge whether this patch of skin is
            # good enough: coverage below 90% means holes/outliers here, so skip the whole
            # patch; this also ensures all sigmas use the same patches for a fair comparison.
            _, cov_raw = triangle_area_over_region(xyz_mm, keep, max_gap_mm=gap_max)
            if cov_raw < 0.9:
                continue

            mask = (keep * 255).astype(np.uint8)
            for sg in sigmas:
                xyz_s = smooth_xyz(xyz_mm, sg)
                gt_mm2, cov = triangle_area_over_region(xyz_s, keep,
                                                        max_gap_mm=gap_max)
                if gt_mm2 <= 0 or cov < 0.9:
                    continue
                gt_cm2 = gt_mm2 / 100.0
                res = integ.integrate(rgb, mask, K, M, ("lidar", depth_m),
                                      normal_smooth_sigma=sg)
                ours = res["area_cm2"]
                rows.append({
                    "scene": tag, "sigma": sg, "radius_mm": args.radius_mm,
                    "n_px": int(keep.sum()),
                    "gt_area_cm2": round(gt_cm2, 4),
                    "our_area_cm2": round(ours, 4),
                    "flat_area_cm2": round(res.get("flat_area_cm2", float('nan')), 4),
                    "rel_err_pct": round(abs(ours - gt_cm2) / gt_cm2 * 100.0, 3),
                })

    if not rows:
        print("❌ no valid samples")
        return 1

    print("\n" + "=" * 74)
    print(f"Real human surface ({args.cloud}) area validation  "
          f"{len({r['scene'] for r in rows})} scenes / "
          f"{len(rows)//max(len(sigmas),1)} patches")
    print("=" * 74)
    print(f"  {'normal σ':>10}{'mean rel err':>14}{'median':>10}{'P90':>10}{'max':>10}"
          f"{'compensation':>10}")
    for sg in sigmas:
        sub = [r for r in rows if r["sigma"] == sg]
        if not sub:
            continue
        e = np.array([r["rel_err_pct"] for r in sub])
        ratio = np.median([r["our_area_cm2"] / max(r["flat_area_cm2"], 1e-9)
                           for r in sub])
        print(f"  {sg:>10.0f}{e.mean():>13.2f}%{np.median(e):>9.2f}%"
              f"{np.percentile(e, 90):>9.2f}%{e.max():>9.2f}%{ratio:>10.3f}")

    best = min(sigmas, key=lambda s: np.median([r["rel_err_pct"]
                                                for r in rows if r["sigma"] == s] or [1e9]))
    sub = [r for r in rows if r["sigma"] == best]
    e = np.array([r["rel_err_pct"] for r in sub])
    gt = np.array([r["gt_area_cm2"] for r in sub])
    ou = np.array([r["our_area_cm2"] for r in sub])
    print(f"\n  optimal σ={best:g}: mean error {e.mean():.2f}% / median {np.median(e):.2f}% "
          f"/ P90 {np.percentile(e, 90):.2f}%")
    print(f"  median true area {np.median(gt):.2f} cm² (range {gt.min():.2f}~{gt.max():.2f})  "
          f"correlation R = {np.corrcoef(gt, ou)[0, 1]:.4f}")

    if args.out:
        json.dump({"config": vars(args), "rows": rows},
                  open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\n✅ details -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
