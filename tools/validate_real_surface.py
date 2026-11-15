#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""validate_real_surface.py — validate the surface-area integrator on real skin point clouds.

背景（为什么这一步重要）：
    此前我们所有三维面积验证都是**合成几何**（平面/圆柱/球面）。那些实验能证明公式
    数学上没写错，但证明不了它对**真实皮肤**有效——真皮肤有褶皱、弧度不规则、
    溃疡边缘还有陡坎。公开数据里能同时提供「三维点云 + 专家掩膜」的资源极少，
    WoundsDB 是其中一个：它的每个 scene 都带 stereo-mesh.ply / depth-mesh.ply，
    是**按相机网格排列的结构化点云**，xyz 单位是毫米。

做法（不需要掩膜就能验证）：
    1. 点云本身就可以当作一台 pinhole 相机看到的深度图：网格坐标 (row,col) 就是像素，
       每个像素有真实的 3D 点。用最小二乘从 x=(u-cx)·z/fx 反解出这台虚拟相机的内参
       fx, fy, cx, cy —— 这样我们**不依赖任何外部标定**就能自洽地建立成像模型。
    2. 在点云上取一块圆形区域（半径 R 毫米，按三维距离而不是像素距离取，
       所以它在真实曲面上≈一个圆盘），
    3. 真实面积 = 把这块区域的三维点做三角剖分后求和（Ground Truth），
    4. 我们的算法 = SurfaceAreaIntegrator 走 'lidar' 分支，
       喂同一个虚拟相机的 K 和同一份深度图，算出来的面积，
    5. 对比两者的相对误差。

这一步验证的是「同一份真实几何，我们的公式能不能算回它的面积」，
**不含分割误差** —— 分割误差是另一回事（见 analyze_area_vs_dice.py）。

用法：
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


# ------------------------------------------------------------------ PLY 读取
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
        raise ValueError(f"{path}: {nv} 个点与 {h}x{w} 不符")
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
    # 有效格数 / 本应覆盖的格数 —— 覆盖率低的区域说明点云本身质量差，
    # 这时"部分计数"会让真值严重偏低（实测出现"真值 < 投影面积"这种几何不可能的结果），
    # 正确做法是整块丢弃而不是硬凑。
    expect = max(float((0.25 * (k[:-1, :-1] + k[1:, :-1] + k[:-1, 1:]
                                + k[1:, 1:])[:, :, 0]).sum()), 1e-9)
    return area, float(wgt[:, :, 0].sum() / expect)


def nominal_spacing(xyz) -> float:
    """点云正常点距中位（毫米）—— 用来设定"离群点"判定阈值。"""
    d = np.linalg.norm(np.diff(xyz, axis=1), axis=2)
    return float(np.median(d))


def smooth_xyz(xyz, sigma):
    """同尺度平滑点云本身，用来得到与之匹配的 Ground Truth。

    为什么 GT 也要平滑：不同尺度下"曲面面积"不是一个数——越细看面积越大
    （皮肤微观褶皱会贡献面积）。临床要测的是**宏观**面积，
    所以「用平滑 sigma 求法向」必须和「用同样 sigma 的曲面当真值」配对比较。
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
    ap.add_argument("--scenes", type=int, default=6, help="取几个 scene 做实验")
    ap.add_argument("--patches", type=int, default=8, help="每个 scene 取几块区域")
    ap.add_argument("--radius_mm", type=float, default=15.0, help="区域半径（毫米）")
    ap.add_argument("--cloud", default="stereo-mesh.ply",
                    choices=["stereo-mesh.ply", "depth-mesh.ply"])
    ap.add_argument("--sigmas", default="0,1,2,3,5,8",
                    help="对比的法向平滑强度（像素）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="结果 JSON")
    args = ap.parse_args()

    shape = {"stereo-mesh.ply": (1024, 768), "depth-mesh.ply": (176, 144)}[args.cloud]
    scenes = sorted(glob.glob(os.path.join(args.root, "case_*", "day_*", "results",
                                           "scene_*")))
    scenes = [s for s in scenes if os.path.exists(os.path.join(s, args.cloud))]
    if not scenes:
        print(f"❌ {args.root} 下没找到含 {args.cloud} 的 scene")
        return 1
    scenes = scenes[: args.scenes]
    sigmas = [float(x) for x in args.sigmas.split(",") if x != ""]
    print(f"▶ 真实曲面验证：{len(scenes)} 个 scene / {args.cloud} / "
          f"每 scene {args.patches} 块 R={args.radius_mm}mm 区域\n")

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
        xyz_mm = xyz_mm[:, :, list(perm)]          # 摆正为标准针孔系
        fx, fy, cx, cy = Kc
        K = intrinsics_to_K(fx, fy, cx, cy)
        depth_m = xyz_mm[:, :, 2] / 1000.0          # 毫米 -> 米（算法约定）
        z = xyz_mm[:, :, 2]
        M = np.eye(4)                               # 点云已在相机系，相机即原点
        gap_max = nominal_spacing(xyz_mm) * 4.0     # 超过 4 倍正常点距的格子视为离群

        case = re.search(r"case_(\d+)", sdir)
        day = re.search(r"day_(\d+)", sdir)
        scene = re.search(r"scene_(\d+)", sdir)
        tag = (f"case_{case.group(1) if case else '?'}__"
               f"day_{day.group(1) if day else '?'}__"
               f"scene_{scene.group(1) if scene else '?'}")

        # ---- 先定 patch（与 sigma 无关，保证不同 sigma 比较同一块皮肤）
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
              f"fx={fx:.0f} cx={cx:.0f} cy={cy:.0f} 轴序{perm} "
              f"残差 {resid:.3f}px 深度中位 {np.median(depth_m)*1000:.0f}mm "
              f"({len(patches)} 块)")

        for (cy0, cx0, keep) in patches:
            # 先用**未平滑**点云判定这块皮肤本身质量够不够好：
            # 覆盖率不足 90% 说明此处点云有空洞/离群，直接整块跳过，
            # 且这样各 sigma 用的是同一批 patch，比较才公平。
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
        print("❌ 没有有效样本")
        return 1

    print("\n" + "=" * 74)
    print(f"真实人体曲面（{args.cloud}）面积验证  "
          f"{len({r['scene'] for r in rows})} scene / {len(rows)//max(len(sigmas),1)} 块区域")
    print("=" * 74)
    print(f"  {'法向平滑σ':>10}{'相对误差 均值':>14}{'中位':>10}{'P90':>10}{'最大':>10}"
          f"{'补偿倍数':>10}")
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
    print(f"\n  最优 σ={best:g}：误差均值 {e.mean():.2f}% / 中位 {np.median(e):.2f}% "
          f"/ P90 {np.percentile(e, 90):.2f}%")
    print(f"  真实面积中位 {np.median(gt):.2f} cm²（范围 {gt.min():.2f}~{gt.max():.2f}）  "
          f"相关性 R = {np.corrcoef(gt, ou)[0, 1]:.4f}")

    if args.out:
        json.dump({"config": vars(args), "rows": rows},
                  open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\n✅ 明细 -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
