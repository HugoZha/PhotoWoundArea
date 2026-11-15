#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_area_vs_dice.py — attribute area error to two-dimensional outline quality.

为什么需要这个脚本：
    端到端面积误差的大盘数字（比如 44%）几乎没用 —— 它把"分割彻底失败"和
    "分割略有偏差"混在一起。真正有用的问题是：
        **当分割达到什么水平时，面积误差才能落到临床可接受区间？**
    本脚本在「已知几何真值」的合成曲面上，对同一批样本同时记录
    dice 与面积相对误差，按 dice 分档统计，给出这个对应关系。

用法：
    python tools/analyze_area_vs_dice.py --model runs/publicA/best.pt \
        --patch_dir data/patches_public --n 40
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--patch_dir", default=None)
    ap.add_argument("--part_type", default="limb_forearm")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from wg_train.synth3d import SurfaceSynth
    from wg_train.metrics import dice
    from wg_algorithms.surface_area import SurfaceAreaIntegrator
    from wg_algorithms.segmentation import WoundSegModel, _keep_largest_cc

    seg = WoundSegModel(weights_path=args.model, device="auto",
                        img_size=args.img_size)
    integ = SurfaceAreaIntegrator()
    synth = SurfaceSynth(img_size=args.img_size, part_type=args.part_type,
                         seed=args.seed, patch_dir=args.patch_dir)

    recs = []
    for i in range(args.n):
        smp = synth.render()
        rgb, gt = smp["rgb"], smp["mask"]
        K, M = smp["K"], smp["M"]
        truth = smp["gt_area_cm2"]

        mask, _info = seg.segment(rgb)
        mask = _keep_largest_cc(mask)
        d = dice(mask, gt)

        # 真值掩膜下的面积（隔离几何/积分误差）
        a_gtmask = integ.integrate(rgb, gt, K, M, ("lidar", smp["depth"]))
        # 模型掩膜下的面积（端到端）
        a_pred = integ.integrate(rgb, mask, K, M, ("lidar", smp["depth"]))
        err = abs(a_pred["area_cm2"] - truth) / max(truth, 1e-9) * 100.0
        err_geom = abs(a_gtmask["area_cm2"] - truth) / max(truth, 1e-9) * 100.0
        recs.append({"dice": float(d), "err": float(err),
                     "err_geom": float(err_geom), "gt_area": float(truth)})

    errs = np.array([r["err"] for r in recs])
    errs_g = np.array([r["err_geom"] for r in recs])
    dices = np.array([r["dice"] for r in recs])

    print(f"\n══ 面积误差归因（n={args.n}，部位={args.part_type}）══\n")
    print(f"  纯几何/积分误差（用真值掩膜）: 均值 {errs_g.mean():5.2f}%  "
          f"中位 {np.median(errs_g):5.2f}%   max {errs_g.max():5.2f}%")
    print(f"  端到端（模型分割）          : 均值 {errs.mean():5.2f}%  "
          f"中位 {np.median(errs):5.2f}%   max {errs.max():5.2f}%")

    print("\n  ── 按分割质量分档 → 面积相对误差 ──")
    print(f"  {'dice 区间':<14}{'样本':>5}{'面积误差均值':>14}{'中位':>10}{'最大':>10}")
    bins = [(0.95, 1.01, "≥0.95"), (0.90, 0.95, "0.90-0.95"),
            (0.80, 0.90, "0.80-0.90"), (0.50, 0.80, "0.50-0.80"),
            (0.0, 0.50, "<0.50 失败")]
    for lo, hi, label in bins:
        sel = (dices >= lo) & (dices < hi)
        if sel.sum() == 0:
            print(f"  {label:<14}{0:>5}{'—':>14}{'—':>10}{'—':>10}")
            continue
        e = errs[sel]
        print(f"  {label:<14}{int(sel.sum()):>5}{e.mean():>13.2f}%"
              f"{np.median(e):>9.2f}%{e.max():>9.2f}%")

    ok = dices >= 0.90
    if ok.sum():
        print(f"\n  分割达标(dice≥0.90)的 {int(ok.sum())}/{args.n} 个样本中，"
              f"面积误差中位 {np.median(errs[ok]):.2f}%")
    print("\n  结论口径：面积误差 ≈ 分割误差的放大；表中的'几何误差'是这条链路的"
          "\n  理论下限，端到端要接近它，前提是分割稳定在 dice 0.95 以上。")

    gts = np.array([r["gt_area"] for r in recs])

    # ---- 相关性 + 创面大小分层（MedIA 归因 Figure 素材）----
    corr = float(np.corrcoef(dices, errs)[0, 1]) if len(dices) > 2 else float("nan")
    print(f"\n  corr(dice, 面积误差) = {corr:+.3f}   (n={len(dices)})")

    med = float(np.median(gts))
    small = gts < med
    if int(small.sum()) and int((~small).sum()):
        print(f"\n  ── 按创面大小分层（自适应阈值 {med:.1f} cm²）──")
        for lab, sel in (("小创面", small), ("大创面", ~small)):
            print(f"  {lab:<8} n={int(sel.sum()):>3}  "
                  f"dice 中位 {np.median(dices[sel]):.3f}  "
                  f"面积误差 中位 {np.median(errs[sel]):.2f}%  均值 {errs[sel].mean():.2f}%")

    summary = {
        "n": args.n,
        "corr_dice_err": corr,
        "size_threshold_cm2": round(med, 3),
        "geometry_error_median_pct": float(np.median(errs_g)),
        "e2e_error_median_pct": float(np.median(errs)),
        "small_wound": {"n": int(small.sum()),
                        "dice_median": float(np.median(dices[small])),
                        "area_err_median_pct": float(np.median(errs[small])),
                        "area_err_mean_pct": float(errs[small].mean())},
        "large_wound": {"n": int((~small).sum()),
                        "dice_median": float(np.median(dices[~small])),
                        "area_err_median_pct": float(np.median(errs[~small])),
                        "area_err_mean_pct": float(errs[~small].mean())},
    }

    out = os.path.join(ROOT, "runs", "area_attribution.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import json
    with open(out, "w") as f:
        json.dump({"summary": summary, "records": recs}, f, indent=2)
    print(f"\n  明细+摘要已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
