#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_area_vs_dice.py — attribute area error to two-dimensional outline quality.

Why this script is needed:
    The headline number of the end-to-end area error (e.g. 44%) is almost useless — it lumps
    together "segmentation failed completely" and "segmentation is slightly off". The truly
    useful question is:
        **At what level of segmentation does the area error fall into a clinically
        acceptable range?**
    On synthetic surfaces with "known geometric ground truth", this script records both
    dice and relative area error for the same set of samples, stratifies by dice, and
    gives this correspondence.

Usage:
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

        # area under the ground-truth mask (isolating geometric/integration error)
        a_gtmask = integ.integrate(rgb, gt, K, M, ("lidar", smp["depth"]))
        # area under the model mask (end-to-end)
        a_pred = integ.integrate(rgb, mask, K, M, ("lidar", smp["depth"]))
        err = abs(a_pred["area_cm2"] - truth) / max(truth, 1e-9) * 100.0
        err_geom = abs(a_gtmask["area_cm2"] - truth) / max(truth, 1e-9) * 100.0
        recs.append({"dice": float(d), "err": float(err),
                     "err_geom": float(err_geom), "gt_area": float(truth)})

    errs = np.array([r["err"] for r in recs])
    errs_g = np.array([r["err_geom"] for r in recs])
    dices = np.array([r["dice"] for r in recs])

    print(f"\n══ Area error attribution (n={args.n}, region={args.part_type}) ══\n")
    print(f"  Pure geometric/integration error (ground-truth mask): mean {errs_g.mean():5.2f}%  "
          f"median {np.median(errs_g):5.2f}%   max {errs_g.max():5.2f}%")
    print(f"  End-to-end (model segmentation): mean {errs.mean():5.2f}%  "
          f"median {np.median(errs):5.2f}%   max {errs.max():5.2f}%")

    print("\n  ── Stratified by segmentation quality → relative area error ──")
    print(f"  {'dice interval':<14}{'samples':>5}{'mean area error':>14}{'median':>10}{'max':>10}")
    bins = [(0.95, 1.01, "≥0.95"), (0.90, 0.95, "0.90-0.95"),
            (0.80, 0.90, "0.80-0.90"), (0.50, 0.80, "0.50-0.80"),
            (0.0, 0.50, "<0.50 failure")]
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
        print(f"\n  Among the {int(ok.sum())}/{args.n} samples that meet the segmentation "
              "target (dice≥0.90), "
              f"median area error {np.median(errs[ok]):.2f}%")
    print("\n  Conclusion/convention: area error ≈ amplification of segmentation error; the "
          "'geometric error' in the table is the theoretical lower bound of this chain;\n"
          "  for end-to-end to approach it, segmentation must stay stably above dice 0.95.")

    gts = np.array([r["gt_area"] for r in recs])

    # ---- correlation + wound-size stratification (material for the MedIA attribution Figure) ----
    corr = float(np.corrcoef(dices, errs)[0, 1]) if len(dices) > 2 else float("nan")
    print(f"\n  corr(dice, area error) = {corr:+.3f}   (n={len(dices)})")

    med = float(np.median(gts))
    small = gts < med
    if int(small.sum()) and int((~small).sum()):
        print(f"\n  ── Stratified by wound size (adaptive threshold {med:.1f} cm²) ──")
        for lab, sel in (("small wound", small), ("large wound", ~small)):
            print(f"  {lab:<8} n={int(sel.sum()):>3}  "
                  f"median dice {np.median(dices[sel]):.3f}  "
                  f"median area error {np.median(errs[sel]):.2f}%  mean {errs[sel].mean():.2f}%")

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
    print(f"\n  details + summary written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
