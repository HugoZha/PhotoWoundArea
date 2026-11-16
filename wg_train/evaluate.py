#!/usr/bin/env python3
"""
evaluate.py — segmentation evaluation CLI with an optional area-chain check
===================================================

Usage (synthetic validation set, same distribution as train but a different seed):
    python -m wg_train.evaluate --model runs/exp1/best.pt --synthetic 60 --out runs/exp1/eval

Usage (real validation set):
    python -m wg_train.evaluate --model runs/exp1/best.pt \
        --image_dir data/val_images --mask_dir data/val_masks

Optional area-chain check (predicted mask -> curved-surface area):
    ... --area_demo

Output:
    <out>/metrics.json        per-metric means
    <out>/samples_*.png       prediction/ground-truth overlay visualizations (up to 8 images)
    <out>/area_demo.json      area-chain demo results (if enabled)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from wg_algorithms.seg_model import load_wound_model, predict_proba, HAS_TORCH


def _json_safe(obj):
    """Convert inf/nan to null.

    Why this is needed: Python's json.dump by default writes `Infinity` / `NaN`, which are
    **non-standard JSON** literals that parsers in other languages reject outright.
    HD95 becomes inf on a complete miss, so this step is necessary.
    """
    import math
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _auto_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if HAS_TORCH:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _save_overlay(path, rgb, pred, gt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(rgb); ax[0].set_title("RGB"); ax[0].axis("off")
    ax[1].imshow(rgb); ax[1].imshow(pred > 0, alpha=0.4, cmap="Reds")
    ax[1].set_title("Pred"); ax[1].axis("off")
    ax[2].imshow(rgb); ax[2].imshow(gt > 0, alpha=0.4, cmap="Greens")
    ax[2].set_title("GT"); ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def area_demo(model, device, img_size: int = 256, distance: float = 0.30, fx: float = 1500.0):
    """Validate "real model mask -> 3D surface area" end-to-end: synthesize a wound, predict its mask with the model,
    then use SurfaceAreaIntegrator on a known plane (at distance D) to compute the real area and compare it with the GT mask area."""
    from wg_algorithms.segmentation import _keep_largest_cc
    from wg_algorithms.geometry import intrinsics_to_K
    from wg_algorithms.curvature_prior import PlaneModel
    from wg_algorithms.surface_area import SurfaceAreaIntegrator
    from wg_train import synthesize_data

    rgb, gt = synthesize_data.generate_pair(
        img_size=img_size,
        rng=np.random.default_rng(12345), wound_frac=0.20)
    proba = predict_proba(model, rgb, device, img_size)
    pred = (proba >= 0.5).astype(np.uint8) * 255
    pred = _keep_largest_cc(pred)

    K = intrinsics_to_K(fx, fx, img_size / 2, img_size / 2)
    M = np.eye(4)
    plane = PlaneModel([0, 0, distance], [0, 0, 1])
    integ = SurfaceAreaIntegrator()
    zeros = np.zeros((img_size, img_size, 3), np.uint8)
    a_pred = integ.integrate(zeros, pred, K, M, ("model", plane))
    a_gt = integ.integrate(zeros, gt, K, M, ("model", plane))
    return {
        "distance_m": distance,
        "pred_area_cm2": a_pred["area_cm2"],
        "gt_area_cm2": a_gt["area_cm2"],
        "pred_px": int((pred > 0).sum()),
        "gt_px": int((gt > 0).sum()),
        "rel_err_pct": abs(a_pred["area_cm2"] - a_gt["area_cm2"]) /
                       max(a_gt["area_cm2"], 1e-6) * 100.0,
    }


def _forward_proba(model, x, tta: bool = False):
    """Forward pass to obtain the probability map [H,W].

    With tta=True, average three forward passes ("horizontal + vertical flips" + original).
    Why it is worth doing: it needs no retraining and costs only 3× inference time,
    and for small targets sensitive to boundary jitter (in this project the wound median occupies only 6.8% of the frame)
    it usually gives a small, stable dice gain;
    moreover, since the area is integrated from mask pixels, a steadier boundary directly means a smaller area error.
    """
    import torch
    with torch.no_grad():
        p = model(x)[0, 0]
        if not tta:
            return p
        ps = [p]
        for dim in (3, 2):                       # 3=W flip, 2=H flip
            pf = model(torch.flip(x, [dim]))[0, 0]
            ps.append(torch.flip(pf, [dim - 2]))  # flip back to the original orientation before averaging
        return torch.stack(ps).mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to best.pt / last.pt")
    ap.add_argument("--synthetic", type=int, default=0, help="Number of synthetic validation samples (different seed from train)")
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--mask_dir", default=None)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1000, help="Random seed for the synthetic validation set (avoid the training seed)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/exp1/eval")
    ap.add_argument("--area_demo", action="store_true", help="Also run the area-chain demo")
    ap.add_argument("--no_largest_cc", action="store_true",
                    help="Disable the 'keep only the largest connected component' postprocessing (enable this for validation sets with multiple lesions)")
    ap.add_argument("--tta", action="store_true", help="Flip test-time augmentation (horizontal + vertical, probability maps averaged)")
    ap.add_argument("--thr", type=float, default=0.5, help="Binarization threshold (default 0.5)")
    ap.add_argument("--tune_thr", action="store_true",
                    help="Search for the optimal threshold on this set ⚠️ only do this on the [validation set]; "
                         "tuning on the test set leaks test-set information into the evaluation")
    args = ap.parse_args()

    if not HAS_TORCH:
        print("❌ Evaluation requires torch; please run `pip install torch` first.")
        sys.exit(1)

    device = _auto_device(args.device)
    model = load_wound_model(args.model, device=device)
    print(f"▶ evaluating model: {args.model} | device: {device}")

    from torch.utils.data import DataLoader
    from wg_train.dataset import ImageMaskDataset, SyntheticSegDataset
    from wg_train.metrics import evaluate_pairs

    if args.synthetic > 0:
        ds = SyntheticSegDataset(n=args.synthetic, img_size=args.img_size,
                                 augment=False, seed=args.seed, wound_frac=0.20)
    else:
        ds = ImageMaskDataset(args.image_dir, args.mask_dir, img_size=args.img_size,
                              augment=False, seed=args.seed)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    preds, gts = [], []
    os.makedirs(args.out, exist_ok=True)
    n_save = 0
    keep_cc = None
    if not args.no_largest_cc:
        from wg_algorithms.segmentation import _keep_largest_cc as keep_cc

    # Collect probability maps first: threshold tuning needs the same predictions recomputed at different thresholds; you cannot pick the threshold while inferring
    probas, gts_raw, rgbs = [], [], []
    for i, batch in enumerate(ld):
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        probas.append(_forward_proba(model, x, tta=args.tta).cpu().numpy())
        gts_raw.append((y[0, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255)
        if n_save < 8:
            rgbs.append((x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        n_save += 1

    def _thr_metrics(thr: float, cheap: bool = False):
        """cheap=True computes only dice — threshold tuning sweeps 19 thresholds,
        and the full metric suite's HD95/boundary-F1 is too slow to be worth paying for here."""
        ps, gs = [], []
        for pb, gt in zip(probas, gts_raw):
            pred = (pb >= thr).astype(np.uint8) * 255
            if keep_cc is not None:
                pred = keep_cc(pred)
            ps.append(pred); gs.append(gt)
        if cheap:
            from wg_train.metrics import dice
            return ps, gs, {"dice": float(np.mean([dice(p, g) for p, g in zip(ps, gs)]))}
        return ps, gs, evaluate_pairs(ps, gs)

    thr = args.thr
    if args.tune_thr:
        best = (-1.0, args.thr, None)
        sweep = [round(0.05 * k, 2) for k in range(1, 20)]
        print("  Tuning threshold (this set, dice criterion only)…")
        for t in sweep:
            d = _thr_metrics(t, cheap=True)[2]["dice"]
            if d > best[0]:
                best = (d, t, None)
        thr = best[1]
        print(f"  Optimal threshold {thr:.2f} (dice {best[0]:.4f}), "
              f"vs. default 0.5: {_thr_metrics(0.5, cheap=True)[2]['dice']:.4f}")
        print("  ⚠️ This threshold was selected on the validation set; when evaluating the test set, pass it explicitly via --thr")

    preds, gts, metrics = _thr_metrics(thr)
    metrics["threshold"] = thr
    metrics["tta"] = bool(args.tta)

    for i in range(min(8, len(rgbs))):
        _save_overlay(os.path.join(args.out, f"samples_{i:02d}.png"), rgbs[i], preds[i], gts[i])

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    print("\n" + "=" * 56)
    print("Evaluation metrics (validation-set means)")
    print("=" * 56)
    for k, v in metrics.items():
        print(f"  {k:>12}: {v}")
    print(f"\n✅ Metrics written to {os.path.join(args.out, 'metrics.json')}")

    if args.area_demo:
        demo = area_demo(model, device, img_size=args.img_size)
        with open(os.path.join(args.out, "area_demo.json"), "w") as f:
            json.dump(demo, f, indent=2)
        print("\n" + "=" * 56)
        print("Area-chain demo (synthetic wound @ plane distance %.2f m)" % demo["distance_m"])
        print("=" * 56)
        print(f"  GT mask area       : {demo['gt_area_cm2']:.2f} cm²  (pixels {demo['gt_px']})")
        print(f"  predicted area     : {demo['pred_area_cm2']:.2f} cm²  (pixels {demo['pred_px']})")
        print(f"  relative error     : {demo['rel_err_pct']:.2f}%")
        print(f"  → The real model mask drove the 3D area computation; the error mainly comes from the segmentation boundary and the assumed plane.")
        print(f"✅ Area demo written to {os.path.join(args.out, 'area_demo.json')}")


if __name__ == "__main__":
    main()
