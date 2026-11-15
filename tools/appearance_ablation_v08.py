#!/usr/bin/env python3
"""Controlled single-variable appearance ablation.

Two purposes:

1. ``--probe``: identify which checkpoint produced the published
   ``runs/area_attribution.json`` (median error 12.27%, Dice 0.906,
   r = -0.947).  This matters because the paper's 1.50% -> 12.27% comparison
   is only a single-variable comparison if both arms use the SAME model.

2. default: run the strict single-variable ablation with that checkpoint.
   Both arms share the same 60 scenes (same seed, geometry, camera, depth,
   expert masks and analytic area); only the wound's colour/texture differs.
   Stream alignment: the patched arm draws one extra RNG integer for patch
   selection, so the unpatched arm draws the same integer after render().
   No synthetic module default is changed.

For every scene both the expert-mask integral and the predicted-mask integral
are computed, so the segmentation contribution is measured inside the scene.

Run examples:
  python tools/appearance_ablation_v08.py --probe
  python tools/appearance_ablation_v08.py --model runs/synth3d/best.pt
Out: runs/appearance_ablation_v08.json  (final ablation only)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wg_train.synth3d import SurfaceSynth  # noqa: E402
from wg_train.metrics import dice  # noqa: E402
from wg_algorithms.surface_area import SurfaceAreaIntegrator  # noqa: E402
from wg_algorithms.segmentation import WoundSegModel, _keep_largest_cc  # noqa: E402

PATCHES = str(ROOT / "data" / "patches_public")
OUT = ROOT / "runs" / "appearance_ablation_v08.json"
REF = ROOT / "runs" / "area_attribution.json"
N = 60
SEED = 7
CANDIDATES = [
    "runs/synth3d/best.pt",
    "runs/wdb_ft/best.pt",
    "runs/publicA_cropped_ft/best.pt",
    "runs/publicA_cropped/best.pt",
    "runs/publicA/best.pt",
    "runs/wdb_ms60/best.pt",
]


def n_patches() -> int:
    return len([f for f in os.listdir(PATCHES)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
                and not f.lower().endswith("_mask.png")])


def run(patch_dir: str | None, seg: WoundSegModel,
        integ: SurfaceAreaIntegrator, npatch: int) -> list[dict]:
    synth = SurfaceSynth(img_size=256, part_type="limb_forearm",
                         seed=SEED, patch_dir=patch_dir)
    recs = []
    for _ in range(N):
        smp = synth.render()
        if patch_dir is None:
            # Align the geometry stream with the patched arm (which draws one
            # integer to pick a patch inside render()).
            synth.rng.integers(0, npatch)
        rgb, gt = smp["rgb"], smp["mask"]
        K, M = smp["K"], smp["M"]
        truth = smp["gt_area_cm2"]

        mask, _ = seg.segment(rgb)
        mask = _keep_largest_cc(mask)
        d = float(dice(mask, gt))
        a_gtmask = integ.integrate(rgb, gt, K, M, ("lidar", smp["depth"]))
        a_pred = integ.integrate(rgb, mask, K, M, ("lidar", smp["depth"]))
        recs.append({
            "dice": d,
            "err": abs(a_pred["area_cm2"] - truth) / max(truth, 1e-9) * 100.0,
            "err_geom": abs(a_gtmask["area_cm2"] - truth) / max(truth, 1e-9) * 100.0,
            "gt_area": float(truth),
        })
    return recs


def summarise(recs: list[dict]) -> dict:
    d = np.array([r["dice"] for r in recs])
    e = np.array([r["err"] for r in recs])
    g = np.array([r["err_geom"] for r in recs])
    return {
        "n": len(recs),
        "dice_median": round(float(np.median(d)), 4),
        "dice_mean": round(float(np.mean(d)), 4),
        "err_median": round(float(np.median(e)), 4),
        "err_mean": round(float(np.mean(e)), 4),
        "geom_floor_median": round(float(np.median(g)), 4),
        "corr_dice_err": round(float(np.corrcoef(d, e)[0, 1]), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="identify the checkpoint that reproduces area_attribution.json")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    assert Path(PATCHES).is_dir(), f"patch directory missing: {PATCHES}"
    npatch = n_patches()
    ref = json.loads(REF.read_text()) if REF.exists() else {}
    if ref:
        r = ref["records"]
        print("REFERENCE runs/area_attribution.json:",
              summarise(r), "\n")

    models = CANDIDATES if args.probe else [args.model or "runs/synth3d/best.pt"]
    results = {}
    for rel in models:
        path = ROOT / rel
        if not path.is_file():
            print(f"skip (missing): {rel}")
            continue
        seg = WoundSegModel(weights_path=str(path), device="auto", img_size=256)
        integ = SurfaceAreaIntegrator()
        patched = run(PATCHES, seg, integ, npatch)
        s = summarise(patched)
        results[rel] = s
        print(f"{rel:34s} {s}", flush=True)
        if not args.probe:
            plain = run(None, seg, integ, npatch)
            a = np.array([x["gt_area"] for x in plain])
            b = np.array([x["gt_area"] for x in patched])
            assert np.allclose(a, b, atol=1e-9), \
                "scene geometry drifted between arms; ablation is not controlled"
            print(f"controlled: {N} scenes, max |dA_GT| = {np.abs(a-b).max():.3e} cm²")
            sp = summarise(plain)
            out = {
                "protocol": {
                    "checkpoint": rel, "n": N, "seed": SEED,
                    "patch_dir": os.path.relpath(PATCHES, ROOT),
                    "img_size": 256, "part_type": "limb_forearm",
                    "arms": ["synthetic appearance", "real wound appearance"],
                    "stream_alignment": "unpatched arm draws one matching RNG integer after render",
                    "single_variable_check": "gt_area_cm2 sequences identical to 1e-9",
                },
                "synthetic_appearance": sp,
                "real_appearance": s,
                "delta": {
                    "err_median_pct_points": round(s["err_median"] - sp["err_median"], 4),
                    "dice_median": round(s["dice_median"] - sp["dice_median"], 4),
                    "geom_floor_median": round(s["geom_floor_median"] - sp["geom_floor_median"], 4),
                },
                "records": {"synthetic_appearance": plain,
                            "real_appearance": patched},
            }
            OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"synthetic_appearance": sp, "real_appearance": s,
                              "delta": out["delta"]}, indent=2))
            print(f"-> {OUT}")

    if args.probe:
        tgt = summarise(json.loads(REF.read_text())["records"])["err_median"]
        best = min(results.items(),
                   key=lambda kv: abs(kv[1]["err_median"] - tgt), default=(None, None))
        print(f"\nclosest to reference {tgt}%: {best[0]} -> {best[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
