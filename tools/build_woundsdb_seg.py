#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_woundsdb_seg.py — build WoundsDB segmentation crops from expert masks.

Input: data/raw/woundsdb/patients_partial/patients/case_X/day_Y/data/scene_N/{photo.jpg,mask.png}
Output: data/woundsdb_seg/{images,masks}/caseX__dayY__sceneZ.{jpg,png}
        + split.json (train/val/test grouped by case, to prevent patient-level leakage)

Key decisions:
  - Downsample to a long edge of 1600: 4896px originals are resized to 256 during training
    anyway, so downsampling directly saves 10x disk space while the mask boundary is still
    adequate (wound ~40px at 1600px);
  - Black-border cropping uses the same convention as tools/crop_black_border.py
    (consecutive pure-black rows/columns);
  - All days/scenes of the same case go into the same group — a natural safeguard against
    patient-level leakage;
  - Also run a mask health check (foreground fraction, binariness); abnormal samples are
    removed and recorded.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT_ROOT = os.path.join(ROOT, "data/raw/woundsdb/patients_partial/patients")
OUT = os.path.join(ROOT, "data/woundsdb_seg")
LONG_EDGE = 1600


def black_borders(a: np.ndarray, thr: int = 12):
    m = (a.max(axis=2) <= thr)
    if m.all():
        return 0, 0, 0, 0

    def run(bits):
        n = 0
        for b in bits:
            if b:
                n += 1
            else:
                break
        return n

    t, b = run(m.all(axis=1)), run(m.all(axis=1)[::-1])
    l, r = run(m.all(axis=0)), run(m.all(axis=0)[::-1])
    return (t if t >= 2 else 0, b if b >= 2 else 0,
            l if l >= 2 else 0, r if r >= 2 else 0)


def roi_box(mask: np.ndarray, rng: np.random.Generator,
            scale_lo: float, scale_hi: float, jitter: float,
            min_side: int = 64):
    """Crop a square ROI from the GT bounding box — simulating "the user pointing the camera at the wound" framing.

    Convention note (important): this uses the GT **location** information, but the model
    must still decide where the contour is — equivalent to "detection given, segmentation
    only" in two-stage instance segmentation, with exactly the same convention as the roi
    mode of the end-to-end validation e2e_woundsdb.py. GT does not take part in
    thresholding / area / post-processing, so it does not constitute an evaluation leak.

    Returns (l, t, r, b); out-of-bounds boxes are shifted inward; returns None if the
    image is smaller than the ROI.
    """
    ys, xs = np.nonzero(mask > 127)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    side = int(round(max(bw, bh) * rng.uniform(scale_lo, scale_hi)))
    side = max(side, min_side)
    h, w = mask.shape[:2]
    side = min(side, h, w)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    off = side * jitter
    cx += rng.uniform(-off, off)
    cy += rng.uniform(-off, off)
    l = int(round(cx - side / 2.0))
    t = int(round(cy - side / 2.0))
    # shift out-of-bounds boxes inward (do not pad with black borders — black borders have
    # been shown to be a shortcut feature for the model)
    l = max(0, min(l, w - side))
    t = max(0, min(t, h - side))
    return l, t, l + side, t + side


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", action="store_true",
                    help="crop a close-up ROI from the GT bounding box (otherwise output the distant full frame)")
    ap.add_argument("--roi_scale", default="1.6,2.6",
                    help="training-set ROI scale: comma-separated multiple scales enable "
                         "multi-scale augmentation, or two values mean random sampling "
                         "within that interval")
    ap.add_argument("--val_scale", type=float, default=2.5,
                    help="fixed ROI scale for the val/test groups (same value as e2e inference, to ensure comparability)")
    ap.add_argument("--roi_jitter", type=float, default=0.08,
                    help="random offset ratio of the ROI centre (simulating framing error); 0=strictly centred")
    ap.add_argument("--roi_side", type=int, default=512,
                    help="uniform output side length of the ROI")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="output directory (default data/woundsdb_seg[_roi])")
    args = ap.parse_args()
    raw = [float(x) for x in args.roi_scale.split(",")]
    if len(raw) > 2:
        # multi-scale augmentation: each training sample produces len(raw) copies at
        # different framing scales
        tr_scales = raw
        lo_s, hi_s = raw[0], raw[-1]
    else:
        tr_scales = None
        lo_s, hi_s = raw[0], raw[-1]

    out_dir = args.out or (OUT + ("_roi" if args.roi else ""))
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    cases = sorted({p.split(os.sep)[-4] for p in
                    glob.glob(os.path.join(PAT_ROOT, "case_*", "day_*", "data", "scene_*"))})
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(cases))
    n_tr = int(len(cases) * 0.70)
    n_va = int(len(cases) * 0.15)
    split = {"train": sorted(cases[i] for i in perm[:n_tr]),
             "val": sorted(cases[i] for i in perm[n_tr:n_tr + n_va]),
             "test": sorted(cases[i] for i in perm[n_tr + n_va:])}

    rows, bad = [], []
    counts = {k: 0 for k in split}
    for mp in sorted(glob.glob(os.path.join(PAT_ROOT, "case_*", "day_*",
                                            "data", "scene_*", "mask.png"))):
        d = os.path.dirname(mp)
        case, day, scene = d.split(os.sep)[-4], d.split(os.sep)[-3], d.split(os.sep)[-1]
        stem = f"{case}__{day}__{scene}"
        grp = next(g for g in ("train", "val", "test") if case in split[g])

        photo = np.asarray(Image.open(os.path.join(d, "photo.jpg")).convert("RGB"))
        mask = np.asarray(Image.open(mp).convert("L"))
        if mask.shape != photo.shape[:2]:
            bad.append({"stem": stem, "why": f"size mismatch {mask.shape} vs {photo.shape}"})
            continue
        t, b, l, r = black_borders(photo)
        h, w = photo.shape[:2]
        photo = photo[t:h - b, l:w - r]
        mask = mask[t:h - b, l:w - r]

        uniq = np.unique(mask)
        if not np.isin(uniq, (0, 255)).all():
            bad.append({"stem": stem, "why": f"non-binary uniq {uniq[:5]}"})
            continue
        # in ROI mode, the foreground fraction must be computed **after cropping** — a
        # distant full frame has only 0.03%
        frac = float((mask > 127).mean())
        if not args.roi and not (1e-4 <= frac <= 0.5):
            bad.append({"stem": stem, "why": f"full-frame foreground {frac:.4f}"})
            continue

        # ---- generate several variants of this sample
        # training set: multi-scale augmentation (crop the same GT at different framing
        # scales, both expanding the data and improving framing robustness)
        # val/test: fixed 2.5x, matching the e2e inference value for cross-configuration comparability
        if not args.roi:
            variants = [(stem, None)]
        elif grp == "train" and tr_scales:
            variants = [(f"{stem}__s{i}", (sc, sc, args.roi_jitter))
                        for i, sc in enumerate(tr_scales)]
        else:
            variants = [(stem, (args.val_scale, args.val_scale, 0.0))]

        for stem_out, spec in variants:
            ph, mk, fr = photo, mask, frac
            if spec is not None:
                sclo, schi, jit = spec
                box = roi_box(mk, rng, sclo, schi, jit)
                if box is None:
                    bad.append({"stem": stem_out, "why": "empty mask"})
                    continue
                l, t, r, b = box
                ph = ph[t:b, l:r]
                mk = mk[t:b, l:r]
                ph = np.asarray(Image.fromarray(ph).resize(
                    (args.roi_side, args.roi_side), Image.LANCZOS))
                mk = np.asarray(Image.fromarray(mk).resize(
                    (args.roi_side, args.roi_side), Image.NEAREST))
                fr = float((mk > 127).mean())
                if not (0.005 <= fr <= 0.6):
                    bad.append({"stem": stem_out, "why": f"ROI foreground {fr:.4f}"})
                    continue
            else:
                sc = LONG_EDGE / max(ph.shape[:2])
                if sc < 1:
                    ph = np.asarray(Image.fromarray(ph).resize(
                    (round(ph.shape[1] * sc), round(ph.shape[0] * sc)),
                    Image.LANCZOS))
                    mk = np.asarray(Image.fromarray(mk).resize(
                    (ph.shape[1], ph.shape[0]), Image.NEAREST))

            Image.fromarray(ph).save(os.path.join(out_dir, "images", stem_out + ".jpg"), quality=92)
            Image.fromarray(mk).save(os.path.join(out_dir, "masks", stem_out + ".png"))
            rows.append({"stem": stem_out, "case": case, "day": day, "scene": scene,
                         "group": grp, "fg_frac": round(fr, 5),
                         "size": [ph.shape[1], ph.shape[0]]})
            counts[grp] += 1

    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({"split": split, "samples": rows, "bad": bad,
                   "config": {"roi": args.roi, "roi_scale": [lo_s, hi_s],
                              "roi_jitter": args.roi_jitter,
                              "roi_side": args.roi_side, "seed": args.seed}}, f,
                  indent=2, ensure_ascii=False)
    print(f"✅ wrote {len(rows)} pairs (train {counts['train']} / val {counts['val']} "
          f"/ test {counts['test']}), removed {len(bad)} abnormal pairs")
    if bad:
        for x in bad[:5]:
            print("   [!]", x)
    print(f"   -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
