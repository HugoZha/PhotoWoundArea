#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_mix.py — build the mixed public/WoundsDB training set with domain weighting.

Rationale:
  WoundsDB has only 188 expert masks, and in distant full frames the wound occupies
  merely 0.03%~0.3% of the pixels. Training directly on full frames learns nothing
  (val_dice≈0). Hence:
    1) crop ROIs from the GT bounding boxes (build_woundsdb_seg.py --roi) to raise the
       proportion to ~6%;
    2) multi-scale augmentation (crop the same GT at multiple framing scales) expands the
       training pairs from 112 to 558;
    3) domain weighting: control the sampling ratio of public : woundsdb
       (w_repeat / n_pub).

In implementation, "weighted sampling" = creating symbolic links in the directory
according to a repeat factor (no data copying, saving disk).
Links are absolute so that split lists resolve from any working directory.

Usage:
    python tools/build_mix.py --wdb_dir data/woundsdb_seg_ms --out data/mix_ms \
        --n_pub 700
"""
from __future__ import annotations

import argparse
import json
import os
import random


def link(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wdb_dir", default="data/woundsdb_seg_ms")
    ap.add_argument("--pub_dir", default="data/public_splits_cropped/train")
    ap.add_argument("--out", default="data/mix_ms")
    ap.add_argument("--n_pub", type=int, default=700,
                    help="number of samples randomly drawn from the public-domain training set (0=all)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = os.getcwd()
    wdb = os.path.join(root, args.wdb_dir)
    pub = os.path.join(root, args.pub_dir)
    out = os.path.join(root, args.out)
    val_out = out + "_val"

    for d in (out, val_out):
        for k in ("images", "masks"):
            os.makedirs(os.path.join(d, k), exist_ok=True)

    rng = random.Random(args.seed)

    # ---- public domain (guard against catastrophic forgetting)
    imgs = sorted(os.listdir(os.path.join(pub, "images")))
    rng.shuffle(imgs)
    if args.n_pub > 0:
        imgs = imgs[:args.n_pub]
    n_pub = 0
    for f in imgs:
        stem = os.path.splitext(f)[0]
        cand = [stem + e for e in (".png", ".jpg", ".jpeg")
                if os.path.exists(os.path.join(pub, "masks", stem + e))]
        if not cand:
            continue
        link(os.path.join(pub, "images", f), os.path.join(out, "images", "pub__" + f))
        link(os.path.join(pub, "masks", cand[0]), os.path.join(out, "masks", "pub__" + cand[0]))
        n_pub += 1

    # ---- WoundsDB domain (in-domain data, already multi-scale augmented via --roi_scale)
    meta = json.load(open(os.path.join(wdb, "split.json")))
    n_wdb = 0
    for s in meta["samples"]:
        if s["group"] != "train":
            continue
        stem = s["stem"]
        link(os.path.join(wdb, "images", stem + ".jpg"),
             os.path.join(out, "images", "wdb__" + stem + ".jpg"))
        link(os.path.join(wdb, "masks", stem + ".png"),
             os.path.join(out, "masks", "wdb__" + stem + ".png"))
        n_wdb += 1

    # ---- validation set: use only WoundsDB val (fixed scale) to measure in-domain generalization
    n_val = 0
    for s in meta["samples"]:
        if s["group"] != "val":
            continue
        stem = s["stem"]
        link(os.path.join(wdb, "images", stem + ".jpg"),
             os.path.join(val_out, "images", stem + ".jpg"))
        link(os.path.join(wdb, "masks", stem + ".png"),
             os.path.join(val_out, "masks", stem + ".png"))
        n_val += 1

    # broken-link self-check (ls cannot reveal broken links; os.path.exists is required)
    broken = 0
    for d in (out, val_out):
        for k in ("images", "masks"):
            p = os.path.join(d, k)
            for f in os.listdir(p):
                if not os.path.exists(os.path.join(p, f)):
                    broken += 1
                    print("  [!] broken link:", os.path.join(p, f))

    print(f"✅ {args.out}: public {n_pub} + woundsdb {n_wdb} = {n_pub + n_wdb} training images; "
          f"val {n_val}; broken links {broken}")
    return 0 if broken == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
