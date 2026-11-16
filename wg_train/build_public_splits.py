#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_public_splits.py — multi-source public datasets -> ready-to-train train/val/test splits.

Why this step (instead of a random split):
    In public wound datasets, the same patient often has images from **multiple follow-up visits / multiple views** (especially true for FUSeg).
    With a random split, images of the same patient end up in both train and test, the model has "seen" that wound,
    and metrics are inflated — the literature explicitly calls this patient-level leakage.
    This script splits by **group (≈ patient/visit)**, ensuring all images of a group go into a single subset.

Output:
    <out>/train/{images,masks}     symlinks to the original files (no extra disk usage)
    <out>/val/{images,masks}
    <out>/test/{images,masks}
    <out>/split_manifest.csv        filename, split, source, group, mask_frac
    <out>/split_report.json        split statistics and leakage-check conclusions

Usage:
    python -m wg_train.build_public_splits --root data/public \
        --sources fuseg2021,medetec --out data/public_splits
    python -m wg_train.build_public_splits --root data/public --extra_synth data/synth3d
    python -m wg_train.build_public_splits --infer_group     # only inspect group-inference results
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------- group inference
_TAIL_NUM = re.compile(r"[_\-](?:rgb|img|image|color|left|right|crop|patch)?\d{1,4}$", re.I)


def infer_group(stem: str, source: str, regex: str | None = None) -> str:
    """Infer the group id from the filename.

    Priority:
      1. the first capture group of --group_regex
      2. what remains after stripping the source prefix + trailing number
      3. fallback: the whole stem (which means one group per image; the split degenerates into a random split and the script warns)
    """
    if regex:
        m = re.search(regex, stem)
        if m and m.groups():
            return f"{source}::{m.group(1)}"
    s = stem
    # Strip the "<source>_<split>_" / "wt_" prefixes added during normalization.
    # Group keys must be cleaned; otherwise all images collapse into one group.
    #    (e.g. "fuseg2021_train_0011" would leave "train" if only "fuseg2021_" were removed).
    m = re.match(r"^(fuseg2021|medetec|azh|woundtissue)_(train|validation|val|test|all)_(.+)$", s)
    if m:
        source, s = m.group(1), m.group(3)
    elif s.startswith("wt_"):
        s = s[3:]
    # Repeatedly strip trailing numbers
    while True:
        s2 = _TAIL_NUM.sub("", s)
        if s2 == s or not s2:
            break
        s = s2
    return f"{source}::{s or stem}"


# ------------------------------------------------------------------- main flow
def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(root: str, sources: list[str]):
    rows = []
    for src in sources:
        mf = os.path.join(root, src, "manifest.csv")
        if not os.path.exists(mf):
            print(f"  [!] Skipping {src}: no manifest ({mf})")
            continue
        with open(mf) as f:
            for r in csv.DictReader(f):
                r["_img"] = os.path.join(root, src, "images", r["file"])
                r["_msk"] = os.path.join(root, src, "masks", r["file"])
                rows.append(r)
        print(f"  {src}: {sum(1 for r in rows if r['source'] == src)} rows")
    return rows


def build(args):
    random.seed(args.seed)
    rows = load_rows(args.root, args.sources)
    if args.extra_synth and os.path.isdir(args.extra_synth):
        extra = load_rows(os.path.dirname(args.extra_synth),
                          [os.path.basename(args.extra_synth)])
        for r in extra:
            r["_img"] = os.path.join(args.extra_synth, "images", r["file"])
            r["_msk"] = os.path.join(args.extra_synth, "masks", r["file"])
        print(f"  Synthetic data merged in: {len(extra)} rows")
        rows += extra

    if not rows:
        print("❌ No usable samples found.")
        return 1

    # ---- Deduplication: identical images landing in both train and test is the most direct form of data leakage
    seen, dedup, n_dup = {}, [], 0
    for r in rows:
        key = _md5(r["_img"])
        if key in seen:
            n_dup += 1
            continue
        seen[key] = r
        dedup.append(r)
    if n_dup:
        print(f"  Deduplication: removed {n_dup} exact duplicate images")
    kept = dedup

    # Empty-mask filtering
    dropped = 0
    kept2 = []
    for r in kept:
        try:
            if float(r.get("mask_frac", 1)) < args.min_mask_frac:
                dropped += 1
                continue
        except ValueError:
            pass
        if not (os.path.exists(r["_img"]) and os.path.exists(r["_msk"])):
            dropped += 1
            continue
        kept2.append(r)
    kept = kept2
    if dropped:
        print(f"  Filtered out {dropped} samples (empty/too-small mask or missing files, threshold {args.min_mask_frac})")

    for r in kept:
        r["group"] = infer_group(os.path.splitext(r["file"])[0], r["source"],
                                 args.group_regex)

    groups = sorted({r["group"] for r in kept})
    n_samp = len(kept)
    print(f"\n  samples {n_samp} | inferred groups {len(groups)} "
          f"(avg. {n_samp/max(len(groups),1):.1f} images/group)")
    if len(groups) >= n_samp * 0.95:
        print("  [!] group count ≈ sample count; patient identity may not have been recognized; in that case the split is equivalent to a random split, "
              "and metrics may still be optimistic. Consider using --group_regex to specify a patient-ID rule.")
    elif args.show_groups:
        print("  sample groups:", groups[:8])

    # Split by group
    rnd = random.Random(args.seed)
    gshuffled = sorted(groups)
    rnd.shuffle(gshuffled)
    tr, va, te = args.ratios
    n_g = len(gshuffled)
    n_train = max(1, int(n_g * tr))
    n_val = max(1, int(n_g * va))
    assign = {}
    for g in gshuffled[:n_train]:
        assign[g] = "train"
    for g in gshuffled[n_train:n_train + n_val]:
        assign[g] = "val"
    for g in gshuffled[n_train + n_val:]:
        assign[g] = "test"

    out_root = args.out
    out_rows = []
    counts = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        for sub in ("images", "masks"):
            os.makedirs(os.path.join(out_root, split, sub), exist_ok=True)

    for r in kept:
        sp = assign[r["group"]]
        stem = os.path.splitext(r["file"])[0]
        # Prefix with the source to avoid same-name collisions across sources
        uniq = f"{r['source']}__{stem}.png"
        li = os.path.join(out_root, sp, "images", uniq)
        lm = os.path.join(out_root, sp, "masks", uniq)
        # Symlink targets must be absolute: relative targets resolve against
        #    the link's own directory, not the current working directory.
        for link, target in ((li, r["_img"]), (lm, r["_msk"])):
            if os.path.lexists(link):
                os.remove(link)
            target = os.path.abspath(target)
            try:
                os.symlink(target, link)
            except OSError:      # fall back to copying when the filesystem does not support symlinks
                import shutil
                shutil.copy2(target, link)
        counts[sp] += 1
        out_rows.append({"file": uniq, "split": sp, "source": r["source"],
                         "group": r["group"], "mask_frac": r.get("mask_frac", "")})

    with open(os.path.join(out_root, "split_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "split", "source", "group", "mask_frac"])
        w.writeheader()
        w.writerows(out_rows)

    # ---- Leakage check: the same group must not span multiple splits
    by_split = {}
    for r in out_rows:
        by_split.setdefault(r["split"], set()).add(r["group"])
    leaks = []
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                inter = by_split.get(a, set()) & by_split.get(b, set())
                if inter:
                    leaks.append((a, b, len(inter)))

    report = {
        "n_samples": len(out_rows),
        "n_groups": len(groups),
        "counts": counts,
        "ratios": args.ratios,
        "seed": args.seed,
        "min_mask_frac": args.min_mask_frac,
        "group_leakage": leaks,
        "sources": args.sources,
        "extra_synth": args.extra_synth,
    }
    with open(os.path.join(out_root, "split_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Split result: train={counts['train']}  val={counts['val']}  test={counts['test']}")
    if leaks:
        print(f"  [✗] Group leakage detected: {leaks}")
    else:
        print("  ✓ Group leakage check passed: train/val/test groups are pairwise disjoint")
    print(f"  Output: {out_root}/split_manifest.csv, split_report.json")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "public"))
    ap.add_argument("--sources", default="fuseg2021,medetec,azh,woundtissue")
    ap.add_argument("--extra_synth", default=None,
                    help="Directory of an additional synthetic dataset to merge in (contains images/masks/manifest.csv)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "public_splits"))
    ap.add_argument("--ratios", default="0.7,0.15,0.15")
    ap.add_argument("--min_mask_frac", type=float, default=0.001,
                    help="Lower bound on mask fraction; filters out empty and near-empty annotations")
    ap.add_argument("--group_regex", default=None,
                    help="Regex to extract the patient/group ID from filenames (first capture group)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show_groups", action="store_true")
    ap.add_argument("--infer_group", action="store_true",
                    help="Only print group-inference results; do not write files")
    args = ap.parse_args()

    args.sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    args.ratios = tuple(float(x) for x in args.ratios.split(","))
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        print("❌ ratios must sum to 1")
        return 1

    if args.infer_group:
        rows = load_rows(args.root, args.sources)
        for r in rows[:15]:
            print(f"  {r['file'][:44]:<46} -> {infer_group(os.path.splitext(r['file'])[0], r['source'], args.group_regex)}")
        gs = {infer_group(os.path.splitext(r['file'])[0], r['source'], args.group_regex)
              for r in rows}
        print(f"  {len(rows)} samples -> {len(gs)} groups")
        return 0

    return build(args)


if __name__ == "__main__":
    sys.exit(main())
