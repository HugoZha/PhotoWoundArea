#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_mix.py — build the mixed public/WoundsDB training set with domain weighting.

Rationale:
  WoundsDB 只有 188 张专家掩膜，且远拍全图里伤口仅占 0.03%~0.3% 像素。
  直接整图训练学不到东西（val_dice≈0）。因此需要：
    1) 按 GT 包围盒裁 ROI（build_woundsdb_seg.py --roi），把占比提到 ~6%；
    2) 多尺度扩增（同一 GT 裁多个取景尺度）把训练对从 112 扩到 558；
    3) 域加权：控制 public : woundsdb 的采样比例（w_repeat / n_pub）。

实现上「加权采样」= 目录里按重复系数建符号链接（不复制数据，省磁盘）。
Links are absolute so that split lists resolve from any working directory.

用法：
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
                    help="从公开域训练集随机抽取的样本数（0=全部）")
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

    # ---- 公开域（防灾难性遗忘）
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

    # ---- WoundsDB 域（域内数据，按 --roi_scale 已做多尺度扩增）
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

    # ---- 验证集：只用 WoundsDB val（固定尺度），衡量域内泛化
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

    # 断链自检（ls 看不出断链，必须 os.path.exists）
    broken = 0
    for d in (out, val_out):
        for k in ("images", "masks"):
            p = os.path.join(d, k)
            for f in os.listdir(p):
                if not os.path.exists(os.path.join(p, f)):
                    broken += 1
                    print("  [!] 断链:", os.path.join(p, f))

    print(f"✅ {args.out}: public {n_pub} + woundsdb {n_wdb} = {n_pub + n_wdb} 训练图；"
          f"val {n_val}；断链 {broken}")
    return 0 if broken == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
