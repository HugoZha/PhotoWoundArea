#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_woundsdb_seg.py — build WoundsDB segmentation crops from expert masks.

输入：data/raw/woundsdb/patients_partial/patients/case_X/day_Y/data/scene_N/{photo.jpg,mask.png}
输出：data/woundsdb_seg/{images,masks}/caseX__dayY__sceneZ.{jpg,png}
      + split.json（按 case 分组的 train/val/test，防患者级泄漏）

关键决策：
  - 降采样到长边 1600：4896px 原图训练时反正 resize 到 256，直接降可省 10 倍磁盘
    且掩膜边界足够（1600px 下伤口 ~40px）；
  - 裁黑边口径与 tools/crop_black_border.py 一致（连续纯黑行/列）；
  - 同 case 的所有 day/scene 归同一组 —— 这是患者级泄漏的天然防线；
  - 顺带做一次掩膜体检（前景占比、二值性），异常样本剔除并记录。
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
    """按 GT 包围盒裁出正方形 ROI —— 模拟「用户把镜头对准创面」的取景。

    口径说明（重要）：这里用了 GT 的**位置**信息，但模型仍需自行决定
    轮廓在哪里 —— 等价于两阶段实例分割里的「检测已给定、只做分割」，
    与端到端验证 e2e_woundsdb.py 的 roi 模式完全同口径。GT 不参与
    阈值 / 面积 / 后处理，因此不构成评测泄漏。

    返回 (l, t, r, b)，越界会向内平移；图像比 ROI 还小时返回 None。
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
    # 越界向内平移（不 pad 黑边 —— 黑边已被证实是模型的捷径特征）
    l = max(0, min(l, w - side))
    t = max(0, min(t, h - side))
    return l, t, l + side, t + side


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", action="store_true",
                    help="按 GT 包围盒裁出近拍 ROI（否则输出远拍全图）")
    ap.add_argument("--roi_scale", default="1.6,2.6",
                    help="训练集 ROI 尺度：逗号分隔的多个尺度时做多尺度扩增，"
                         "或两个值表示在该区间内随机采样")
    ap.add_argument("--val_scale", type=float, default=2.5,
                    help="val/test 组固定 ROI 尺度（与 e2e 推理取值一致，保证可比）")
    ap.add_argument("--roi_jitter", type=float, default=0.08,
                    help="ROI 中心随机偏移比例（模拟取景误差），0=严格居中")
    ap.add_argument("--roi_side", type=int, default=512,
                    help="ROI 统一输出边长")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="输出目录（默认 data/woundsdb_seg[_roi]）")
    args = ap.parse_args()
    raw = [float(x) for x in args.roi_scale.split(",")]
    if len(raw) > 2:
        # 多尺度扩增：每个训练样本产出 len(raw) 个不同取景尺度的副本
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
            bad.append({"stem": stem, "why": f"尺寸不符 {mask.shape} vs {photo.shape}"})
            continue
        t, b, l, r = black_borders(photo)
        h, w = photo.shape[:2]
        photo = photo[t:h - b, l:w - r]
        mask = mask[t:h - b, l:w - r]

        uniq = np.unique(mask)
        if not np.isin(uniq, (0, 255)).all():
            bad.append({"stem": stem, "why": f"非二值 uniq {uniq[:5]}"})
            continue
        # ROI 模式下前景占比要在**裁剪之后**才算 —— 远拍全图本来就只有 0.03%
        frac = float((mask > 127).mean())
        if not args.roi and not (1e-4 <= frac <= 0.5):
            bad.append({"stem": stem, "why": f"全图前景 {frac:.4f}"})
            continue

        # ---- 生成该样本的若干变体
        # 训练集：多尺度扩增（同一 GT 裁不同取景尺度，既扩量又增强取景鲁棒性）
        # val/test：固定 2.5 倍，与 e2e 推理取值一致，保证跨配置可比
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
                    bad.append({"stem": stem_out, "why": "空掩膜"})
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
                    bad.append({"stem": stem_out, "why": f"ROI 前景 {fr:.4f}"})
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
    print(f"✅ 写入 {len(rows)} 对（train {counts['train']} / val {counts['val']} "
          f"/ test {counts['test']}），剔除 {len(bad)} 对异常")
    if bad:
        for x in bad[:5]:
            print("   [!]", x)
    print(f"   -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
