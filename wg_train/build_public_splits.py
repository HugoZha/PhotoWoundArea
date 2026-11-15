#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_public_splits.py — 多源公开数据集 -> 可直接训练的 train/val/test 划分。

为什么要这一步（而不是随机划分）：
    公开伤口数据集里，同一个患者往往有**多次复诊/多个视角**的图（FUSeg 尤为明显）。
    如果随机划分，同一患者的图会同时出现在 train 和 test，模型"见过"这个伤口，
    指标会虚高 —— 文献里明确把这个叫 patient-level leakage。
    本脚本按 **group（≈患者/就诊）** 划分，保证同一 group 的所有图只进一个子集。

产出：
    <out>/train/{images,masks}     symlink 到原始文件（不额外占用磁盘）
    <out>/val/{images,masks}
    <out>/test/{images,masks}
    <out>/split_manifest.csv        filename, split, source, group, mask_frac
    <out>/split_report.json        划分统计与泄漏检查结论

用法：
    python -m wg_train.build_public_splits --root data/public \
        --sources fuseg2021,medetec --out data/public_splits
    python -m wg_train.build_public_splits --root data/public --extra_synth data/synth3d
    python -m wg_train.build_public_splits --infer_group     # 只看看 group 推断结果
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


# --------------------------------------------------------------- group 推断
_TAIL_NUM = re.compile(r"[_\-](?:rgb|img|image|color|left|right|crop|patch)?\d{1,4}$", re.I)


def infer_group(stem: str, source: str, regex: str | None = None) -> str:
    """从文件名推断 group id。

    优先级：
      1. --group_regex 的第一个捕获组
      2. 去掉源前缀 + 尾部序号后剩下的部分
      3. 兜底：整个 stem（意味着一图一组，划分退化成随机划分，脚本会告警）
    """
    if regex:
        m = re.search(regex, stem)
        if m and m.groups():
            return f"{source}::{m.group(1)}"
    s = stem
    # 去掉标准化阶段加的 "<source>_<split>_" / "wt_" 前缀。
    # Group keys must be cleaned; otherwise all images collapse into one group.
    #    （例如 "fuseg2021_train_0011" 若只剥 "fuseg2021_" 会剩下 "train"）。
    m = re.match(r"^(fuseg2021|medetec|azh|woundtissue)_(train|validation|val|test|all)_(.+)$", s)
    if m:
        source, s = m.group(1), m.group(3)
    elif s.startswith("wt_"):
        s = s[3:]
    # 反复剥离尾部序号
    while True:
        s2 = _TAIL_NUM.sub("", s)
        if s2 == s or not s2:
            break
        s = s2
    return f"{source}::{s or stem}"


# ------------------------------------------------------------------- 主流程
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
            print(f"  [!] 跳过 {src}：无 manifest ({mf})")
            continue
        with open(mf) as f:
            for r in csv.DictReader(f):
                r["_img"] = os.path.join(root, src, "images", r["file"])
                r["_msk"] = os.path.join(root, src, "masks", r["file"])
                rows.append(r)
        print(f"  {src}: {sum(1 for r in rows if r['source'] == src)} 行")
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
        print(f"  合成数据并入: {len(extra)} 行")
        rows += extra

    if not rows:
        print("❌ 没有任何可用样本。")
        return 1

    # ---- 去重：完全相同的图若同时落进 train 和 test，就是最直接的数据泄漏
    seen, dedup, n_dup = {}, [], 0
    for r in rows:
        key = _md5(r["_img"])
        if key in seen:
            n_dup += 1
            continue
        seen[key] = r
        dedup.append(r)
    if n_dup:
        print(f"  去重：移除 {n_dup} 张完全重复图")
    kept = dedup

    # 空掩膜过滤
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
        print(f"  过滤 {dropped} 个样本（空/过小掩膜或文件缺失，阈值 {args.min_mask_frac}）")

    for r in kept:
        r["group"] = infer_group(os.path.splitext(r["file"])[0], r["source"],
                                 args.group_regex)

    groups = sorted({r["group"] for r in kept})
    n_samp = len(kept)
    print(f"\n  样本 {n_samp} | 推断 group 数 {len(groups)} "
          f"(平均 {n_samp/max(len(groups),1):.1f} 图/组)")
    if len(groups) >= n_samp * 0.95:
        print("  [!] group 数≈样本数，可能未能识别患者；此时划分等价于随机划分，"
              "指标仍可能偏乐观。建议用 --group_regex 指定患者 ID 规则。")
    elif args.show_groups:
        print("  group 样例:", groups[:8])

    # 按 group 划分
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
        # 带源前缀，避免不同源同名冲突
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
            except OSError:      # 文件系统不支持符号链接时退化复制
                import shutil
                shutil.copy2(target, link)
        counts[sp] += 1
        out_rows.append({"file": uniq, "split": sp, "source": r["source"],
                         "group": r["group"], "mask_frac": r.get("mask_frac", "")})

    with open(os.path.join(out_root, "split_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "split", "source", "group", "mask_frac"])
        w.writeheader()
        w.writerows(out_rows)

    # ---- 泄漏检查：同一 group 不得跨 split
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

    print(f"\n  划分结果: train={counts['train']}  val={counts['val']}  test={counts['test']}")
    if leaks:
        print(f"  [✗] 检测到组泄漏: {leaks}")
    else:
        print("  ✓ 组泄漏检查通过：train/val/test 的 group 两两不相交")
    print(f"  产出: {out_root}/split_manifest.csv, split_report.json")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "public"))
    ap.add_argument("--sources", default="fuseg2021,medetec,azh,woundtissue")
    ap.add_argument("--extra_synth", default=None,
                    help="额外并入的合成数据集目录（含 images/masks/manifest.csv）")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "public_splits"))
    ap.add_argument("--ratios", default="0.7,0.15,0.15")
    ap.add_argument("--min_mask_frac", type=float, default=0.001,
                    help="掩膜占比下限，过滤空标注与近空样本")
    ap.add_argument("--group_regex", default=None,
                    help="从文件名提取患者/group ID 的正则（第一个捕获组）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show_groups", action="store_true")
    ap.add_argument("--infer_group", action="store_true",
                    help="只打印 group 推断效果，不写文件")
    args = ap.parse_args()

    args.sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    args.ratios = tuple(float(x) for x in args.ratios.split(","))
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        print("❌ ratios 之和必须为 1")
        return 1

    if args.infer_group:
        rows = load_rows(args.root, args.sources)
        for r in rows[:15]:
            print(f"  {r['file'][:44]:<46} -> {infer_group(os.path.splitext(r['file'])[0], r['source'], args.group_regex)}")
        gs = {infer_group(os.path.splitext(r['file'])[0], r['source'], args.group_regex)
              for r in rows}
        print(f"  {len(rows)} 样本 -> {len(gs)} group")
        return 0

    return build(args)


if __name__ == "__main__":
    sys.exit(main())
