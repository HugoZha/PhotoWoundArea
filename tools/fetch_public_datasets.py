#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download and normalise public wound segmentation datasets.

为什么单独写这个脚本：
    公开数据集的目录结构各不相同（AZH/FUSeg/Medetec/WoundTissue），且托管分散。
    They are normalised to the format expected by wg_train:

        data/public/<source>/images/*.png
        data/public/<source>/masks/*.png
        data/public/<source>/manifest.csv

能力：
    1. parallel_download — HTTP Range 多线程分片下载（单连接被限速时显著加速）
    2. fetch_zip / fetch_github_dir — 拉取各类托管源
    3. standardize — 配对整理 + 二值化 + manifest
    4. audit — 数据体检（数量/分辨率/掩膜占比/空掩膜/尺寸不一致/近似重复）

用法：
    python tools/fetch_public_datasets.py --list
    python tools/fetch_public_datasets.py --source fuseg
    python tools/fetch_public_datasets.py --source all --audit
    python tools/fetch_public_datasets.py --source fuseg --conn 16 --keep_zip

合规：各数据集 License 不同（见 --list），下载使用前请自行确认用途是否被允许。
"""

import argparse
import csv
import hashlib
import os
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "data", "public")

# ---------------------------------------------------------------- 数据集登记
# license 一栏务必如实标注；获取不到的写 None，不要猜测。
SOURCES = {
    "fuseg2021": {
        "desc": "FUSeg / Foot Ulcer Segmentation Challenge 2021 (UWM + AZH)，1210 图 / 1010 标签",
        "license": "CC BY-ND 4.0（禁止再分发衍生数据集，请勿二次公开）",
        "size_mb": 322,
        "kind": "github_zip",
        "url": "https://codeload.github.com/uwm-bigdata/wound-segmentation/zip/refs/heads/master",
        "subdir": "data/Foot Ulcer Segmentation Challenge",
    },
    "medetec": {
        "desc": "Medetec diabetic foot ulcer 224px（含于 FUSeg 仓库），160 对",
        "license": "CC BY-ND 4.0（同仓库，需遵循其条款）",
        "size_mb": 10,
        "kind": "github_zip",
        "url": "https://codeload.github.com/uwm-bigdata/wound-segmentation/zip/refs/heads/master",
        "subdir": "data/Medetec_foot_ulcer_224",
    },
    "azh": {
        "desc": "AZH Wound Care Center dataset patches（含于 FUSeg 仓库，18MB zip）",
        "license": "CC BY-ND 4.0（同仓库）",
        "size_mb": 18,
        "kind": "github_zip_inner",
        "url": "https://codeload.github.com/uwm-bigdata/wound-segmentation/zip/refs/heads/master",
        "subdir": "data/wound_dataset/azh_wound_care_center_dataset_patches.zip",
    },
    "woundtissue": {
        "desc": "WoundTissue (arXiv 2502.10652)，6 类组织标注 -> 二值化为伤口 vs 背景",
        "license": "见仓库 LICENSE（使用前请确认）",
        "size_mb": 1,
        "kind": "github_api_dir",
        "url": "https://api.github.com/repos/akabircs/WoundTissue/contents",
        "dirs": ["image", "label"],
    },
}

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ------------------------------------------------------------ 多线程分片下载
def _get(url, timeout=60):
    """完整读取一个小文件（仅用于 <20MB 的 API 响应或示例图）。"""
    import urllib.request

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "PhotoWoundArea/1.0")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _range_get(url, start, end, timeout=180):
    """取 [start, end] 字节；带重试，返回 (start, bytes)。"""
    import urllib.request

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "PhotoWoundArea/1.0")
    req.add_header("Range", f"bytes={start}-{end}")
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return start, r.read()
        except Exception as e:
            last = e
    raise RuntimeError(f"分片下载失败 bytes={start}-{end}: {last}")


def _probe_size(url, timeout=60):
    """用 Range: bytes=0-1 探测总大小。

    为什么不 HEAD：部分 CDN（含 GitHub codeload）对 HEAD 返回不一致的结果，
    一旦失败就会退化成把几百 MB 一次性读进内存。
    """
    import urllib.request

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "PhotoWoundArea/1.0")
    req.add_header("Range", "bytes=0-1")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            cr = r.headers.get("Content-Range") or ""
            if "/" in cr:                       # bytes 0-1/338172456
                return int(cr.rsplit("/", 1)[-1])
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else 0
    except Exception:
        return 0


def _stream_download(url, dest, timeout=180):
    """不支持分片时的兜底：流式写盘，不做全量内存缓冲。"""
    import urllib.request

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "PhotoWoundArea/1.0")
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            n += len(buf)
    return n


def parallel_download(url, dest, conn=8, chunk_mb=8):
    """HTTP Range 多线程分片下载，每线程按各自 offset 直接写入目标文件。

    - 先用 Range 探测总大小；探测失败 -> 单线程流式下载
    - 所有分片逐一入队（而非只取前 conn 片），保证完整下载
    - 支持断点：已下满的目标文件会跳过重投（由调用方判断）
    """
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    size = _probe_size(url)
    if size <= 0:
        print("   [i] 无法探测文件大小/不支持分片，单线程流式下载")
        n = _stream_download(url, dest)
        print(f"   ✓ 下载完成 {dest} ({n/1e6:.1f} MB)")
        return n

    c = chunk_mb * 1024 * 1024
    starts = list(range(0, size, c))
    print(f"   size {size/1e6:.1f} MB | {len(starts)} 片 x {chunk_mb}MB | {conn} 线程")

    with open(dest, "wb") as f:
        f.truncate(size)  # 预分配；磁盘不足时会立刻失败而不是下到一半

    done = 0
    lock = threading.Lock()

    def worker(start):
        nonlocal done
        end = min(start + c - 1, size - 1)
        s, data = _range_get(url, start, end)
        # 每个线程独立 fd，各自 seek 到自己的 offset —— 互不干扰
        with open(dest, "r+b") as fw:
            fw.seek(s)
            fw.write(data)
        with lock:
            done += len(data)
            if len(starts) > 8:
                pct = done / size * 100
                nbar = int(pct / 5)
                print(f"\r      [{'#'*nbar}{'.'*(20-nbar)}] {pct:5.1f}%",
                      end="", flush=True)
        return len(data)

    with ThreadPoolExecutor(max_workers=conn) as ex:
        list(ex.map(worker, starts))
    print(f"\n   ✓ 下载完成 {dest} ({size/1e6:.1f} MB)")
    return size


# ------------------------------------------------------------------ 拉取逻辑
def _ensure_repo_zip(conn=8, keep=False):
    """下载 uwm-bigdata/wound-segmentation 仓库 zip 并解压到 raw/。

    注意：codeload.github.com **不支持 Range 请求**（无 Accept-Ranges，zip 为流式生成），
    因此这里交回 curl 做单线程流式下载，而不是 parallel_download。
    """
    url = SOURCES["fuseg2021"]["url"]
    zp = os.path.join(RAW_DIR, "wound-segmentation.zip")
    ex = os.path.join(RAW_DIR, "wound-segmentation")
    if os.path.isdir(ex):
        print("   [i] 已解压，跳过下载")
        return ex
    if not os.path.exists(zp) or os.path.getsize(zp) < 1_000_000:
        print(f"   下载 {url}\n   （约 340MB，不支持分片，单线程约需 10-20 分钟）")
        import subprocess
        r = subprocess.call(["curl", "-sSL", "--retry", "3", "-A",
                             "PhotoWoundArea/1.0", "-o", zp, url])
        if r != 0 or not os.path.exists(zp) or os.path.getsize(zp) < 1_000_000:
            raise RuntimeError(f"仓库 zip 下载失败（curl exit={r}）")
        print(f"   ✓ zip {os.path.getsize(zp)/1e6:.1f} MB")
    print("   解压中 …")
    with zipfile.ZipFile(zp) as z:
        z.extractall(RAW_DIR)
    inner = os.path.join(RAW_DIR, "wound-segmentation-master")
    if os.path.isdir(inner) and not os.path.isdir(ex):
        os.rename(inner, ex)
    if not keep:
        os.remove(zp)
    return ex


def _pairs_from_dirs(img_dir, lbl_dir):
    """按同名（忽略扩展名）配对 image/label。"""
    if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
        return []
    labels = {}
    for n in os.listdir(lbl_dir):
        if n.lower().endswith(IMG_EXT):
            labels[os.path.splitext(n)[0]] = os.path.join(lbl_dir, n)
    out = []
    for n in sorted(os.listdir(img_dir)):
        if not n.lower().endswith(IMG_EXT):
            continue
        stem = os.path.splitext(n)[0]
        if stem in labels:
            out.append((os.path.join(img_dir, n), labels[stem], n))
    return out


def _binarize_threshold(lbl):
    """Binarise labels whose foreground encoding differs between releases.

    FUSeg uses black (0,0,0) background with white (255,255,255) foreground;
    Medetec uses black background with (1,1,1) foreground, which is almost
    black. A plain >127 threshold silently empties all 160 Medetec masks, so
    for discrete labels (at most 8 distinct values) the most frequent value
    is taken as background and everything else as foreground. Continuous or
    greyscale labels keep the conventional >127 threshold.
    """
    g = np.array(lbl.convert("L"))
    vals = np.unique(g)
    if len(vals) <= 8:
        counts = [(int((g == v).sum()), int(v)) for v in vals]
        bg_val = max(counts)[1]          # 占比最大的颜色 = 背景
        return (g != bg_val).astype(np.uint8) * 255
    return (g > 127).astype(np.uint8) * 255


def _binarize_colormap(lbl, bg_colors=((0, 0, 255), (0, 0, 0)), tol=60.0):
    """Handle colour-coded label maps (for example the wound-tissue atlas):
        Granulation=Red, Slough=Yellow, Maceration=White, Necrosis=Black,
        Tendon=Sky blue, Bone=Cream, Background=Blue

    Two conventions matter:
      1. Converting to greyscale and thresholding at 127 fails because red
         (255,0,0) becomes only 76 and is taken as background, emptying the mask.
      2. Background is not a single colour: besides the blue (0,0,255),
         unlabelled black (0,0,0) canvas areas also occur. Excluding only
         blue labels those areas as wound (mask coverage reaches 100%). The
         rule therefore uses the distance to any background colour.
    """
    arr = np.asarray(lbl.convert("RGB"), dtype=np.float32)
    dists = np.stack([np.linalg.norm(arr - np.array(c, np.float32), axis=2)
                      for c in bg_colors], axis=0)
    return (dists.min(axis=0) > tol).astype(np.uint8) * 255


def _save_pair(img_path, lbl_path, name, img_dst, msk_dst, rows, source, split,
               strict=True, binarize=None):
    """读图 -> 掩膜二值化 -> 保存统一格式。返回 True 表示成功。"""
    try:
        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path).convert("L")
    except Exception as e:
        print(f"   [!] 读取失败 {name}: {e}")
        return False
    if img.size != lbl.size:
        if strict:
            print(f"   [!] 尺寸不一致，跳过 {name}: {img.size} vs {lbl.size}")
            return False
        lbl = lbl.resize(img.size, Image.NEAREST)
    arr = np.array(lbl)
    binarize = binarize or _binarize_threshold
    m01 = binarize(lbl)
    stem = os.path.splitext(name)[0]
    out_img = os.path.join(img_dst, f"{stem}.png")
    out_msk = os.path.join(msk_dst, f"{stem}.png")
    img.save(out_img)
    Image.fromarray(m01).save(out_msk)
    rows.append({
        "file": f"{stem}.png",
        "source": source,
        "split": split,
        "w_px": img.size[0],
        "h_px": img.size[1],
        "mask_frac": round(float((m01 > 0).mean()), 5),
    })
    return True


def fetch_fuseg(conn=8, keep_zip=False):
    ex = _ensure_repo_zip(conn, keep_zip)
    rows = []
    total = 0
    for src_name, sub in (("fuseg2021", SOURCES["fuseg2021"]["subdir"]),
                          ("medetec", SOURCES["medetec"]["subdir"])):
        for split in ("train", "validation", "val", "test"):
            idir = os.path.join(ex, sub, split, "images")
            ldir = os.path.join(ex, sub, split, "labels")
            if not (os.path.isdir(idir) and os.path.isdir(ldir)):
                continue
            pairs = _pairs_from_dirs(idir, ldir)
            if not pairs:
                continue
            dst_src = "fuseg2021" if src_name == "fuseg2021" else "medetec"
            img_dst = os.path.join(OUT_DIR, dst_src, "images")
            msk_dst = os.path.join(OUT_DIR, dst_src, "masks")
            os.makedirs(img_dst, exist_ok=True)
            os.makedirs(msk_dst, exist_ok=True)
            n = 0
            for ip, lp, nm in pairs:
                prefix = f"{dst_src}_{split}_"
                safe = prefix + nm
                if _save_pair(ip, lp, safe, img_dst, msk_dst, rows,
                              dst_src, split):
                    n += 1
            total += n
            print(f"   {dst_src}/{split}: {n} 对")
    return total, rows


def fetch_azh(conn=8, keep_zip=False):
    ex = _ensure_repo_zip(conn, keep_zip)
    inner = os.path.join(ex, "data", "wound_dataset",
                         "azh_wound_care_center_dataset_patches.zip")
    if not os.path.exists(inner):
        print("   [!] 仓库内未找到 azh patches zip，跳过")
        return 0, []
    work = os.path.join(RAW_DIR, "azh")
    os.makedirs(work, exist_ok=True)
    with zipfile.ZipFile(inner) as z:
        z.extractall(work)
    # 自动找 images/labels 配对（常见结构：*/Images/*, */Labels/* 或 */image/*,*/mask/*）
    img_dirs, lbl_dirs = [], []
    for root, _, files in os.walk(work):
        low = root.lower()
        if files and any(f.lower().endswith(IMG_EXT) for f in files):
            if "label" in low or "mask" in low or "ground" in low or "gt" in low:
                lbl_dirs.append(root)
            elif "imag" in low or "img" in low:
                img_dirs.append(root)
    rows = []
    img_dst = os.path.join(OUT_DIR, "azh", "images")
    msk_dst = os.path.join(OUT_DIR, "azh", "masks")
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(msk_dst, exist_ok=True)
    n = 0
    for idir in img_dirs:
        # 同级/兄弟目录里找 label
        cand = None
        for ldir in lbl_dirs:
            if os.path.commonpath([idir, ldir]) and _pairs_from_dirs(idir, ldir):
                cand = ldir
                break
        if cand is None:
            continue
        for ip, lp, nm in _pairs_from_dirs(idir, cand):
            safe = "azh_" + nm
            if _save_pair(ip, lp, safe, img_dst, msk_dst, rows, "azh", "all"):
                n += 1
    print(f"   azh: {n} 对（来自 {len(img_dirs)} 个图像目录）")
    return n, rows


def _get_file(url, dest, timeout=90, retries=3):
    """下载单文件；失败时清理半成品，避免留下损坏文件被后续当成已存在。"""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    last = None
    for _ in range(retries):
        try:
            data = _get(url, timeout=timeout)
            tmp = dest + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            return dest
        except Exception as e:
            last = e
            # 清理半成品再重试
            for p in (dest, dest + ".tmp"):
                if os.path.exists(p):
                    os.remove(p)
    raise RuntimeError(f"下载失败 {url}: {last}")


def fetch_woundtissue(conn=8, keep_zip=False):
    """GitHub API 列举目录并下载所有样本（该仓库仅含部分示例图）。"""
    import json

    rows = []
    listing = {}
    for d in ("image", "label"):
        data = json.loads(_get(
            f"https://api.github.com/repos/akabircs/WoundTissue/contents/{d}"))
        # 直接用 API 返回的 download_url，天然处理了空格/括号等转义
        listing[d] = {os.path.splitext(i["name"])[0]: i["download_url"]
                      for i in data}
    img_dst = os.path.join(OUT_DIR, "woundtissue", "images")
    msk_dst = os.path.join(OUT_DIR, "woundtissue", "masks")
    raw_img = os.path.join(RAW_DIR, "woundtissue_tmp", "image")
    raw_lbl = os.path.join(RAW_DIR, "woundtissue_tmp", "label")
    for d in (raw_img, raw_lbl, img_dst, msk_dst):
        os.makedirs(d, exist_ok=True)
    n = 0
    for stem, iurl in listing["image"].items():
        if stem not in listing["label"]:
            continue
        # 用 stem 做本地文件名，天然避开 image/label 同名冲突
        ip = os.path.join(raw_img, stem + ".png")
        lp = os.path.join(raw_lbl, stem + ".png")
        try:
            _get_file(iurl, ip)
            _get_file(listing["label"][stem], lp)
        except Exception as e:
            print(f"   [!] 跳过 {stem}: {e}")
            continue
        if _save_pair(ip, lp, "wt_" + stem + ".png",
                      img_dst, msk_dst, rows, "woundtissue", "all",
                      binarize=_binarize_colormap):
            n += 1
    print(f"   woundtissue: {n} 对")
    return n, rows


FETCHERS = {
    "fuseg2021": fetch_fuseg,
    "medetec": fetch_fuseg,   # 同一个仓库，一起取
    "azh": fetch_azh,
    "woundtissue": fetch_woundtissue,
}


# ------------------------------------------------------------------ 数据体检
def audit(out_dir=OUT_DIR):
    print("\n══ 数据体检 ══")
    report = {}
    for src in sorted(os.listdir(out_dir)):
        sdir = os.path.join(out_dir, src)
        mfile = os.path.join(sdir, "manifest.csv")
        if not os.path.exists(mfile):
            continue
        with open(mfile) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        fr = np.array([float(r["mask_frac"]) for r in rows])
        sizes = {(int(r["w_px"]), int(r["h_px"])) for r in rows}
        # 近似重复检测（图像 md5）
        dups = 0
        seen = set()
        idir = os.path.join(sdir, "images")
        for nm in os.listdir(idir):
            if not nm.endswith(".png"):
                continue
            h = hashlib.md5(open(os.path.join(idir, nm), "rb").read()).hexdigest()
            if h in seen:
                dups += 1
            seen.add(h)
        empty = int((fr == 0).sum())
        tiny = int((fr < 0.001).sum())
        print(f"\n[{src}]  n={len(rows)}")
        print(f"  掩膜占比: mean={fr.mean()*100:.2f}%  中位={np.median(fr)*100:.2f}%  "
              f"min={fr.min()*100:.3f}%  max={fr.max()*100:.2f}%")
        print(f"  空掩膜: {empty}   近乎空(<0.1%): {tiny}   完全重复图: {dups}")
        print(f"  分辨率种类: {len(sizes)} -> {sorted(sizes)[:4]}")
        report[src] = {"n": len(rows), "empty": empty, "dup": dups,
                       "frac_mean": float(fr.mean()),
                       "frac_min": float(fr.min()), "sizes": len(sizes)}
        if empty:
            print(f"  [!] 发现 {empty} 张空掩膜样本 —— 训练前建议剔除")
    return report


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="下载并标准化公开伤口数据集")
    ap.add_argument("--source", default="all",
                    help="fuseg2021 / medetec / azh / woundtissue / all")
    ap.add_argument("--conn", type=int, default=8, help="下载并发数")
    ap.add_argument("--keep_zip", action="store_true")
    ap.add_argument("--list", action="store_true", help="列出可用数据源")
    ap.add_argument("--audit", action="store_true", help="对已下载数据做体检")
    args = ap.parse_args()

    if args.list:
        print(f"{'源':<14}{'体量':<10}{'License':<16}说明")
        for k, v in SOURCES.items():
            print(f"{k:<14}{v['size_mb']:<10}{str(v['license'])[:14]:<16}{v['desc']}")
        return 0

    if args.audit:
        audit()
        return 0

    todo = list(SOURCES) if args.source == "all" else [args.source]
    # fuseg2021 与 medetec 同在一个仓库 zip 中，只触发一次下载解压
    if "fuseg2021" in todo and "medetec" in todo:
        todo.remove("medetec")

    flat_rows = []
    for name in todo:
        print(f"\n▶ {name}: {SOURCES[name]['desc']}")
        try:
            n, rows = FETCHERS[name](conn=args.conn, keep_zip=args.keep_zip)
        except Exception as e:
            print(f"   [✗] {name} 获取失败: {e}")
            continue
        flat_rows.extend(rows)
        print(f"   ✓ {name} 完成 {n} 对")

    # 按 row 里的 source 字段分组写 manifest（一次拉取可能包含多个源）
    by_src = {}
    for r in flat_rows:
        by_src.setdefault(r["source"], []).append(r)
    for src, rows in by_src.items():
        sdir = os.path.join(OUT_DIR, src)
        with open(os.path.join(sdir, "manifest.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"   manifest: {sdir}/manifest.csv ({len(rows)} 行)")

    audit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
