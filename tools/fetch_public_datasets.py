#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download and normalise public wound segmentation datasets.

Why a separate script:
    Public datasets have different directory structures (AZH/FUSeg/Medetec/WoundTissue) and are
    hosted in scattered places. They are normalised to the format expected by wg_train:

        data/public/<source>/images/*.png
        data/public/<source>/masks/*.png
        data/public/<source>/manifest.csv

Features:
    1. parallel_download — HTTP Range multi-threaded chunked download (a big speedup when a single
       connection is rate-limited)
    2. fetch_zip / fetch_github_dir — fetch from various hosting sources
    3. standardize — pair organisation + binarization + manifest
    4. audit — dataset checkup (count/resolution/mask fraction/empty masks/size mismatch/near-duplicates)

Usage:
    python tools/fetch_public_datasets.py --list
    python tools/fetch_public_datasets.py --source fuseg
    python tools/fetch_public_datasets.py --source all --audit
    python tools/fetch_public_datasets.py --source fuseg --conn 16 --keep_zip

Compliance: each dataset has a different license (see --list). Before downloading and using,
confirm for yourself that your intended use is permitted.
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

# ---------------------------------------------------------------- dataset registry
# The license field must be filled in truthfully; if it cannot be found, write None — do not guess.
SOURCES = {
    "fuseg2021": {
        "desc": "FUSeg / Foot Ulcer Segmentation Challenge 2021 (UWM + AZH), 1210 images / 1010 labels",
        "license": "CC BY-ND 4.0 (redistribution of derived datasets is prohibited, do not republish)",
        "size_mb": 322,
        "kind": "github_zip",
        "url": "https://codeload.github.com/uwm-bigdata/wound-segmentation/zip/refs/heads/master",
        "subdir": "data/Foot Ulcer Segmentation Challenge",
    },
    "medetec": {
        "desc": "Medetec diabetic foot ulcer 224px (included in the FUSeg repo), 160 pairs",
        "license": "CC BY-ND 4.0 (same repo, its terms apply)",
        "size_mb": 10,
        "kind": "github_zip",
        "url": "https://codeload.github.com/uwm-bigdata/wound-segmentation/zip/refs/heads/master",
        "subdir": "data/Medetec_foot_ulcer_224",
    },
    "azh": {
        "desc": "AZH Wound Care Center dataset patches (included in the FUSeg repo, 18MB zip)",
        "license": "CC BY-ND 4.0 (same repo)",
        "size_mb": 18,
        "kind": "github_zip_inner",
        "url": "https://codeload.github.com/uwm-bigdata/wound-segmentation/zip/refs/heads/master",
        "subdir": "data/wound_dataset/azh_wound_care_center_dataset_patches.zip",
    },
    "woundtissue": {
        "desc": "WoundTissue (arXiv 2502.10652), 6 tissue classes annotated -> binarized to wound vs background",
        "license": "see the repo LICENSE (confirm before use)",
        "size_mb": 1,
        "kind": "github_api_dir",
        "url": "https://api.github.com/repos/akabircs/WoundTissue/contents",
        "dirs": ["image", "label"],
    },
}

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ------------------------------------------------------------ multi-threaded chunked download
def _get(url, timeout=60):
    """Read a small file in full (only for <20MB API responses or sample images)."""
    import urllib.request

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "PhotoWoundArea/1.0")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _range_get(url, start, end, timeout=180):
    """Fetch bytes [start, end]; with retries, returns (start, bytes)."""
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
    raise RuntimeError(f"chunk download failed bytes={start}-{end}: {last}")


def _probe_size(url, timeout=60):
    """Probe the total size with Range: bytes=0-1.

    Why not HEAD: some CDNs (including GitHub codeload) return inconsistent
    results for HEAD, and on failure this degrades to reading several hundred
    MB into memory all at once.
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
    """Fallback when chunking is unsupported: stream to disk without full in-memory buffering."""
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
    """HTTP Range multi-threaded chunked download, each thread writes directly to the target file
    at its own offset.

    - First probe the total size with Range; on failure -> single-threaded streaming download
    - Enqueue every chunk one by one (not just the first conn chunks), to guarantee a full download
    - Supports resume: an already fully downloaded target file is skipped (decided by the caller)
    """
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    size = _probe_size(url)
    if size <= 0:
        print("   [i] Cannot probe file size / chunking unsupported, using single-threaded streaming download")
        n = _stream_download(url, dest)
        print(f"   ✓ download complete {dest} ({n/1e6:.1f} MB)")
        return n

    c = chunk_mb * 1024 * 1024
    starts = list(range(0, size, c))
    print(f"   size {size/1e6:.1f} MB | {len(starts)} chunks x {chunk_mb}MB | {conn} threads")

    with open(dest, "wb") as f:
        f.truncate(size)  # preallocate; fail immediately if disk is insufficient, rather than halfway through

    done = 0
    lock = threading.Lock()

    def worker(start):
        nonlocal done
        end = min(start + c - 1, size - 1)
        s, data = _range_get(url, start, end)
        # Each thread has its own fd and seeks to its own offset — no interference
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
    print(f"\n   ✓ download complete {dest} ({size/1e6:.1f} MB)")
    return size


# ------------------------------------------------------------------ fetch logic
def _ensure_repo_zip(conn=8, keep=False):
    """Download the uwm-bigdata/wound-segmentation repo zip and extract it to raw/.

    Note: codeload.github.com **does not support Range requests** (no Accept-Ranges,
    the zip is generated on the fly), so we hand this back to curl for a
    single-threaded streaming download instead of parallel_download.
    """
    url = SOURCES["fuseg2021"]["url"]
    zp = os.path.join(RAW_DIR, "wound-segmentation.zip")
    ex = os.path.join(RAW_DIR, "wound-segmentation")
    if os.path.isdir(ex):
        print("   [i] Already extracted, skipping download")
        return ex
    if not os.path.exists(zp) or os.path.getsize(zp) < 1_000_000:
        print(f"   downloading {url}\n   (~340MB, chunking unsupported, single-threaded takes about 10-20 minutes)")
        import subprocess
        r = subprocess.call(["curl", "-sSL", "--retry", "3", "-A",
                             "PhotoWoundArea/1.0", "-o", zp, url])
        if r != 0 or not os.path.exists(zp) or os.path.getsize(zp) < 1_000_000:
            raise RuntimeError(f"repo zip download failed (curl exit={r})")
        print(f"   ✓ zip {os.path.getsize(zp)/1e6:.1f} MB")
    print("   extracting …")
    with zipfile.ZipFile(zp) as z:
        z.extractall(RAW_DIR)
    inner = os.path.join(RAW_DIR, "wound-segmentation-master")
    if os.path.isdir(inner) and not os.path.isdir(ex):
        os.rename(inner, ex)
    if not keep:
        os.remove(zp)
    return ex


def _pairs_from_dirs(img_dir, lbl_dir):
    """Pair image/label by the same name (ignoring extension)."""
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
        bg_val = max(counts)[1]          # the colour with the largest share = background
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
    """Read image -> binarize mask -> save in a unified format. Returns True on success."""
    try:
        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path).convert("L")
    except Exception as e:
        print(f"   [!] failed to read {name}: {e}")
        return False
    if img.size != lbl.size:
        if strict:
            print(f"   [!] size mismatch, skipping {name}: {img.size} vs {lbl.size}")
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
            print(f"   {dst_src}/{split}: {n} pairs")
    return total, rows


def fetch_azh(conn=8, keep_zip=False):
    ex = _ensure_repo_zip(conn, keep_zip)
    inner = os.path.join(ex, "data", "wound_dataset",
                         "azh_wound_care_center_dataset_patches.zip")
    if not os.path.exists(inner):
        print("   [!] azh patches zip not found in repo, skipping")
        return 0, []
    work = os.path.join(RAW_DIR, "azh")
    os.makedirs(work, exist_ok=True)
    with zipfile.ZipFile(inner) as z:
        z.extractall(work)
    # Auto-find image/label pairs (common structures: */Images/*, */Labels/* or */image/*,*/mask/*)
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
        # find the label in the same or a sibling directory
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
    print(f"   azh: {n} pairs (from {len(img_dirs)} image directories)")
    return n, rows


def _get_file(url, dest, timeout=90, retries=3):
    """Download a single file; on failure clean up partial files so a corrupted file
    is not later mistaken for an existing one."""
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
            # clean up partial files before retrying
            for p in (dest, dest + ".tmp"):
                if os.path.exists(p):
                    os.remove(p)
    raise RuntimeError(f"download failed {url}: {last}")


def fetch_woundtissue(conn=8, keep_zip=False):
    """List directories via the GitHub API and download all samples
    (this repo only contains some sample images)."""
    import json

    rows = []
    listing = {}
    for d in ("image", "label"):
        data = json.loads(_get(
            f"https://api.github.com/repos/akabircs/WoundTissue/contents/{d}"))
        # Use the download_url returned by the API directly; escaping of spaces,
        # parentheses, etc. is handled naturally
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
        # Use the stem as the local filename, naturally avoiding image/label name collisions
        ip = os.path.join(raw_img, stem + ".png")
        lp = os.path.join(raw_lbl, stem + ".png")
        try:
            _get_file(iurl, ip)
            _get_file(listing["label"][stem], lp)
        except Exception as e:
            print(f"   [!] skipping {stem}: {e}")
            continue
        if _save_pair(ip, lp, "wt_" + stem + ".png",
                      img_dst, msk_dst, rows, "woundtissue", "all",
                      binarize=_binarize_colormap):
            n += 1
    print(f"   woundtissue: {n} pairs")
    return n, rows


FETCHERS = {
    "fuseg2021": fetch_fuseg,
    "medetec": fetch_fuseg,   # same repo, fetched together
    "azh": fetch_azh,
    "woundtissue": fetch_woundtissue,
}


# ------------------------------------------------------------------ dataset checkup
def audit(out_dir=OUT_DIR):
    print("\n══ Dataset checkup ══")
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
        # near-duplicate detection (image md5)
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
        print(f"  mask fraction: mean={fr.mean()*100:.2f}%  median={np.median(fr)*100:.2f}%  "
              f"min={fr.min()*100:.3f}%  max={fr.max()*100:.2f}%")
        print(f"  empty masks: {empty}   near-empty (<0.1%): {tiny}   exact duplicates: {dups}")
        print(f"  resolution varieties: {len(sizes)} -> {sorted(sizes)[:4]}")
        report[src] = {"n": len(rows), "empty": empty, "dup": dups,
                       "frac_mean": float(fr.mean()),
                       "frac_min": float(fr.min()), "sizes": len(sizes)}
        if empty:
            print(f"  [!] found {empty} empty-mask samples — recommend removing them before training")
    return report


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Download and standardize public wound datasets")
    ap.add_argument("--source", default="all",
                    help="fuseg2021 / medetec / azh / woundtissue / all")
    ap.add_argument("--conn", type=int, default=8, help="download concurrency")
    ap.add_argument("--keep_zip", action="store_true")
    ap.add_argument("--list", action="store_true", help="list available data sources")
    ap.add_argument("--audit", action="store_true", help="run a checkup on already-downloaded data")
    args = ap.parse_args()

    if args.list:
        print(f"{'source':<14}{'size':<10}{'License':<16}description")
        for k, v in SOURCES.items():
            print(f"{k:<14}{v['size_mb']:<10}{str(v['license'])[:14]:<16}{v['desc']}")
        return 0

    if args.audit:
        audit()
        return 0

    todo = list(SOURCES) if args.source == "all" else [args.source]
    # fuseg2021 and medetec live in the same repo zip, so trigger download/extraction only once
    if "fuseg2021" in todo and "medetec" in todo:
        todo.remove("medetec")

    flat_rows = []
    for name in todo:
        print(f"\n▶ {name}: {SOURCES[name]['desc']}")
        try:
            n, rows = FETCHERS[name](conn=args.conn, keep_zip=args.keep_zip)
        except Exception as e:
            print(f"   [✗] {name} fetch failed: {e}")
            continue
        flat_rows.extend(rows)
        print(f"   ✓ {name} completed {n} pairs")

    # Group by the source field in each row when writing manifests (one fetch may contain multiple sources)
    by_src = {}
    for r in flat_rows:
        by_src.setdefault(r["source"], []).append(r)
    for src, rows in by_src.items():
        sdir = os.path.join(OUT_DIR, src)
        with open(os.path.join(sdir, "manifest.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"   manifest: {sdir}/manifest.csv ({len(rows)} rows)")

    audit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
