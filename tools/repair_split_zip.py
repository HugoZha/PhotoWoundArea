#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repair_split_zip.py — rescue files from a split zip that is **missing its last volume**.

Why it is needed:
    The WoundsDB Patients archive was split by the downloader into
    patients.zip + patients.z01…z05. When the last volume is missing (as is
    currently the case — only up to z04 remains), the whole zip's **central
    directory is lost**, and `unzip` reports a bunch of "bad zipfile offset"
    errors and refuses to extract.

    But zip has a property that can save the day: every member is preceded by
    its own **local file header** (`PK\\x03\\x04`), which contains the filename,
    compression method, and compressed size. The central directory is only an
    "index"; losing the index does not mean losing the data — by scanning
    **member by member** along the data stream, the files that are still intact
    can be rescued.

Usage:
    # 1) First see what can be rescued (scan the index only, no extraction)
    python tools/repair_split_zip.py --list

    # 2) Rescue only the modalities you need (default mask + photo)
    python tools/repair_split_zip.py --extract --filter mask,photo --out data/raw/woundsdb/patients_partial

    # 3) Scan only the first N members / dry run; confirm it is correct before the full run
    python tools/repair_split_zip.py --extract --limit 200 --dry-run

⚠️ Limitation: members that span the missing volume will fail to decompress and be
   skipped (a few files near the end cannot be rescued). To get the complete
   data, you should still use tools/fetch_woundsdb.py to fetch the missing last
   piece and then extract normally.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(ROOT, "data", "raw", "woundsdb")
SPLIT_PARTS = ["patients.zip"] + [f"patients.z{i:02d}" for i in range(1, 9)]

LH_SIG = b"PK\x03\x04"
CD_SIG = b"PK\x01\x02"
DD_SIG = b"PK\x07\x08"


def autodetect_parts(paths: list[str]) -> list[tuple[str, int]]:
    """Auto-detect the volume order and whether to skip the spanning marker;
    returns [(path, bytes skipped)].

    Info-ZIP split-archive conventions:
      · Only the **first volume** has a 4-byte spanning marker `PK\\x07\\x08` at
        its start; the other volumes are pure data.
      · Only the **last volume** has the EOCD `PK\\x05\\x06` at its tail (that
        is, the central directory of the whole archive).
      · The filename of the last volume is **not necessarily** last (in this
        case it is patients.zip, while z01 is the first volume).
    So you cannot sort by filename; you must locate them by these two markers.
    """
    exist = [p for p in paths if os.path.exists(p)]
    if not exist:
        return []
    first = last = None
    for p in exist:
        with open(p, "rb") as f:
            head = f.read(4)
        if head in (b"PK\x07\x08", b"PK\x03\x04") and first is None:
            first = p
    for p in exist:
        sz = os.path.getsize(p)
        with open(p, "rb") as f:
            f.seek(max(0, sz - 70000))
            tail = f.read()
        if b"PK\x05\x06" in tail and last is None:
            last = p
    rest = sorted(p for p in exist if p not in (first, last))
    order = ([first] if first else []) + rest + ([last] if last else [])
    out = []
    for p in order:
        with open(p, "rb") as f:
            head = f.read(4)
        out.append((p, 4 if head == b"PK\x07\x08" else 0))
    return out


class MultiPartStream:
    """Concatenate multiple volume files into one read-only continuous byte stream."""

    def __init__(self, paths: list[str]):
        parts = autodetect_parts(paths)
        self.paths = [p for p, _ in parts]
        self.skips = [s for _, s in parts]
        self.fhs = [open(p, "rb") for p in self.paths]
        self.sizes = [os.path.getsize(p) - s for p, s in parts]
        self.offsets = []
        acc = 0
        for s in self.sizes:
            self.offsets.append(acc)
            acc += s
        self.total = acc
        self.pos = 0

    def _locate(self, off: int) -> tuple[int, int]:
        """Global offset -> (which file, offset within file, already accounting for marker skip)."""
        for i, (o, s) in enumerate(zip(self.offsets, self.sizes)):
            if o <= off < o + s:
                return i, off - o + self.skips[i]
        return len(self.fhs) - 1, self.sizes[-1]

    def read_at(self, off: int, n: int) -> bytes:
        """Read n bytes from global offset off (auto-concatenated across volumes);
        if fewer are available, return what was actually read."""
        out = bytearray()
        remaining = n
        cur = off
        while remaining > 0 and cur < self.total:
            i, local = self._locate(cur)
            # The **physical** size of this volume = logical size + skipped marker bytes.
            # ⚠️ Easy to mistakenly write sizes[i] - local: that would deduct the skip a
            #    second time, so all data read past the volume boundary would be wrong
            #    (every decompression would fail).
            phys_size = self.sizes[i] + self.skips[i]
            self.fhs[i].seek(local)
            take = min(remaining, phys_size - local)
            if take <= 0:
                break
            chunk = self.fhs[i].read(take)
            if not chunk:
                break
            out += chunk
            cur += len(chunk)
            remaining -= len(chunk)
        return bytes(out)

    def close(self):
        for f in self.fhs:
            f.close()

    # ---- The methods below make it compatible with Python's standard-library zipfile ----
    # This way there is no need to first concatenate the volumes into a real 3.7GB
    # file (especially important when disk space is insufficient): ZipFile(stream)
    # can directly and precisely locate and extract members via the central directory.
    def seekable(self):
        return True

    def seek(self, off: int, whence: int = 0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.total + off
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = max(0, self.total - self.pos)
        data = self.read_at(self.pos, n)
        self.pos += len(data)
        return data

    def flush(self):
        pass


def parse_local_header(st: MultiPartStream, off: int):
    """Return (name, method, comp_size, data_off, flags, next_off) or None."""
    hdr = st.read_at(off, 30)
    if len(hdr) < 30 or not hdr.startswith(LH_SIG):
        return None
    # local header layout (from offset 4): ver H | flags H | method H | time H | date H
    #                                    crc I | csize I | usize I | namelen H | extralen H
    (_ver, flags, method, _tm, _dt, _crc, csize, usize,
     nlen, elen) = struct.unpack("<HHHHHIIIHH", hdr[4:30])
    name = st.read_at(off + 30, nlen).decode("utf-8", "replace")
    data_off = off + 30 + nlen + elen
    return name, method, csize, data_off, flags


def scan_entries(st: MultiPartStream, limit=0, progress=True):
    """Scan local headers sequentially to get the member list. Stop at the central directory."""
    entries = []
    off = 0
    n = 0
    bad_streak = 0
    while off < st.total and bad_streak < 3:
        head = st.read_at(off, 4)
        if head.startswith(CD_SIG) or head.startswith(DD_SIG):
            break                       # reached the central directory, end of the body
        got = parse_local_header(st, off)
        if got is None:
            # data area (actual data between local headers) — keep searching for the next signature
            off += 1
            bad_streak += 1
            continue
        name, method, csize, data_off, flags = got
        bad_streak = 0
        entries.append({
            "name": name, "method": method, "csize": csize,
            "off": data_off, "flags": flags,
        })
        off = data_off + csize
        n += 1
        if progress and n % 500 == 0:
            print(f"   scanning… {n} members  {off/st.total*100:.0f}%", flush=True)
        if limit and n >= limit:
            break
    return entries


def _stream_inflate(st: MultiPartStream, off: int, size: int) -> bytes:
    """Read in chunks + streaming inflate; raises zlib.error when data is truncated."""
    do = zlib.decompressobj(-zlib.MAX_WBITS)
    out = bytearray()
    CH = 1 << 20
    got = 0
    while got < size:
        take = min(CH, size - got)
        chunk = st.read_at(off + got, take)
        if not chunk:
            raise EOFError("data stream truncated (missing volume)")
        out += do.decompress(chunk)
        got += len(chunk)
    out += do.flush()
    return bytes(out)


def wanted(name: str, filters: list[str] | None) -> str | None:
    """Return the modality name this member matches, or None if it matches none."""
    base = os.path.basename(name)
    if not base:
        return None
    if not filters:
        return base
    low = base.lower()
    for f in filters:
        if low.startswith(f.lower()):
            return f
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts_dir", default=SPLIT_DIR)
    ap.add_argument("--list", action="store_true", help="only list members that can be scanned")
    ap.add_argument("--extract", action="store_true", help="perform extraction")
    ap.add_argument("--filter", default="mask,photo",
                    help="filter by filename prefix, comma-separated; empty string = all")
    ap.add_argument("--out", default=os.path.join(SPLIT_DIR, "patients_partial"))
    ap.add_argument("--limit", type=int, default=0, help="only process the first N members")
    ap.add_argument("--dry-run", action="store_true", help="only count, do not write to disk")
    ap.add_argument("--force-scan", action="store_true",
                    help="force skipping the central directory and scan local headers sequentially "
                         "instead. Use this when offsets in the central directory do not match the "
                         "real layout")
    args = ap.parse_args()

    parts = [os.path.join(args.parts_dir, p) for p in SPLIT_PARTS]
    if not os.path.exists(parts[0]):
        print(f"❌ {parts[0]} not found")
        return 1
    st = MultiPartStream(parts)
    print(f"▶ {len(st.paths)} volumes available, total {st.total/1e9:.2f} GB")
    for p, s in zip(st.paths, st.sizes):
        print(f"   {os.path.basename(p):16s} {s/1e6:9.1f} MB")

    # ---- Preferred: when the central directory is present (all volumes) use zipfile — most accurate and fastest ----
    print("\n▶ Trying to open via the central directory…")
    try:
        if args.force_scan:
            raise RuntimeError("--force-scan: skipping central directory")
        import collections
        import zipfile
        zf = zipfile.ZipFile(st)
        names = zf.namelist()
        print(f"   ✅ central directory available: {len(names)} members")
        cnt = collections.Counter(os.path.basename(n) for n in names)
        print("\n   counts by filename (top 15):")
        for k, v in cnt.most_common(15):
            print(f"     {k:44s} {v:5d}")
        if args.filter:
            fs = [f.lower() for f in args.filter.split(",") if f]
            hit = [n for n in names
                   if os.path.basename(n).lower().startswith(tuple(fs))]
            print(f"\n   matched filter {args.filter}: {len(hit)}")
            if args.extract:
                os.makedirs(args.out, exist_ok=True)
                ok = fail = 0
                for i, n in enumerate(hit):
                    try:
                        if not args.dry_run:
                            zf.extract(n, args.out)
                        ok += 1
                    except Exception:
                        fail += 1
                    if (i + 1) % 200 == 0:
                        print(f"   extracting… {i+1}/{len(hit)}  success {ok}  failed {fail}",
                              flush=True)
                print(f"\n{'✅' if fail == 0 else '⚠️'} success {ok}  failed {fail}")
                if not args.dry_run:
                    print(f"   output directory: {args.out}")
                if fail:
                    print("   (the failed members fall on the genuinely missing volume)")
        else:
            print("\n(add --extract to perform extraction)")
        st.close()
        return 0
    except Exception as e:
        print(f"   ⚠️ central directory unavailable ({type(e).__name__}: {str(e)[:80]})")
        print("      → falling back to sequential local-header scanning")
        st.seek(0)

    print("\n▶ Scanning member index (no central directory, locating via local headers one by one)…")
    entries = scan_entries(st, limit=args.limit)
    print(f"   found {len(entries)} members, data stream advanced to {st.offsets and st.total}")

    filters = [f for f in args.filter.split(",") if f] if args.filter else None
    hit = [e for e in entries if wanted(e["name"], filters)]
    print(f"   matched filter {filters or 'all'}: {len(hit)}")

    if args.list:
        import collections
        cnt = collections.Counter(wanted(e["name"], None) for e in hit)
        print("\n   counts by filename:")
        for k, v in cnt.most_common(20):
            print(f"     {k:42s} {v:5d}")
        st.close()
        return 0

    if not args.extract:
        print("\n(add --extract to perform extraction)")
        st.close()
        return 0

    os.makedirs(args.out, exist_ok=True)
    ok = fail = 0
    failed = []
    for i, e in enumerate(hit):
        kind = wanted(e["name"], filters)
        dst = os.path.join(args.out, e["name"].replace("\\", "/"))
        try:
            if e["method"] == 0:
                data = st.read_at(e["off"], e["csize"])
                if len(data) < e["csize"]:
                    raise EOFError("stored data incomplete")
            elif e["method"] == 8:
                data = _stream_inflate(st, e["off"], e["csize"])
            else:
                raise ValueError(f"unsupported compression method {e['method']}")
            if not args.dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f:
                    f.write(data)
            ok += 1
        except Exception as ex:
            fail += 1
            failed.append((e["name"], str(ex)[:60]))
        if (i + 1) % 100 == 0:
            print(f"   extracting… {i+1}/{len(hit)}  success {ok}  failed {fail}", flush=True)

    print(f"\n{'✅' if fail == 0 else '⚠️'} success {ok}  failed {fail}")
    if not args.dry_run:
        print(f"   output directory: {args.out}")
    for n, why in failed[:8]:
        print(f"   failure sample: {n}  ({why})")
    st.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
