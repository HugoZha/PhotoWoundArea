#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repair_split_zip.py — 从**缺少最后分卷**的 split zip 里抢救文件。

为什么需要它：
    WoundsDB 的 Patients 包被下载器切成 patients.zip + patients.z01…z05。
    最后一个分卷缺失时（当前正是如此，只剩到 z04），整个 zip 的**中央目录丢了**，
    `unzip` 会报一堆 "bad zipfile offset" 然后拒绝解压。

    但 zip 有个特性可以救命：每个成员前面都有独立的 **local file header**
    (`PK\\x03\\x04`)，里面有文件名、压缩方法、压缩后大小。中央目录只是一份"索引"，
    丢了索引不等于丢了数据 —— 顺着数据流**逐成员扫描**就能把还完整的文件救出来。

用法：
    # 1) 先看看能救出什么（只扫索引，不解压）
    python tools/repair_split_zip.py --list

    # 2) 只抢救需要的模态（默认 mask + photo）
    python tools/repair_split_zip.py --extract --filter mask,photo --out data/raw/woundsdb/patients_partial

    # 3) 只扫前 N 个成员/试跑，确认无误再全量
    python tools/repair_split_zip.py --extract --limit 200 --dry-run

⚠️ 局限：缺失分卷横跨的成员会解压失败并被跳过（尾部少量文件救不回来）。
   真要完整数据，还是应该用 tools/fetch_woundsdb.py 补齐最后一片后正常解压。
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
    """自动判定分卷顺序与是否要跳过 spanning marker，返回 [(路径, 跳过的字节数)]。

    Info-ZIP split-archive conventions:
      · 只有**第一卷**开头有 4 字节 spanning marker `PK\\x07\\x08`，其余卷直接是数据；
      · **最后一卷**尾部才有 EOCD `PK\\x05\\x06`（也就是整个包的中央目录）；
      · 最后一卷的文件名**不一定**排在最后（本次就是 patients.zip，而 z01 才是第一卷）。
    所以不能按文件名排序，必须靠这两个标记定位。
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
    """把多个分卷文件拼成一个只读的连续字节流。"""

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
        """全局偏移 -> (第几个文件, 文件内偏移，已计入 marker 跳过)。"""
        for i, (o, s) in enumerate(zip(self.offsets, self.sizes)):
            if o <= off < o + s:
                return i, off - o + self.skips[i]
        return len(self.fhs) - 1, self.sizes[-1]

    def read_at(self, off: int, n: int) -> bytes:
        """从全局偏移 off 读 n 字节（跨分卷自动拼接），不足则返回实际读到的。"""
        out = bytearray()
        remaining = n
        cur = off
        while remaining > 0 and cur < self.total:
            i, local = self._locate(cur)
            # 该分卷的**物理**大小 = 逻辑大小 + 跳过的 marker 字节数。
            # ⚠️ 容易写错成 sizes[i] - local：那样会把 skip 重复扣一次，
            #    越过分卷边界后读到的数据全是错的（解压会全部失败）。
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

    # ---- 下面几个方法让它兼容 Python 标准库 zipfile ----
    # 这样无需先把分卷拼成一个 3.7GB 的真实文件（磁盘不够时尤其重要），
    # 直接 ZipFile(stream) 就能按中央目录精确定位并抽取成员。
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
    """返回 (name, method, comp_size, data_off, flags, next_off) 或 None。"""
    hdr = st.read_at(off, 30)
    if len(hdr) < 30 or not hdr.startswith(LH_SIG):
        return None
    # local header 布局(从偏移4起): ver H | flags H | method H | time H | date H
    #                              crc I | csize I | usize I | namelen H | extralen H
    (_ver, flags, method, _tm, _dt, _crc, csize, usize,
     nlen, elen) = struct.unpack("<HHHHHIIIHH", hdr[4:30])
    name = st.read_at(off + 30, nlen).decode("utf-8", "replace")
    data_off = off + 30 + nlen + elen
    return name, method, csize, data_off, flags


def scan_entries(st: MultiPartStream, limit=0, progress=True):
    """顺序扫描 local header 得到成员列表。遇到中央目录停止。"""
    entries = []
    off = 0
    n = 0
    bad_streak = 0
    while off < st.total and bad_streak < 3:
        head = st.read_at(off, 4)
        if head.startswith(CD_SIG) or head.startswith(DD_SIG):
            break                       # 到了中央目录，正文结束
        got = parse_local_header(st, off)
        if got is None:
            # 数据区（ local header 之间有实际数据）—— 往下找下一个签名
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
            print(f"   扫描中… {n} 个成员  {off/st.total*100:.0f}%", flush=True)
        if limit and n >= limit:
            break
    return entries


def _stream_inflate(st: MultiPartStream, off: int, size: int) -> bytes:
    """按块读+流式 inflate；数据被截断时抛出 zlib.error。"""
    do = zlib.decompressobj(-zlib.MAX_WBITS)
    out = bytearray()
    CH = 1 << 20
    got = 0
    while got < size:
        take = min(CH, size - got)
        chunk = st.read_at(off + got, take)
        if not chunk:
            raise EOFError("数据流被截断（缺分卷）")
        out += do.decompress(chunk)
        got += len(chunk)
    out += do.flush()
    return bytes(out)


def wanted(name: str, filters: list[str] | None) -> str | None:
    """返回该成员命中的模态名，未命中返回 None。"""
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
    ap.add_argument("--list", action="store_true", help="只列出能扫描到的成员")
    ap.add_argument("--extract", action="store_true", help="执行抽取")
    ap.add_argument("--filter", default="mask,photo",
                    help="按文件名前缀筛选，逗号分隔；空字符串=全部")
    ap.add_argument("--out", default=os.path.join(SPLIT_DIR, "patients_partial"))
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个成员")
    ap.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    ap.add_argument("--force-scan", action="store_true",
                    help="强制跳过中央目录，改为顺序扫描 local header。"
                         "当中央目录里的偏移量与真实布局不一致时用它")
    args = ap.parse_args()

    parts = [os.path.join(args.parts_dir, p) for p in SPLIT_PARTS]
    if not os.path.exists(parts[0]):
        print(f"❌ 找不到 {parts[0]}")
        return 1
    st = MultiPartStream(parts)
    print(f"▶ 可用分卷 {len(st.paths)} 个，合计 {st.total/1e9:.2f} GB")
    for p, s in zip(st.paths, st.sizes):
        print(f"   {os.path.basename(p):16s} {s/1e6:9.1f} MB")

    # ---- 首选：中央目录还在（分卷齐全）时用 zipfile，最准也最快 ----
    print("\n▶ 尝试按中央目录打开…")
    try:
        if args.force_scan:
            raise RuntimeError("--force-scan：跳过中央目录")
        import collections
        import zipfile
        zf = zipfile.ZipFile(st)
        names = zf.namelist()
        print(f"   ✅ 中央目录可用：{len(names)} 个成员")
        cnt = collections.Counter(os.path.basename(n) for n in names)
        print("\n   按文件名统计（前 15）：")
        for k, v in cnt.most_common(15):
            print(f"     {k:44s} {v:5d}")
        if args.filter:
            fs = [f.lower() for f in args.filter.split(",") if f]
            hit = [n for n in names
                   if os.path.basename(n).lower().startswith(tuple(fs))]
            print(f"\n   命中过滤 {args.filter}：{len(hit)} 个")
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
                        print(f"   抽取中… {i+1}/{len(hit)}  成功 {ok}  失败 {fail}",
                              flush=True)
                print(f"\n{'✅' if fail == 0 else '⚠️'} 成功 {ok}  失败 {fail}")
                if not args.dry_run:
                    print(f"   产物目录：{args.out}")
                if fail:
                    print("   （失败的成员落在确实缺失的分卷上）")
        else:
            print("\n（加 --extract 才会抽取）")
        st.close()
        return 0
    except Exception as e:
        print(f"   ⚠️ 中央目录不可用（{type(e).__name__}: {str(e)[:80]}）")
        print("      → 退化为顺序扫描 local header 的兜底方案")
        st.seek(0)

    print("\n▶ 扫描成员索引（缺中央目录，靠 local header 逐个找）…")
    entries = scan_entries(st, limit=args.limit)
    print(f"   找到 {len(entries)} 个成员，数据流推进到 {st.offsets and st.total}")

    filters = [f for f in args.filter.split(",") if f] if args.filter else None
    hit = [e for e in entries if wanted(e["name"], filters)]
    print(f"   命中过滤 {filters or '全部'}：{len(hit)} 个")

    if args.list:
        import collections
        cnt = collections.Counter(wanted(e["name"], None) for e in hit)
        print("\n   按文件名统计：")
        for k, v in cnt.most_common(20):
            print(f"     {k:42s} {v:5d}")
        st.close()
        return 0

    if not args.extract:
        print("\n（加 --extract 才会抽取）")
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
                    raise EOFError("stored 数据不完整")
            elif e["method"] == 8:
                data = _stream_inflate(st, e["off"], e["csize"])
            else:
                raise ValueError(f"不支持的压缩方法 {e['method']}")
            if not args.dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f:
                    f.write(data)
            ok += 1
        except Exception as ex:
            fail += 1
            failed.append((e["name"], str(ex)[:60]))
        if (i + 1) % 100 == 0:
            print(f"   抽取中… {i+1}/{len(hit)}  成功 {ok}  失败 {fail}", flush=True)

    print(f"\n{'✅' if fail == 0 else '⚠️'} 成功 {ok}  失败 {fail}")
    if not args.dry_run:
        print(f"   产物目录：{args.out}")
    for n, why in failed[:8]:
        print(f"   失败样例：{n}  ({why})")
    st.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
