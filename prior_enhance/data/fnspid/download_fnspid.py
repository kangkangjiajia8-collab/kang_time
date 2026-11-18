#!/usr/bin/env python3
"""下载 FNSPID 子集到 prior_enhance/data/fnspid/raw/。

完整新闻 CSV 约 23GB，默认只流式采样；股价 zip 约 589MB 全量下载。
可按股票代码单独拉取新闻（流式过滤）或从 zip 抽取单股 CSV（HTTP Range，约百 KB）。

用法:
    python prior_enhance/data/fnspid/download_fnspid.py
    python prior_enhance/data/fnspid/download_fnspid.py --news-max-rows 50000
    python prior_enhance/data/fnspid/download_fnspid.py --symbol AAL --news-only
    python prior_enhance/data/fnspid/download_fnspid.py --symbol AAL --stock-only
    python prior_enhance/data/fnspid/download_fnspid.py --symbol AAL
    HF_ENDPOINT=https://hf-mirror.com python prior_enhance/data/fnspid/download_fnspid.py
"""

from __future__ import annotations

import argparse
import csv
import io
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, hf_hub_url

RAW = Path(__file__).resolve().parent / "raw"
REPO = "Zihan1004/FNSPID"
NEWS_FILE = "Stock_news/nasdaq_exteral_data.csv"
ZIP_FILE = "Stock_price/full_history.zip"
ZIP_MIN_BYTES = 560 * 1024 * 1024


def _news_file_valid(path: Path, min_rows: int = 1) -> bool:
    """新闻 CSV 有效：除表头外至少 min_rows 行。"""
    if not path.is_file() or path.stat().st_size < 200:
        return False
    try:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for i, _ in enumerate(reader):
                if i + 1 >= min_rows:
                    return True
        return False
    except OSError:
        return False


def _news_url() -> str:
    return hf_hub_url(REPO, NEWS_FILE, repo_type="dataset")


def _zip_url() -> str:
    return hf_hub_url(REPO, ZIP_FILE, repo_type="dataset")


def _symbol_col_index(header: list[str]) -> int:
    for name in ("Stock_symbol", "Stock_Symbol", "stock_symbol"):
        if name in header:
            return header.index(name)
    raise RuntimeError(f"未找到 Stock_symbol 列: {header}")


def _stream_news_sample(out_path: Path, max_rows: int, max_bytes: int) -> int:
    buf = io.BytesIO()
    with requests.get(_news_url(), stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                break
            buf.write(chunk)
            if buf.tell() >= max_bytes:
                break

    text = buf.getvalue().decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) < 2:
        raise RuntimeError("新闻流式采样失败：行数不足")

    rows_written = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        reader = csv.reader(lines)
        header = next(reader)
        writer = csv.writer(f)
        writer.writerow(header)
        for row in reader:
            if len(row) < len(header):
                continue
            writer.writerow(row)
            rows_written += 1
            if rows_written >= max_rows:
                break
    return rows_written


def _stream_news_for_symbol(out_path: Path, symbol: str) -> int:
    """流式扫描 nasdaq CSV（按 Stock_symbol 字母序），只保留指定股票行。"""
    symbol = symbol.upper()
    rows_written = 0
    in_target = False
    bytes_read = 0
    last_report = 0

    print(f">>> 流式过滤新闻: {symbol}（需扫描至该 symbol 区块结束）")
    with requests.get(_news_url(), stream=True, timeout=120) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True
        text_stream = io.TextIOWrapper(resp.raw, encoding="utf-8", errors="replace")
        reader = csv.reader(text_stream)
        header = next(reader)
        sym_idx = _symbol_col_index(header)

        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in reader:
                if hasattr(resp.raw, "tell"):
                    bytes_read = resp.raw.tell()
                if bytes_read - last_report >= 256 * 1024 * 1024:
                    print(f"    已扫描 {bytes_read / 1e9:.2f} GB …")
                    last_report = bytes_read

                if len(row) <= sym_idx:
                    continue
                sym = row[sym_idx].strip().upper()
                if sym == symbol:
                    in_target = True
                    writer.writerow(row)
                    rows_written += 1
                elif in_target:
                    break

    if rows_written == 0:
        raise RuntimeError(
            f"未找到 {symbol} 的新闻（可能 symbol 不存在，或需扫描完整 23GB 文件）"
        )
    print(f"    扫描共 {bytes_read / 1e9:.2f} GB")
    return rows_written


def _parse_zip_central_directory(url: str) -> tuple[int, list[tuple[str, int, int, int, int]]]:
    """返回 (zip_size, [(filename, comp_size, uncomp_size, local_offset, comp_method), ...])。"""
    head = requests.head(url, timeout=60, allow_redirects=True)
    head.raise_for_status()
    size = int(head.headers.get("content-length", 0))
    if size <= 0:
        raise RuntimeError("无法获取 zip 大小")

    tail_size = min(size, 256 * 1024)
    tail = requests.get(
        url,
        headers={"Range": f"bytes={size - tail_size}-{size - 1}"},
        timeout=120,
    )
    tail.raise_for_status()
    data = tail.content

    eocd_sig = b"\x50\x4b\x05\x06"
    idx = data.rfind(eocd_sig)
    if idx < 0:
        raise RuntimeError("zip 中央目录解析失败")

    cd_size = struct.unpack("<I", data[idx + 12 : idx + 16])[0]
    cd_offset = struct.unpack("<I", data[idx + 16 : idx + 20])[0]

    cd_resp = requests.get(
        url,
        headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"},
        timeout=120,
    )
    cd_resp.raise_for_status()
    cd = cd_resp.content

    entries: list[tuple[str, int, int, int, int]] = []
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != b"\x50\x4b\x01\x02":
            break
        comp_method = struct.unpack("<H", cd[pos + 10 : pos + 12])[0]
        comp_size = struct.unpack("<I", cd[pos + 20 : pos + 24])[0]
        uncomp_size = struct.unpack("<I", cd[pos + 24 : pos + 28])[0]
        fn_len = struct.unpack("<H", cd[pos + 28 : pos + 30])[0]
        extra = struct.unpack("<H", cd[pos + 30 : pos + 32])[0]
        comment = struct.unpack("<H", cd[pos + 32 : pos + 34])[0]
        local_offset = struct.unpack("<I", cd[pos + 42 : pos + 46])[0]
        fn = cd[pos + 46 : pos + 46 + fn_len].decode("utf-8", errors="replace")
        entries.append((fn, comp_size, uncomp_size, local_offset, comp_method))
        pos += 46 + fn_len + extra + comment
    return size, entries


def _find_stock_entry(
    entries: list[tuple[str, int, int, int, int]], symbol: str
) -> tuple[str, int, int, int, int]:
    symbol = symbol.upper()
    candidates = [
        e
        for e in entries
        if e[0].endswith(".csv")
        and not e[0].startswith("__MACOSX")
        and e[0].split("/")[-1].upper() == f"{symbol}.CSV"
    ]
    if not candidates:
        candidates = [
            e
            for e in entries
            if e[0].endswith(".csv")
            and not e[0].startswith("__MACOSX")
            and symbol in e[0].upper()
        ]
    if not candidates:
        raise FileNotFoundError(f"zip 目录中未找到 {symbol} 股价文件")
    return candidates[0]


def _extract_stock_via_range(symbol: str, out_path: Path) -> None:
    """HTTP Range 从 remote zip 抽取单股 CSV，无需下载完整 589MB。"""
    url = _zip_url()
    _, entries = _parse_zip_central_directory(url)
    fn, comp_size, uncomp_size, local_offset, comp_method = _find_stock_entry(entries, symbol)

    lh = requests.get(
        url,
        headers={"Range": f"bytes={local_offset}-{local_offset + 30}"},
        timeout=60,
    )
    lh.raise_for_status()
    fn_len = struct.unpack("<H", lh.content[26:28])[0]
    extra = struct.unpack("<H", lh.content[28:30])[0]
    data_start = local_offset + 30 + fn_len + extra
    data_end = data_start + comp_size - 1

    print(f">>> Range 抽取 {fn} ({comp_size / 1024:.1f} KB 压缩)")
    comp = requests.get(
        url,
        headers={"Range": f"bytes={data_start}-{data_end}"},
        timeout=120,
    )
    comp.raise_for_status()
    if len(comp.content) != comp_size:
        raise RuntimeError(f"压缩块大小不符: 期望 {comp_size}, 实际 {len(comp.content)}")

    if comp_method == 0:
        raw = comp.content
    elif comp_method == 8:
        raw = zlib.decompress(comp.content, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"不支持的压缩方式: {comp_method}")

    if uncomp_size and len(raw) != uncomp_size:
        print(f"    警告: 解压大小 {len(raw)} != 目录记录 {uncomp_size}", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)


def _extract_stock_from_local_zip(symbol: str, zip_path: Path, out_path: Path) -> None:
    sym_file = f"{symbol.lower()}.csv"
    with zipfile.ZipFile(zip_path) as zf:
        match = next((n for n in zf.namelist() if n.lower().endswith(sym_file)), None)
        if match is None:
            match = next(
                (n for n in zf.namelist() if symbol.lower() in n.lower() and n.endswith(".csv")),
                None,
            )
        if match is None:
            raise FileNotFoundError(f"本地 zip 中未找到 {symbol}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(zf.read(match))


def _download_stock_zip(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "full_history.zip"

    if zip_path.is_file() and zip_path.stat().st_size >= ZIP_MIN_BYTES:
        return zip_path

    print(">>> huggingface_hub 下载股价 zip（支持断点续传）")
    downloaded = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=ZIP_FILE,
        local_dir=str(raw_dir),
    )
    zip_path = Path(downloaded)
    if zip_path.stat().st_size < ZIP_MIN_BYTES:
        raise RuntimeError(f"zip 不完整: {zip_path.stat().st_size} bytes")
    return zip_path


def _download_stock_for_symbol(symbol: str, raw_dir: Path, use_range: bool = True) -> Path:
    """下载单股 FNSPID 官方股价 CSV → raw/stock_{SYMBOL}.csv。"""
    out_path = raw_dir / f"stock_{symbol.upper()}.csv"
    if out_path.is_file() and out_path.stat().st_size > 100:
        print(f"股价已存在: {out_path}")
        return out_path

    zip_path = raw_dir / "full_history.zip"
    if zip_path.is_file() and zip_path.stat().st_size >= ZIP_MIN_BYTES:
        print(f">>> 从本地 zip 解压 {symbol}")
        _extract_stock_from_local_zip(symbol, zip_path, out_path)
        return out_path

    if use_range:
        try:
            _extract_stock_via_range(symbol, out_path)
            return out_path
        except Exception as exc:
            print(f"Range 抽取失败 ({exc})，改下完整 zip …", file=sys.stderr)

    path = _download_stock_zip(raw_dir)
    _extract_stock_from_local_zip(symbol, path, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 FNSPID 数据子集")
    parser.add_argument("--symbol", type=str, help="股票代码，如 AAL")
    parser.add_argument("--news-only", action="store_true", help="只下载该 symbol 的新闻")
    parser.add_argument("--stock-only", action="store_true", help="只下载该 symbol 的股价")
    parser.add_argument("--no-range", action="store_true", help="股价不用 HTTP Range，改下完整 zip")
    parser.add_argument("--news-max-rows", type=int, default=20000)
    parser.add_argument("--news-max-mb", type=int, default=80)
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-stock", action="store_true")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)

    if args.symbol:
        symbol = args.symbol.upper()
        do_news = args.news_only or (not args.news_only and not args.stock_only)
        do_stock = args.stock_only or (not args.news_only and not args.stock_only)

        if do_news:
            news_path = RAW / f"news_{symbol}.csv"
            if _news_file_valid(news_path):
                with open(news_path, encoding="utf-8", newline="") as f:
                    n_existing = sum(1 for _ in csv.reader(f)) - 1
                print(f"新闻已存在: {news_path} ({n_existing} 行)")
            else:
                if news_path.is_file():
                    print(f"新闻文件无效（仅表头或为空），重新下载: {news_path}")
                n = _stream_news_for_symbol(news_path, symbol)
                print(f"新闻: {news_path} ({n} 行)")

        if do_stock:
            path = _download_stock_for_symbol(symbol, RAW, use_range=not args.no_range)
            print(f"股价: {path} ({path.stat().st_size / 1024:.1f} KB)")
        return

    news_path = RAW / "news_sample_nasdaq.csv"

    if not args.skip_news:
        if news_path.is_file() and news_path.stat().st_size > 1000:
            print(f"新闻样本已存在: {news_path}")
        else:
            n = _stream_news_sample(
                news_path,
                max_rows=args.news_max_rows,
                max_bytes=args.news_max_mb * 1024 * 1024,
            )
            print(f"新闻样本: {news_path} ({n} 行)")

    if not args.skip_stock:
        try:
            path = _download_stock_zip(RAW)
            print(f"股价 zip: {path} ({path.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:
            print(f"zip 下载失败: {exc}", file=sys.stderr)
            print("可手动: HF_ENDPOINT=https://hf-mirror.com python", __file__)
            sys.exit(1)

    try:
        hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename="README.md",
            local_dir=str(RAW),
        )
    except Exception as exc:
        print(f"README 下载跳过: {exc}")


if __name__ == "__main__":
    main()
