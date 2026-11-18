#!/usr/bin/env python3
"""拼接 FNSPID AAL 数据集：样本新闻 + 官方股价 → processed/。

用法:
    python prior_enhance/data/fnspid/build_dataset.py
    python prior_enhance/data/fnspid/build_dataset.py --symbol AAL
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
RAW = DIR / "raw"
PROCESSED = DIR / "processed"


def extract_news_from_sample(symbol: str) -> Path:
    sample = RAW / "news_sample_nasdaq.csv"
    out = RAW / f"news_{symbol}.csv"
    if not sample.is_file():
        raise FileNotFoundError(f"缺少 {sample}，请先运行 download_fnspid.py")

    rows_out = 0
    with open(sample, encoding="utf-8", newline="") as fin, open(
        out, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        sym_col = "Stock_symbol" if "Stock_symbol" in header else "Stock_Symbol"
        sym_idx = header.index(sym_col)
        for row in reader:
            if len(row) > sym_idx and row[sym_idx].strip().upper() == symbol:
                writer.writerow(row)
                rows_out += 1

    if rows_out == 0:
        raise RuntimeError(f"样本中未找到 {symbol} 新闻")
    print(f">>> news_{symbol}.csv: {rows_out} 行 (from news_sample_nasdaq.csv)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="拼接 FNSPID 单股数据集")
    parser.add_argument("--symbol", default="AAL")
    args = parser.parse_args()
    symbol = args.symbol.upper()

    stock = RAW / f"stock_{symbol}.csv"
    if not stock.is_file():
        raise FileNotFoundError(
            f"缺少 {stock}，请先: python prior_enhance/data/fnspid/download_fnspid.py "
            f"--symbol {symbol} --stock-only"
        )

    extract_news_from_sample(symbol)
    print(f">>> stock_{symbol}.csv: {stock.stat().st_size / 1024:.1f} KB (FNSPID 官方)")

    prepare = DIR / "prepare_demo.py"
    subprocess.run([sys.executable, str(prepare), "--symbol", symbol], check=True)

    summary_path = PROCESSED / f"{symbol}_analysis_summary.json"
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    manifest = {
        "symbol": symbol,
        "description": "FNSPID 样本新闻 + 官方股价，供 prior_enhance 使用",
        "raw": {
            "news": str(RAW / f"news_{symbol}.csv"),
            "stock": str(stock),
            "news_origin": "news_sample_nasdaq.csv (流式样本子集，非全量)",
            "stock_origin": "FNSPID full_history.zip / stock_{symbol}.csv",
        },
        "processed": summary.get("outputs", {}),
        "stats": {
            "news_rows": summary.get("news_rows_in_sample"),
            "news_date_range": [summary.get("news_date_min"), summary.get("news_date_max")],
            "price_rows": summary.get("price_rows"),
            "price_date_range": [summary.get("price_start"), summary.get("price_end")],
            "decision_time": summary.get("decision_time"),
            "horizon": summary.get("horizon"),
        },
        "sources": {
            "news_source": summary.get("news_source"),
            "price_source": summary.get("price_source"),
        },
    }
    manifest_path = PROCESSED / f"{symbol}_dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f">>> 数据集清单: {manifest_path}")
    print(json.dumps(manifest["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
