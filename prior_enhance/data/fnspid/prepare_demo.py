#!/usr/bin/env python3
"""将 FNSPID 子集整理为 prior_enhance 可用的日频股价 + 新闻事件文本。

输出 (processed/):
  - {symbol}_daily.csv      timestamp, value, item_id（Adj Close）
  - {symbol}_news_events.json
  - {symbol}_event_text.txt  # 决策时刻前 lookback 天内新闻，供 E_m
  - analysis_summary.json

用法:
    python prior_enhance/data/fnspid/prepare_demo.py --symbol AAL
    python prior_enhance/data/fnspid/prepare_demo.py --symbol AAL --quick   # 不等 zip，股价用 yfinance
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

FNspid_DIR = Path(__file__).resolve().parent
RAW = FNspid_DIR / "raw"
PROCESSED = FNspid_DIR / "processed"
ZIP_MIN_BYTES = 560 * 1024 * 1024


def _news_file_valid(path: Path, min_rows: int = 1) -> bool:
    if not path.is_file() or path.stat().st_size < 200:
        return False
    try:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for i, _ in enumerate(reader):
                if i + 1 >= min_rows:
                    return True
        return False
    except OSError:
        return False


def _normalize_price_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    date_col = lower.get("date", df.columns[0])
    close_col = lower.get("adj close") or lower.get("adj_close") or lower.get("close")
    if close_col is None:
        raise KeyError(f"未找到收盘价列: {list(df.columns)}")
    return df, date_col, close_col


def _load_stock_from_csv(symbol: str, csv_path: Path) -> pd.DataFrame:
    df, date_col, close_col = _normalize_price_columns(pd.read_csv(csv_path))
    df["timestamp"] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").dropna(subset=[close_col])
    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "value": df[close_col].astype(float),
        "item_id": symbol.upper(),
    })
    return out.reset_index(drop=True)


def _load_stock_from_zip(symbol: str, zip_path: Path) -> pd.DataFrame:
    sym_file = f"{symbol.lower()}.csv"
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        match = next((n for n in names if n.lower().endswith(sym_file)), None)
        if match is None:
            match = next((n for n in names if symbol.lower() in n.lower() and n.endswith(".csv")), None)
        if match is None:
            raise FileNotFoundError(f"zip 中未找到 {symbol} 股价文件，候选: {names[:10]}")
        with zf.open(match) as f:
            df = pd.read_csv(f)
    df, date_col, close_col = _normalize_price_columns(df)
    df["timestamp"] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").dropna(subset=[close_col])
    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "value": df[close_col].astype(float),
        "item_id": symbol.upper(),
    })
    return out.reset_index(drop=True)


def _load_stock_synthetic(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """无网络时的 demo 回退：按新闻时间范围生成可运行的日频序列（非真实股价）。"""
    import numpy as np

    start_ts = pd.Timestamp(start or "2009-10-01")
    end_ts = pd.Timestamp(end or "2023-12-31")
    idx = pd.bdate_range(start=start_ts, end=end_ts)
    if len(idx) < 60:
        raise ValueError("合成序列太短")
    seed = sum(ord(c) for c in symbol.upper()) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, 0.018, len(idx))
    base = {"AAL": 18.0, "AA": 35.0, "A": 130.0}.get(symbol.upper(), 50.0)
    values = base * np.exp(np.cumsum(log_rets))
    return pd.DataFrame({
        "timestamp": idx,
        "value": values.astype(float),
        "item_id": symbol.upper(),
    })


def _load_stock_stooq(symbol: str) -> pd.DataFrame:
    """Stooq 日频 CSV，无需 API key。"""
    import io
    import requests

    url = f"https://stooq.pl/q/d/l/?s={symbol.lower()}.us&i=d"
    resp = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 (compatible; prior_enhance/1.0)"},
    )
    resp.raise_for_status()
    if resp.text.lstrip().startswith("<!"):
        raise ValueError("Stooq 返回 HTML 而非 CSV")
    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty or "Date" not in df.columns:
        raise ValueError(f"Stooq 未返回 {symbol} 数据")
    df["timestamp"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "value": df["Close"].astype(float),
        "item_id": symbol.upper(),
    })
    return out.dropna().sort_values("timestamp").reset_index(drop=True)


def _load_stock_yfinance(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """quick 模式：yfinance → Stooq → 合成序列（仅 demo）。"""
    try:
        import yfinance as yf
    except ImportError:
        yf = None

    end = end or "2024-01-01"
    start = start or "2009-01-01"

    if yf is not None:
        try:
            t = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
            if not t.empty:
                if isinstance(t.columns, pd.MultiIndex):
                    t.columns = [c[0] if isinstance(c, tuple) else c for c in t.columns]
                close_col = "Close" if "Close" in t.columns else t.columns[0]
                df = t.reset_index()
                date_col = "Date" if "Date" in df.columns else df.columns[0]
                df["timestamp"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
                out = pd.DataFrame({
                    "timestamp": df["timestamp"],
                    "value": df[close_col].astype(float),
                    "item_id": symbol.upper(),
                })
                return out.dropna().sort_values("timestamp").reset_index(drop=True)
        except Exception:
            pass

    try:
        df = _load_stock_stooq(symbol)
        df = df[(df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))]
        if len(df) >= 60:
            return df.reset_index(drop=True)
    except Exception:
        pass

    return _load_stock_synthetic(symbol, start=start, end=end)


def _load_news(news_path: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(news_path, low_memory=False)
    sym_col = "Stock_symbol" if "Stock_symbol" in df.columns else "Stock_Symbol"
    df = df[df[sym_col].astype(str).str.upper() == symbol.upper()].copy()
    df["news_time"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["news_time"])
    title_col = "Article_title" if "Article_title" in df.columns else "Title"
    df["title"] = df[title_col].astype(str)
  # 摘要列优先
    for col in ("Textrank_summary", "Lsa_summary", "Luhn_summary", "Lexrank_summary", "Article"):
        if col in df.columns:
            df["summary"] = df[col].fillna("").astype(str)
            break
    else:
        df["summary"] = ""
    return df.sort_values("news_time")


def build_event_text(
    news_df: pd.DataFrame,
    decision_time: pd.Timestamp,
    symbol: str,
    lookback_days: int = 21,
    max_items: int = 10,
) -> str:
    start = decision_time - pd.Timedelta(days=lookback_days)
    mask = (news_df["news_time"] > start) & (news_df["news_time"] <= decision_time)
    subset = news_df.loc[mask].sort_values("news_time", ascending=False).head(max_items)
    if subset.empty:
        return (
            f"预测决策时刻 {decision_time} 前 {lookback_days} 天内，"
            f"FNSPID 新闻样本中无 {symbol} 相关报道。"
        )
    lines = [
        f"近期金融新闻（FNSPID / Nasdaq，决策前 {lookback_days} 天，共 {len(subset)} 条）："
    ]
    for _, row in subset.iterrows():
        t = row["news_time"].strftime("%Y-%m-%d %H:%M")
        title = row["title"].strip()
        summary = (row.get("summary") or "").strip()
        if summary and len(summary) > 20:
            snippet = summary[:200].replace("\n", " ")
            lines.append(f"- [{t}] {title} | {snippet}")
        else:
            lines.append(f"- [{t}] {title}")
    return "\n".join(lines)


def analyze(news_df: pd.DataFrame, price_df: pd.DataFrame, symbol: str) -> dict:
    pub = news_df["Publisher"].value_counts().head(5).to_dict() if "Publisher" in news_df.columns else {}
    return {
        "symbol": symbol,
        "price_rows": len(price_df),
        "price_start": str(price_df["timestamp"].min()),
        "price_end": str(price_df["timestamp"].max()),
        "price_mean": round(float(price_df["value"].mean()), 4),
        "price_std": round(float(price_df["value"].std()), 4),
        "news_rows_in_sample": len(news_df),
        "news_date_min": str(news_df["news_time"].min()) if len(news_df) else None,
        "news_date_max": str(news_df["news_time"].max()) if len(news_df) else None,
        "top_publishers": pub,
        "sample_titles": news_df["title"].head(5).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAL", help="股票代码，默认 AAL（样本新闻较多）")
    parser.add_argument("--lookback-days", type=int, default=21)
    parser.add_argument("--horizon", type=int, default=5, help="holdout 预测步数（日）")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="不等 full_history.zip；新闻仍用 FNSPID 样本，股价用 yfinance 补齐",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    zip_path = RAW / "full_history.zip"
    stock_csv = RAW / f"stock_{symbol}.csv"
    news_path = RAW / f"news_{symbol}.csv"
    news_source = f"news_{symbol}.csv"
    if not _news_file_valid(news_path):
        if news_path.is_file():
            print(f"警告: {news_path} 无数据行，回退到 news_sample_nasdaq.csv")
        news_path = RAW / "news_sample_nasdaq.csv"
        news_source = "news_sample_nasdaq.csv"
    if not news_path.is_file():
        raise FileNotFoundError(
            f"缺少新闻 CSV。请先: python prior_enhance/data/fnspid/download_fnspid.py "
            f"--symbol {symbol} --news-only"
        )

    news_df = _load_news(news_path, symbol)
    price_source = "fnspid_zip"
    if args.quick:
        news_end = news_df["news_time"].max()
        news_start = news_df["news_time"].min()
        price_df = _load_stock_yfinance(
            symbol,
            start=str(news_start.date() - pd.Timedelta(days=30)),
            end=str(news_end.date() + pd.Timedelta(days=7)),
        )
        price_source = "quick_fallback"
    elif stock_csv.is_file():
        price_df = _load_stock_from_csv(symbol, stock_csv)
        price_source = "fnspid_stock_csv"
    else:
        if not zip_path.is_file() or zip_path.stat().st_size < ZIP_MIN_BYTES:
            raise FileNotFoundError(
                f"股价未就绪: {stock_csv} 或 {zip_path}。"
                f"请: python prior_enhance/data/fnspid/download_fnspid.py --symbol {symbol} --stock-only"
                " 或加 --quick"
            )
        price_df = _load_stock_from_zip(symbol, zip_path)
    news_df["item_id"] = symbol

    horizon = args.horizon
    if len(price_df) < 48 + horizon:
        raise ValueError(f"股价行数不足: {len(price_df)}")

    decision_time = price_df["timestamp"].iloc[len(price_df) - horizon - 1]
    event_text = build_event_text(
        news_df, decision_time, symbol, lookback_days=args.lookback_days,
    )

    csv_out = PROCESSED / f"{symbol}_daily.csv"
    price_df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    events_out = PROCESSED / f"{symbol}_news_events.json"
    events = [
        {
            "time": str(r["news_time"]),
            "title": r["title"],
            "summary": (r.get("summary") or "")[:500],
        }
        for _, r in news_df.iterrows()
    ]
    with open(events_out, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    text_out = PROCESSED / f"{symbol}_event_text.txt"
    text_out.write_text(event_text, encoding="utf-8")

    summary = analyze(news_df, price_df, symbol)
    summary["news_source"] = news_source
    summary["price_source"] = price_source
    summary["mode"] = "quick" if args.quick else "full"
    summary["decision_time"] = str(decision_time)
    summary["horizon"] = horizon
    summary["event_text_preview"] = event_text[:800]
    summary["outputs"] = {
        "daily_csv": str(csv_out),
        "news_events": str(events_out),
        "event_text": str(text_out),
    }
    summary_path = PROCESSED / f"{symbol}_analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
