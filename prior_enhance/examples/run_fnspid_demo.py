#!/usr/bin/env python3
"""
FNSPID 金融新闻 + 股价 demo：将新闻作为 E_m 注入 prior_enhance。

数据:
  prior_enhance/data/fnspid/processed/{SYMBOL}_daily.csv
  prior_enhance/data/fnspid/processed/{SYMBOL}_event_text.txt

准备数据:
  HF_ENDPOINT=https://hf-mirror.com python prior_enhance/data/fnspid/download_fnspid.py
  python prior_enhance/data/fnspid/prepare_demo.py --symbol AAL

用法:
  python prior_enhance/examples/run_fnspid_demo.py
  python prior_enhance/examples/run_fnspid_demo.py --symbol AAL --with-search
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prior_enhance._bootstrap import bootstrap  # noqa: E402

bootstrap(ROOT)

from prior_enhance._path import setup_autoapex_path  # noqa: E402

setup_autoapex_path()

from prior_enhance.config import PriorEnhanceConfig  # noqa: E402
from prior_enhance.data.window import build_forecast_task, build_windows, HOLDOUT_TAG  # noqa: E402
from prior_enhance.pipeline import PriorEnhancePipeline  # noqa: E402

FNspid_DIR = ROOT / "prior_enhance" / "data" / "fnspid"
PROCESSED = FNspid_DIR / "processed"

FIN_DOMAIN = (
    "美股日频收盘价预测：受财报、分析师评级、宏观新闻与行业情绪影响；"
    "波动聚集与趋势切换常见；节假日与突发新闻会导致分布漂移；"
    "非平稳场景应优先近期窗口与轻量模型；高维特征在非平稳期易过拟合。"
)


def _ensure_prepared(symbol: str, *, quick: bool = False) -> tuple[Path, Path]:
    csv_path = PROCESSED / f"{symbol}_daily.csv"
    event_path = PROCESSED / f"{symbol}_event_text.txt"
    if csv_path.is_file() and event_path.is_file() and not quick:
        return csv_path, event_path
    prepare = FNspid_DIR / "prepare_demo.py"
    cmd = [sys.executable, str(prepare), "--symbol", symbol]
    if quick:
        cmd.append("--quick")
    print(f">>> 运行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return csv_path, event_path


def main() -> None:
    parser = argparse.ArgumentParser(description="FNSPID + prior_enhance demo")
    parser.add_argument("--symbol", default="AAL")
    parser.add_argument("--with-search", action="store_true")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--forecast-days", type=int, default=5)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="股价用 yfinance 补齐，不等 full_history.zip",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    csv_path, event_path = _ensure_prepared(symbol, quick=args.quick)
    event_text = event_path.read_text(encoding="utf-8")

    cfg = PriorEnhanceConfig.from_env()
    if not cfg.llm.api_key:
        print("错误: 未配置 DASHSCOPE_API_KEY")
        sys.exit(1)

    cfg.domain_knowledge = FIN_DOMAIN
    cfg.data.freq = "D"
    cfg.window.forecast_days = args.forecast_days
    cfg.window.points_per_day = 1
    cfg.search.val_horizon = min(20, max(5, args.forecast_days * 2))
    if not args.with_search:
        cfg.search.explore_ratio = 0.0

    if args.max_trials is not None:
        cfg.search.max_trials = args.max_trials

    pipe = PriorEnhancePipeline(cfg)

    import pandas as pd

    clean_df = pd.read_csv(csv_path)
    news_df = None
    news_events_path = PROCESSED / f"{symbol}_news_events.json"
    if news_events_path.is_file():
        events = json.loads(news_events_path.read_text(encoding="utf-8"))
        if events:
            news_df = pd.DataFrame([
                {
                    "news_time": pd.to_datetime(e["time"]).tz_localize(None),
                    "title": e.get("title", ""),
                    "summary": e.get("summary", ""),
                }
                for e in events
            ])

    windows = build_windows(clean_df, config=cfg)
    task = build_forecast_task(
        windows[0],
        config=cfg,
        event_text=event_text,
        news_df=news_df,
    )
    prior_set = pipe.generate_priors(task)

    tag = HOLDOUT_TAG
    win_dir = os.path.join(cfg.output_dir, "results", f"fnspid_{symbol.lower()}_{tag}")
    os.makedirs(win_dir, exist_ok=True)

    task_path = os.path.join(win_dir, "task_Tm.json")
    prior_path = os.path.join(win_dir, "priors_Zm.json")
    with open(task_path, "w", encoding="utf-8") as f:
        f.write(task.model_dump_json(indent=2, ensure_ascii=False))
    with open(prior_path, "w", encoding="utf-8") as f:
        f.write(prior_set.model_dump_json(indent=2, ensure_ascii=False))

    summary = {
        "dataset": "FNSPID",
        "symbol": symbol,
        "csv_path": str(csv_path),
        "event_text_path": str(event_path),
        "nu_m": task.context_state.nu_m,
        "n_m": task.context_state.n_m,
        "d_m": task.context_state.d_m,
        "xi_m": task.context_state.xi_m,
        "s_max": task.context_state.s_max,
        "event_text_preview": event_text[:400],
        "n_priors_valid": len(prior_set.valid_priors()),
        "task_path": task_path,
        "prior_path": prior_path,
    }

    if args.with_search:
        search_result, obs_result = pipe.run_search(task, prior_set)
        search_path = os.path.join(win_dir, "search_result.json")
        with open(search_path, "w", encoding="utf-8") as f:
            f.write(search_result.model_dump_json(indent=2, ensure_ascii=False))
        summary.update({
            "with_search": True,
            "best_recipe_id": search_result.best_recipe_id,
            "best_j_val": search_result.best_j_val,
            "n_evaluated": search_result.n_evaluated,
        })
        if obs_result:
            obs_path = os.path.join(win_dir, "observation_result.json")
            with open(obs_path, "w", encoding="utf-8") as f:
                f.write(obs_result.model_dump_json(indent=2, ensure_ascii=False))
            summary["best_j_obs"] = obs_result.best_j_obs
            summary["lambda_m"] = obs_result.lambda_m

    print("\n=== FNSPID prior_enhance demo ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n事件新颖性 ν_m={summary['nu_m']}（有真实新闻时通常高于纯日历 E_m）")


if __name__ == "__main__":
    main()
