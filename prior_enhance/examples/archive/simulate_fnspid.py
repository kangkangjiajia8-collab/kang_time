#!/usr/bin/env python3
"""
FNSPID 快速模拟（无需等 full_history.zip，无需 LLM）。

已归档：主入口请用 prepare_demo.py + run_fnspid_demo.py。
保留供离线调试窗口切分 / S_m / E_m。

用法:
    pip install yfinance   # 若未安装
    python prior_enhance/examples/archive/simulate_fnspid.py --symbol AAL
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prior_enhance._bootstrap import bootstrap  # noqa: E402

bootstrap(ROOT)

from prior_enhance._path import setup_autoapex_path  # noqa: E402

setup_autoapex_path()

import pandas as pd  # noqa: E402

from prior_enhance.config import PriorEnhanceConfig  # noqa: E402
from prior_enhance.data.window import build_forecast_task, build_windows  # noqa: E402

FNspid_DIR = ROOT / "prior_enhance" / "data" / "fnspid"
PROCESSED = FNspid_DIR / "processed"
FIN_DOMAIN = (
    "美股日频收盘价预测：受财报、分析师评级、宏观新闻与行业情绪影响；"
    "非平稳场景应优先近期窗口与轻量模型。"
)


def _ensure_quick(symbol: str) -> Path:
    csv_path = PROCESSED / f"{symbol}_daily.csv"
    if csv_path.is_file():
        return csv_path
    prepare = FNspid_DIR / "prepare_demo.py"
    subprocess.run(
        [sys.executable, str(prepare), "--symbol", symbol, "--quick"],
        check=True,
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="FNSPID 快速模拟（无 LLM，归档脚本）")
    parser.add_argument("--symbol", default="AAL")
    parser.add_argument("--forecast-days", type=int, default=5)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    csv_path = _ensure_quick(symbol)
    event_path = PROCESSED / f"{symbol}_event_text.txt"
    event_text = event_path.read_text(encoding="utf-8") if event_path.is_file() else ""

    cfg = PriorEnhanceConfig.from_env()
    cfg.domain_knowledge = FIN_DOMAIN
    cfg.data.freq = "D"
    cfg.window.forecast_days = args.forecast_days
    cfg.window.points_per_day = 1

    df = pd.read_csv(csv_path)
    windows = build_windows(df, config=cfg)
    w = windows[0]
    task = build_forecast_task(windows[0], config=cfg, event_text=event_text)
    sm = task.context_state

    report = {
        "mode": "quick_simulate",
        "symbol": symbol,
        "price_csv": str(csv_path),
        "price_rows": len(df),
        "horizon": cfg.window.effective_horizon(),
        "H_m_rows": len(w.history_df),
        "O_m_rows": len(w.oos_df),
        "decision_time": str(task.decision_time),
        "S_m": {
            "n_m": sm.n_m,
            "nu_m": sm.nu_m,
            "d_m": sm.d_m,
            "s_max": sm.s_max,
            "cond_labels": sm.cond_labels,
            "scenario": sm.scenario,
        },
        "episode_count": task.episode_count,
        "retrieved_episode_ids": [e.episode_id for e in task.retrieved_episodes[:5]],
        "event_text_preview": event_text[:600],
        "next_steps": [
            "完整 LLM 先验: python prior_enhance/examples/run_fnspid_demo.py --symbol "
            f"{symbol} --quick",
            "含搜索: 加 --with-search --max-trials 4",
            "zip 下完后: prepare_demo.py 不加 --quick，换 FNSPID 官方股价",
        ],
    }

    out = PROCESSED / f"{symbol}_simulate_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== FNSPID 快速模拟（新闻=FNSPID，股价=Stooq/yfinance）===\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()
