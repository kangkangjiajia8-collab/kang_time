#!/usr/bin/env python3
"""
FNSPID 论文四阶段端到端验证（可用 proxy 评估，无需 AutoApex serve）。

用法:
    # 使用已有 priors_Zm.json（跳过 LLM）
    PRIOR_ENHANCE_EVAL_BACKEND=proxy python prior_enhance/examples/validate_fnspid_paper.py --symbol AAL

    # 含 LLM 先验生成
    python prior_enhance/examples/validate_fnspid_paper.py --symbol AAL --with-llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prior_enhance._bootstrap import bootstrap  # noqa: E402

bootstrap(ROOT)

from prior_enhance._path import setup_autoapex_path  # noqa: E402

setup_autoapex_path()

import pandas as pd  # noqa: E402

from prior_enhance.config import PriorEnhanceConfig  # noqa: E402
from prior_enhance.data.window import HOLDOUT_TAG, build_forecast_task, build_windows  # noqa: E402
from prior_enhance.pipeline import PriorEnhancePipeline  # noqa: E402
from prior_enhance.schemas import PriorSet  # noqa: E402
from prior_enhance.validation.evaluator import evaluate_on_oos  # noqa: E402

FNspid_DIR = ROOT / "prior_enhance" / "data" / "fnspid"
PROCESSED = FNspid_DIR / "processed"
FIN_DOMAIN = (
    "美股日频收盘价预测：受财报、分析师评级、宏观新闻与行业情绪影响；"
    "波动聚集与趋势切换常见；节假日与突发新闻会导致分布漂移；"
    "非平稳场景应优先近期窗口与轻量模型；高维特征在非平稳期易过拟合。"
)


def _load_news_df(symbol: str) -> pd.DataFrame | None:
    path = PROCESSED / f"{symbol}_news_events.json"
    if not path.is_file():
        news_csv = FNspid_DIR / "raw" / f"news_{symbol}.csv"
        if not news_csv.is_file():
            return None
        raw = pd.read_csv(news_csv, low_memory=False)
        sym_col = "Stock_symbol" if "Stock_symbol" in raw.columns else "Stock_Symbol"
        raw = raw[raw[sym_col].astype(str).str.upper() == symbol.upper()].copy()
        raw["news_time"] = pd.to_datetime(raw["Date"], utc=True, errors="coerce").dt.tz_localize(None)
        raw["title"] = raw.get("Article_title", raw.get("Title", "")).astype(str)
        for col in ("Textrank_summary", "Lsa_summary", "Article"):
            if col in raw.columns:
                raw["summary"] = raw[col].fillna("").astype(str)
                break
        else:
            raw["summary"] = ""
        return raw.dropna(subset=["news_time"])

    events = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for e in events:
        rows.append({
            "news_time": pd.to_datetime(e["time"]).tz_localize(None),
            "title": e.get("title", ""),
            "summary": e.get("summary", ""),
        })
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(description="FNSPID 论文四阶段验证")
    parser.add_argument("--symbol", default="AAL")
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--max-trials", type=int, default=8)
    parser.add_argument("--forecast-days", type=int, default=5)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    os.environ.setdefault("PRIOR_ENHANCE_EVAL_BACKEND", "proxy")
    os.environ.setdefault("PRIOR_ENHANCE_SKIP_AUTOAPEX_STATS", "1")

    csv_path = PROCESSED / f"{symbol}_daily.csv"
    event_path = PROCESSED / f"{symbol}_event_text.txt"
    if not csv_path.is_file() or not event_path.is_file():
        raise FileNotFoundError(f"请先运行 build_dataset.py --symbol {symbol}")

    cfg = PriorEnhanceConfig.from_env()
    cfg.domain_knowledge = FIN_DOMAIN
    cfg.data.freq = "D"
    cfg.window.forecast_days = args.forecast_days
    cfg.window.points_per_day = 1
    cfg.search.max_trials = args.max_trials
    cfg.search.val_horizon = min(20, max(5, args.forecast_days * 2))
    cfg.search.explore_ratio = 0.15

    pipe = PriorEnhancePipeline(cfg)
    clean_df = pd.read_csv(csv_path)
    news_df = _load_news_df(symbol)
    event_text = event_path.read_text(encoding="utf-8")
    windows = build_windows(clean_df, config=cfg)
    task = build_forecast_task(
        windows[0], config=cfg, event_text=event_text, news_df=news_df,
    )

    win_dir = os.path.join(cfg.output_dir, "results", f"fnspid_{symbol.lower()}_{HOLDOUT_TAG}")
    os.makedirs(win_dir, exist_ok=True)
    prior_path = os.path.join(win_dir, "priors_Zm.json")
    task_path = os.path.join(win_dir, "task_Tm.json")

    audit = {
        "symbol": symbol,
        "phase1": {},
        "phase2": {},
        "phase3": {},
        "phase4": {},
    }

    # Phase 1
    sm = task.context_state
    audit["phase1"] = {
        "implemented": True,
        "T_m": str(task.decision_time),
        "H_m": task.history_rows,
        "O_m": task.oos_rows,
        "S_m": {
            "n_m": sm.n_m, "nu_m": sm.nu_m, "d_m": sm.d_m, "xi_m": sm.xi_m,
            "cond": sm.cond_labels,
        },
        "D_m_count": task.episode_count,
        "news_in_episodes": news_df is not None,
    }

    with open(task_path, "w", encoding="utf-8") as f:
        f.write(task.model_dump_json(indent=2, ensure_ascii=False))

    if args.with_llm:
        if not cfg.llm.api_key:
            print("错误: --with-llm 需要 DASHSCOPE_API_KEY")
            sys.exit(1)
        prior_set = pipe.generate_priors(task)
    elif Path(prior_path).is_file():
        prior_set = PriorSet.model_validate_json(Path(prior_path).read_text(encoding="utf-8"))
        audit["phase1"]["priors_source"] = "cached"
    else:
        print("无缓存 priors 且未 --with-llm，退出")
        sys.exit(1)

    audit["phase1"]["n_priors_valid"] = len(prior_set.valid_priors())
    with open(prior_path, "w", encoding="utf-8") as f:
        f.write(prior_set.model_dump_json(indent=2, ensure_ascii=False))

    # Phase 2-4
    search_result, obs_result = pipe.run_search(task, prior_set)
    search_path = os.path.join(win_dir, "search_result.json")
    with open(search_path, "w", encoding="utf-8") as f:
        f.write(search_result.model_dump_json(indent=2, ensure_ascii=False))

    audit["phase2"] = {
        "implemented": True,
        "best_recipe_id": search_result.best_recipe_id,
        "best_j_val": search_result.best_j_val,
        "n_evaluated": search_result.n_evaluated,
        "explore_ratio": cfg.search.explore_ratio,
        "retain_ids": search_result.retain_ids,
    }

    # Phase 3: r* 在 H_m 上训练并预测 O_m
    history_df = pd.read_csv(task.history_path)
    oos_df = pd.read_csv(task.oos_path) if task.oos_path else None
    best_recipe = getattr(search_result, "_best_recipe", None)
    forecast = None
    if best_recipe and oos_df is not None and len(oos_df) > 0:
        pred, metrics = evaluate_on_oos(
            best_recipe, history_df, oos_df, task.context_state, cfg.data, apex_cfg=cfg.autoapex,
        )
        forecast = {
            "recipe_id": best_recipe.recipe_id,
            "pred": pred.tolist(),
            "actual": oos_df[cfg.data.target_col].astype(float).tolist(),
            "timestamps": oos_df[cfg.data.time_col].astype(str).tolist(),
            "L_obs": metrics.get("L_obs"),
            "J_obs": metrics.get("J_obs"),
        }
        forecast_path = os.path.join(win_dir, "forecast_r_star.json")
        with open(forecast_path, "w", encoding="utf-8") as f:
            json.dump(forecast, f, ensure_ascii=False, indent=2)

    audit["phase3"] = {
        "implemented": forecast is not None,
        "r_star": search_result.best_recipe_id,
        "forecast_saved": forecast is not None,
    }

    if obs_result:
        obs_path = os.path.join(win_dir, "observation_result.json")
        with open(obs_path, "w", encoding="utf-8") as f:
            f.write(obs_result.model_dump_json(indent=2, ensure_ascii=False))
        audit["phase4"] = {
            "implemented": True,
            "best_j_obs": obs_result.best_j_obs,
            "lambda_m": obs_result.lambda_m,
            "n_retain": obs_result.n_retain,
            "gain_accepted": sum(1 for g in obs_result.gain_summary if g.get("accepted")),
        }
    else:
        audit["phase4"] = {"implemented": False, "reason": "no O_m"}

    audit_path = os.path.join(win_dir, "paper_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print("=== FNSPID 论文四阶段验证 ===\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"\n审计报告: {audit_path}")


if __name__ == "__main__":
    main()
