#!/usr/bin/env python3
"""
prior_enhance 单窗口 holdout 演示（DashScope 大模型生成先验）。

全量 CSV 末尾留 H 点作测试 O_m，其余全部为历史 H_m。
默认：小时数据、预测 3 天 → H = 24 × 3 = 72。

用法:
    # mutiagent 环境（推荐）
    bash prior_enhance/examples/run_mutiagent.sh

    # 或手动
    conda activate mutiagent
    python prior_enhance/examples/run_demo.py

API Key 配置: prior_enhance/secrets.py（或环境变量 DASHSCOPE_API_KEY）
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

from prior_enhance.config import PriorEnhanceConfig, DEFAULT_DATA_CSV, LLMConfig, PriorConfig  # noqa: E402
from prior_enhance.pipeline import PriorEnhancePipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="prior_enhance holdout 单窗口 Z_m 生成（LLM）")
    parser.add_argument(
        "--csv",
        default=DEFAULT_DATA_CSV,
        help=f"原始时序 CSV 路径（默认: {DEFAULT_DATA_CSV}）",
    )
    parser.add_argument(
        "--clean-csv",
        default=None,
        help="已清洗 CSV，若指定则跳过 preprocess（加快调试）",
    )
    parser.add_argument(
        "--target-col",
        default="power",
        help="目标列名；代理购电数据默认为 power",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=3,
        help="预测未来天数（默认 3）",
    )
    parser.add_argument(
        "--points-per-day",
        type=int,
        default=24,
        help="每天时点数，小时数据为 24（默认 24）",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="直接指定预测跨度 H；若设置则覆盖 forecast-days × points-per-day",
    )
    parser.add_argument(
        "--freq",
        default=None,
        choices=["5T", "15T", "h", "D", "W-SUN", "M"],
        help="显式指定时间频率；默认从时间戳自动推断",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="DashScope 模型名，默认 deepseek-r1-distill-qwen-1.5b",
    )
    parser.add_argument(
        "--min-priors",
        type=int,
        default=None,
        help=f"最少有效 prior 条数（默认 {LLMConfig.min_priors if hasattr(LLMConfig, 'min_priors') else 10}）",
    )
    parser.add_argument(
        "--max-priors",
        type=int,
        default=None,
        help="最多 prior 条数（默认 15）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认 prior_enhance/outputs",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING"],
        help="日志级别，默认 INFO；也可 PRIOR_ENHANCE_LOG_LEVEL",
    )
    parser.add_argument(
        "--no-log-console",
        action="store_true",
        help="不在控制台打印日志，仅写文件",
    )
    parser.add_argument(
        "--with-search",
        action="store_true",
        help="Phase 2：窗口内 validation search + O_m observation feedback",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="搜索评估 recipe 数量上限（默认 12）",
    )
    args = parser.parse_args()

    cfg = PriorEnhanceConfig.from_env()
    if not cfg.llm.api_key:
        print("错误: 未配置 DASHSCOPE_API_KEY")
        print("请在 prior_enhance/secrets.py 中设置，或 export DASHSCOPE_API_KEY=sk-xxx")
        sys.exit(1)
    cfg.window.forecast_days = args.forecast_days
    cfg.window.points_per_day = args.points_per_day
    if args.horizon is not None:
        cfg.window.horizon = args.horizon
    if args.freq:
        cfg.data.freq = args.freq
    if args.model:
        cfg.llm.model = args.model
    if args.min_priors is not None:
        cfg.llm.min_priors = args.min_priors
    if args.max_priors is not None:
        cfg.llm.max_priors = args.max_priors
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.log_level:
        cfg.logging.level = args.log_level
    if args.no_log_console:
        cfg.logging.console = False
    if args.max_trials is not None:
        cfg.search.max_trials = args.max_trials

    horizon = cfg.window.effective_horizon()
    print(f"holdout 模式: H_m = 全量 - {horizon} 点, O_m = 末尾 {horizon} 点")
    print(f"LLM 模型: {cfg.llm.model}（DashScope 流式）")
    print(f"文本编码器: {cfg.text_encoder.model_name} ({cfg.text_encoder.device})")
    print(f"先验数量要求: {cfg.llm.min_priors}~{cfg.llm.max_priors} 条（cond/target/action/b_llm 四元组）")

    pipe = PriorEnhancePipeline(cfg)
    if args.clean_csv:
        import pandas as pd
        from prior_enhance.data.window import build_forecast_task, build_windows

        clean_df = pd.read_csv(args.clean_csv)
        windows = build_windows(clean_df, config=cfg)
        task = build_forecast_task(windows[0], config=cfg)
        prior_set = pipe.generate_priors(task)
        tag = "holdout"
        win_dir = os.path.join(cfg.output_dir, "results", tag)
        os.makedirs(win_dir, exist_ok=True)
        task_path = os.path.join(win_dir, "task_Tm.json")
        prior_path = os.path.join(win_dir, "priors_Zm.json")
        with open(task_path, "w", encoding="utf-8") as f:
            f.write(task.model_dump_json(indent=2, ensure_ascii=False))
        with open(prior_path, "w", encoding="utf-8") as f:
            f.write(prior_set.model_dump_json(indent=2, ensure_ascii=False))
        summary = {
            "mode": cfg.window.mode,
            "horizon": cfg.window.effective_horizon(),
            "episode_count": task.episode_count,
            "context_state": task.context_state.model_dump(),
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
                "search_path": search_path,
            })
            if obs_result:
                obs_path = os.path.join(win_dir, "observation_result.json")
                with open(obs_path, "w", encoding="utf-8") as f:
                    f.write(obs_result.model_dump_json(indent=2, ensure_ascii=False))
                summary.update({
                    "best_j_obs": obs_result.best_j_obs,
                    "lambda_m": obs_result.lambda_m,
                    "memory_tau": obs_result.memory_tau,
                    "observation_path": obs_path,
                })
    else:
        summary = pipe.run_single_window(
            csv_path=args.csv,
            target_col=args.target_col,
            with_search=args.with_search,
        )

    sm = summary.get("context_state", {})
    print(f"\nS_m: n_m={sm.get('n_m')}, ν_m={sm.get('nu_m')}, d_m={sm.get('d_m')}, s_max={sm.get('s_max')}")
    print(f"|D_m|={summary.get('episode_count', 'N/A')}")

    print("\n=== prior_enhance 运行完成 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if summary.get("log_path"):
        print(f"\n📋 人类可读日志: {summary['log_path']}")
        print(f"📋 审计 JSONL:   {summary['audit_path']}")

    prior_path = summary["prior_path"]
    with open(prior_path, encoding="utf-8") as f:
        prior_data = json.load(f)
    valid = [p for p in prior_data.get("priors", []) if p.get("valid", True)]
    print(f"\n有效 prior 数量: {len(valid)} / {len(prior_data.get('priors', []))}")
    for p in valid[:5]:
        print(f"  - [{p['prior_id']}] cond={p.get('cond')} b={p.get('confidence', p.get('b_llm', 0)):.2f}")
        action = p.get("action", [])
        if isinstance(action, list):
            print(f"    avoid: {', '.join(action[:3])}")
        else:
            print(f"    action: {str(action)[:80]}")

    if summary.get("with_search"):
        print(f"\n搜索: 评估 {summary.get('n_evaluated')} 个 recipe")
        print(f"  r* = {summary.get('best_recipe_id')}  J_val={summary.get('best_j_val'):.4f}")
        if summary.get("best_j_obs") is not None:
            print(f"  O_m 最优 J_obs={summary.get('best_j_obs'):.4f}  λ_m={summary.get('lambda_m'):.4f}")


if __name__ == "__main__":
    main()
