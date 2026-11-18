"""prior_enhance 端到端编排（含透明日志）。"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import List, Optional

from prior_enhance.config import PriorEnhanceConfig
from prior_enhance.data.preprocessor import preprocess_csv
from prior_enhance.data.window import (
    HOLDOUT_TAG,
    build_forecast_task,
    build_windows,
    window_artifact_tag,
)
from prior_enhance.telemetry.run_logger import RunLogger, bind_run_logger, get_run_logger, reset_run_logger
from prior_enhance.prior.generator import generate_priors_for_task
from prior_enhance.prior.llm_client import DashScopeLLMClient
from prior_enhance.observation.feedback import run_observation_feedback
from prior_enhance.observation.memory import PriorMemory
from prior_enhance.observation.prior_store import PriorStore
from prior_enhance.prior.llm_trust import resolve_lambda_m_init
from prior_enhance.schemas import ForecastTask, ObservationResult, PriorSet, SearchResult
from prior_enhance.search.loop import run_prior_guided_search


class PriorEnhancePipeline:
    """
    闭环入口：数据 → 窗口 T_m → S_m → Z_m。

    每次 run_single_window 会生成:
      - outputs/logs/{run_id}.log
      - outputs/logs/{run_id}_audit.jsonl
    """

    def __init__(self, config: Optional[PriorEnhanceConfig] = None):
        self.config = config or PriorEnhanceConfig.from_env()
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._llm: Optional[DashScopeLLMClient] = None
        self._run_logger: Optional[RunLogger] = None

    @property
    def run_logger(self) -> Optional[RunLogger]:
        return self._run_logger or get_run_logger()

    @property
    def llm(self) -> DashScopeLLMClient:
        if self._llm is None:
            self._llm = DashScopeLLMClient(self.config.llm)
        return self._llm

    def _init_run_logger(self) -> RunLogger:
        lc = self.config.logging
        log_dir = lc.log_dir or os.path.join(self.config.output_dir, "logs")
        level = getattr(logging, lc.level.upper(), logging.INFO)
        logger = RunLogger(
            log_dir=log_dir,
            console=lc.console,
            level=level,
        )
        if lc.enabled:
            logger.log_config({
                "output_dir": self.config.output_dir,
                "data": asdict(self.config.data),
                "window": asdict(self.config.window),
                "llm": {
                    **asdict(self.config.llm),
                    "api_key": "***REDACTED***",
                },
                "prior": asdict(self.config.prior),
                "text_encoder": asdict(self.config.text_encoder),
                "search": asdict(self.config.search),
                "logging": asdict(lc),
            })
        return logger

    def preprocess(
        self,
        csv_path: str,
        target_col: Optional[str] = None,
    ) -> str:
        """清洗数据并保存到 outputs/preprocessed/。"""
        rl = self.run_logger
        phase = "PREPROCESS"

        if rl:
            rl.log_input("raw_csv", {"path": csv_path, "target_col": target_col}, phase=phase)

        out_dir = os.path.join(self.config.output_dir, "preprocessed")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(out_dir, f"{base}_clean.csv")

        with rl.step("preprocess_csv", phase=phase, csv_path=csv_path) if rl else _null_ctx():
            clean_df, tsdf = preprocess_csv(
                csv_path,
                cfg=self.config.data,
                target_col=target_col,
                output_path=out_path,
            )

        if rl:
            fill_flags = clean_df["fill_flag"].value_counts().to_dict() if "fill_flag" in clean_df.columns else {}
            rl.log_transform(
                "csv → clean_df",
                before={"rows": len(clean_df), "path": csv_path},
                after={
                    "rows": len(clean_df),
                    "columns": list(clean_df.columns),
                    "fill_flag_counts": fill_flags,
                    "tsdf_loaded": tsdf is not None,
                },
                phase=phase,
            )
            rl.log_artifact("clean_csv", out_path, phase=phase)

        return out_path

    def build_tasks(
        self,
        clean_csv_path: str,
    ) -> List[ForecastTask]:
        """从清洗 CSV 构建 T_m（默认 holdout 单窗口）。"""
        import pandas as pd

        rl = self.run_logger
        phase = "WINDOW"
        horizon = self.config.window.effective_horizon()

        clean_df = pd.read_csv(clean_csv_path)
        if rl:
            rl.log_input("clean_csv", {"path": clean_csv_path, "rows": len(clean_df)}, phase=phase)

        with rl.step("build_windows", phase=phase, mode=self.config.window.mode, horizon=horizon) if rl else _null_ctx():
            windows = build_windows(clean_df, config=self.config)

        if not windows:
            raise ValueError("无法构造预测窗口，请检查数据长度与 horizon 设置")

        w = windows[0]
        if rl:
            rl.log_output("window", {
                "mode": self.config.window.mode,
                "horizon": horizon,
                "H_m_rows": len(w.history_df),
                "O_m_rows": len(w.oos_df),
                "decision_time": str(w.decision_time),
            }, phase=phase)

        tag = window_artifact_tag(w, self.config.window.mode)
        with rl.step(f"build_Tm_{tag}", phase=phase) if rl else _null_ctx():
            task = build_forecast_task(w, config=self.config)
            if rl:
                rl.log_output("T_m", {
                    "decision_time": str(task.decision_time),
                    "history_rows": task.history_rows,
                    "oos_rows": task.oos_rows,
                    "episode_count": task.episode_count,
                    "retrieved_episode_ids": [e.episode_id for e in task.retrieved_episodes],
                    "S_m": task.context_state.model_dump(),
                    "stats_summary": {
                        "freq_code": task.stats.freq_code,
                        "mean": task.stats.mean,
                        "std": task.stats.std,
                        "periodicity": task.stats.periodicity,
                        "sub_periodicity": task.context_state.analysis.sub_periodicity,
                        "trend": task.stats.trend,
                        "persistence": task.context_state.analysis.persistence,
                        "volatility_cluster": task.context_state.analysis.volatility_cluster,
                    },
                }, phase=phase)
                rl.log_artifact("H_m", task.history_path, phase=phase)
                rl.log_artifact("O_m", task.oos_path or "", phase=phase)

        return [task]

    def generate_priors(self, task: ForecastTask) -> PriorSet:
        """对单个 T_m 调用大模型生成 Z_m。"""
        rl = self.run_logger
        phase = "PRIOR"

        if rl:
            rl.log_input("T_m", {
                "window_index": task.window_index,
                "decision_time": str(task.decision_time),
                "S_m": task.context_state.model_dump(),
            }, phase=phase)

        with rl.step("generate_Zm", phase=phase, model=self.config.llm.model) if rl else _null_ctx():
            prior_set = generate_priors_for_task(
                task,
                llm_client=self.llm,
                llm_config=self.config.llm,
                prior_config=self.config.prior,
            )

        if rl:
            valid = prior_set.valid_priors()
            rl.log_output("Z_m", {
                "n_total": len(prior_set.priors),
                "n_valid": len(valid),
                "prior_ids": [p.prior_id for p in prior_set.priors],
                "valid_ids": [p.prior_id for p in valid],
                "rejected": [
                    {"id": p.prior_id, "reason": p.reject_reason}
                    for p in prior_set.priors if not p.valid
                ],
                "source": "llm",
            }, phase=phase)
            if prior_set.prompt_snapshot:
                tag = HOLDOUT_TAG if self.config.window.mode == "holdout" else f"win{task.window_index:04d}"
                prompt_path = os.path.join(
                    self.config.output_dir, "logs",
                    f"{rl.run_id}_prompt_{tag}.txt",
                )
                os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(prior_set.prompt_snapshot)
                rl.log_artifact("llm_prompt", prompt_path, phase=phase)
            if prior_set.raw_llm_response:
                tag = HOLDOUT_TAG if self.config.window.mode == "holdout" else f"win{task.window_index:04d}"
                resp_path = os.path.join(
                    self.config.output_dir, "logs",
                    f"{rl.run_id}_llm_response_{tag}.txt",
                )
                with open(resp_path, "w", encoding="utf-8") as f:
                    f.write(prior_set.raw_llm_response)
                rl.log_artifact("llm_response", resp_path, phase=phase)

        return prior_set

    def _load_prior_store(self) -> PriorStore:
        return PriorStore.load(self.config.default_prior_store_path())

    def _initial_memory(self) -> PriorMemory:
        scfg = self.config.search
        lam, lam_meta = resolve_lambda_m_init(
            configured=scfg.lambda_m,
            model_name=self.config.llm.model,
        )
        if self.run_logger:
            self.run_logger.log_output("lambda_m_init", lam_meta, phase="PIPELINE")
        return PriorMemory(lambda_m=lam)

    def run_search(
        self,
        task: ForecastTask,
        prior_set: PriorSet,
        *,
        memory: Optional[PriorMemory] = None,
        prior_store: Optional[PriorStore] = None,
    ) -> tuple[SearchResult, Optional[ObservationResult]]:
        """Phase 2：窗口内 search +（若有 O_m）observation feedback。"""
        import pandas as pd

        rl = self.run_logger
        scfg = self.config.search
        dcfg = self.config.data

        history_df = pd.read_csv(task.history_path)
        oos_df = pd.read_csv(task.oos_path) if task.oos_path and os.path.isfile(task.oos_path) else None
        memory = memory or self._initial_memory()
        prior_store = prior_store or self._load_prior_store()

        with rl.step("prior_guided_search", phase="SEARCH", max_trials=scfg.max_trials) if rl else _null_ctx():
            search_result = run_prior_guided_search(
                task, prior_set, history_df,
                cfg=scfg, dcfg=dcfg,
                apex_cfg=self.config.autoapex,
                prior_store=prior_store,
                lambda_m=memory.lambda_m,
            )

        obs_result: Optional[ObservationResult] = None
        if oos_df is not None and len(oos_df) > 0:
            with rl.step("observation_feedback", phase="OBSERVE") if rl else _null_ctx():
                obs_result = run_observation_feedback(
                    search_result,
                    prior_set,
                    history_df,
                    oos_df,
                    task.context_state,
                    memory=memory,
                    cfg=scfg,
                    dcfg=dcfg,
                    apex_cfg=self.config.autoapex,
                    prior_store=prior_store,
                )
                if rl:
                    rl.log_output("memory_update", {
                        "tau": obs_result.memory_tau,
                        "lambda_m": obs_result.lambda_m,
                    }, phase="OBSERVE")

        return search_result, obs_result

    def run_rolling_windows(
        self,
        csv_path: str,
        target_col: Optional[str] = None,
        *,
        max_windows: Optional[int] = None,
        with_search: bool = True,
    ) -> List[dict]:
        """
        滚动多窗口 walk-forward：每个窗口 m 独立执行 Phase 1–4，
        持久记忆 τ 与 λ_m 跨窗口传递。
        """
        if self.config.window.mode != "rolling":
            raise ValueError("run_rolling_windows 需要 config.window.mode='rolling'")

        import pandas as pd

        token = None
        if self.config.logging.enabled:
            self._run_logger = self._init_run_logger()
            token = bind_run_logger(self._run_logger)

        memory = self._initial_memory()
        prior_store = self._load_prior_store()
        summaries: List[dict] = []

        try:
            clean_path = self.preprocess(csv_path, target_col=target_col)
            clean_df = pd.read_csv(clean_path)
            windows = build_windows(clean_df, config=self.config)
            if max_windows is not None:
                windows = windows[-max_windows:]

            for w in windows:
                tag = window_artifact_tag(w, "rolling")
                task = build_forecast_task(w, config=self.config)
                prior_set = self.generate_priors(task)

                win_dir = os.path.join(self.config.output_dir, "results", tag)
                os.makedirs(win_dir, exist_ok=True)
                with open(os.path.join(win_dir, "task_Tm.json"), "w", encoding="utf-8") as f:
                    f.write(task.model_dump_json(indent=2, ensure_ascii=False))
                with open(os.path.join(win_dir, "priors_Zm.json"), "w", encoding="utf-8") as f:
                    f.write(prior_set.model_dump_json(indent=2, ensure_ascii=False))

                search_result = None
                obs_result = None
                if with_search:
                    search_result, obs_result = self.run_search(
                        task, prior_set, memory=memory, prior_store=prior_store,
                    )
                    with open(os.path.join(win_dir, "search_result.json"), "w", encoding="utf-8") as f:
                        f.write(search_result.model_dump_json(indent=2, ensure_ascii=False))
                    if obs_result:
                        with open(os.path.join(win_dir, "observation_result.json"), "w", encoding="utf-8") as f:
                            f.write(obs_result.model_dump_json(indent=2, ensure_ascii=False))
                        memory = PriorMemory(tau=obs_result.memory_tau, lambda_m=obs_result.lambda_m)

                summaries.append({
                    "window_index": task.window_index,
                    "decision_time": str(task.decision_time),
                    "history_rows": task.history_rows,
                    "oos_rows": task.oos_rows,
                    "n_priors_valid": len(prior_set.valid_priors()),
                    "best_recipe_id": search_result.best_recipe_id if search_result else None,
                    "best_j_val": search_result.best_j_val if search_result else None,
                    "best_j_obs": obs_result.best_j_obs if obs_result else None,
                    "lambda_m": memory.lambda_m,
                    "result_dir": win_dir,
                })

            roll_path = os.path.join(self.config.output_dir, "results", "rolling_summary.json")
            with open(roll_path, "w", encoding="utf-8") as f:
                json.dump(summaries, f, ensure_ascii=False, indent=2)
            return summaries
        finally:
            if token is not None:
                reset_run_logger(token)
                self._run_logger = None

    def run_single_window(
        self,
        csv_path: str,
        target_col: Optional[str] = None,
        *,
        with_search: bool = False,
        memory: Optional[PriorMemory] = None,
    ) -> dict:
        """
        单窗口完整流程：preprocess → T_m → Z_m → 落盘 JSON + 完整日志。

        holdout 模式：全量数据末尾留 H 点作 O_m，其余为 H_m。
        返回 summary，含 log_path / audit_path。
        """
        token = None
        if self.config.logging.enabled:
            self._run_logger = self._init_run_logger()
            token = bind_run_logger(self._run_logger)

        rl = self._run_logger
        summary: dict = {}
        horizon = self.config.window.effective_horizon()

        try:
            if rl:
                rl.log_input("run_params", {
                    "csv_path": csv_path,
                    "target_col": target_col,
                    "mode": self.config.window.mode,
                    "horizon": horizon,
                    "forecast_days": self.config.window.forecast_days,
                    "points_per_day": self.config.window.points_per_day,
                    "llm_model": self.config.llm.model,
                }, phase="PIPELINE")

            clean_path = self.preprocess(csv_path, target_col=target_col)
            tasks = self.build_tasks(clean_path)
            task = tasks[0]
            prior_set = self.generate_priors(task)

            tag = HOLDOUT_TAG if self.config.window.mode == "holdout" else f"win_{task.window_index:04d}"
            win_dir = os.path.join(self.config.output_dir, "results", tag)
            os.makedirs(win_dir, exist_ok=True)

            task_path = os.path.join(win_dir, "task_Tm.json")
            prior_path = os.path.join(win_dir, "priors_Zm.json")
            search_path = os.path.join(win_dir, "search_result.json")
            obs_path = os.path.join(win_dir, "observation_result.json")

            with open(task_path, "w", encoding="utf-8") as f:
                f.write(task.model_dump_json(indent=2, ensure_ascii=False))
            with open(prior_path, "w", encoding="utf-8") as f:
                f.write(prior_set.model_dump_json(indent=2, ensure_ascii=False))

            search_result = None
            obs_result = None
            if with_search:
                search_result, obs_result = self.run_search(task, prior_set, memory=memory)
                with open(search_path, "w", encoding="utf-8") as f:
                    f.write(search_result.model_dump_json(indent=2, ensure_ascii=False))
                if obs_result:
                    with open(obs_path, "w", encoding="utf-8") as f:
                        f.write(obs_result.model_dump_json(indent=2, ensure_ascii=False))

            summary = {
                "mode": self.config.window.mode,
                "horizon": horizon,
                "window_index": task.window_index,
                "decision_time": str(task.decision_time),
                "history_rows": task.history_rows,
                "oos_rows": task.oos_rows,
                "context_state": task.context_state.model_dump(),
                "episode_count": task.episode_count,
                "n_priors_total": len(prior_set.priors),
                "n_priors_valid": len(prior_set.valid_priors()),
                "task_path": task_path,
                "prior_path": prior_path,
                "history_path": task.history_path,
                "oos_path": task.oos_path,
                "clean_csv_path": clean_path,
            }
            if search_result:
                summary.update({
                    "with_search": True,
                    "best_recipe_id": search_result.best_recipe_id,
                    "best_j_val": search_result.best_j_val,
                    "n_evaluated": search_result.n_evaluated,
                    "search_path": search_path,
                })
            if obs_result:
                summary.update({
                    "best_j_obs": obs_result.best_j_obs,
                    "lambda_m": obs_result.lambda_m,
                    "memory_tau": obs_result.memory_tau,
                    "observation_path": obs_path,
                })

            summary_path = os.path.join(win_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            if rl:
                rl.log_artifact("task_Tm.json", task_path, phase="PIPELINE")
                rl.log_artifact("priors_Zm.json", prior_path, phase="PIPELINE")
                rl.log_artifact("summary.json", summary_path, phase="PIPELINE")
                log_meta = rl.finish(summary)
                summary.update(log_meta)

            return summary

        finally:
            if token is not None:
                reset_run_logger(token)
                self._run_logger = None


class _null_ctx:
    """无 RunLogger 时的空 context manager。"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
