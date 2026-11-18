"""Recipe → AutoApex serve /api/train payload。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from prior_enhance.recipe.catalog import ENSEMBLE_TYPES, OUTPUT_MODES
from prior_enhance.recipe.schema import ForecastingRecipe
from prior_enhance.telemetry.run_logger import log

PHASE = "RECIPE"


def apply_sample_window(history_df: pd.DataFrame, recipe: ForecastingRecipe) -> pd.DataFrame:
    """按 W 策略截断历史（训练前预处理）。"""
    ratio = recipe.W.recent_ratio
    if ratio >= 0.999:
        log.log_transform("apply_W", {"rows": len(history_df)}, {"rows": len(history_df), "strategy": "full"}, phase=PHASE)
        return history_df
    n = len(history_df)
    keep = max(recipe.W.min_history_points, int(n * ratio))
    out = history_df.iloc[-keep:].copy()
    log.log_transform(
        "apply_W",
        before={"rows": n},
        after={"rows": len(out), "strategy": recipe.W.strategy, "recent_ratio": ratio},
        phase=PHASE,
    )
    return out


def recipe_to_hyperparameters_payload(recipe: ForecastingRecipe) -> List[Dict[str, Any]]:
    """
    转为 serve /api/train 的 hyperparameters JSON 列表。

    格式: [{"name": "DirectTabular", "params": {...}}]
    """
    entry: Dict[str, Any] = {"name": recipe.M.name}
    if recipe.theta.params:
        entry["params"] = recipe.theta.params
    else:
        entry["params"] = {}
    return [entry]


def recipe_to_train_payload(
    recipe: ForecastingRecipe,
    task_id: str,
    target: str = "value",
    presets: Optional[str] = None,
) -> Dict[str, Any]:
    """
    构造 /api/train multipart form data（不含 file 二进制）。

    与 serve/api/routers/training.py 字段对齐。
    """
    out_cfg = OUTPUT_MODES.get(recipe.Q.output_mode, OUTPUT_MODES["point"])
    use_quantile = out_cfg.get("use_quantile")

    payload: Dict[str, Any] = {
        "task_id": task_id,
        "target": target,
        "prediction_length": recipe.prediction_length,
        "freq": (recipe.freq or "").strip() or "H",
        "eval_metric": recipe.eval_metric or "MAPE",
        "use_quantile": use_quantile if use_quantile else "",
        "use_tsfresh": recipe.F.use_tsfresh,
        "time_limit": recipe.time_limit,
        "enable_ensemble": recipe.Q.enable_ensemble,
        "hyperparameters": json.dumps(recipe_to_hyperparameters_payload(recipe)),
        "ensemble_types": json.dumps(recipe.Q.ensemble_types or []),
        "enable_shap": False,
    }

    if recipe.hyperparameter_tune_kwargs:
        payload["hyperparameter_tune_kwargs"] = json.dumps(recipe.hyperparameter_tune_kwargs)
    if recipe.V.enabled and recipe.V.num_val_windows is not None:
        payload["num_val_windows"] = recipe.V.num_val_windows
    if presets:
        payload["presets"] = presets

    log.log_output("train_payload", {
        "task_id": task_id,
        "model": recipe.M.name,
        "W": recipe.W.strategy,
        "F": recipe.F.mode,
        "Q": recipe.Q.output_mode,
        "ensemble": recipe.Q.ensemble_types,
        "hyperparameters": recipe_to_hyperparameters_payload(recipe),
    }, phase=PHASE)
    return payload


def infer_best_model_name(recipe: ForecastingRecipe) -> str:
    """推理时 best_model 字段建议值。"""
    if recipe.Q.enable_ensemble and recipe.Q.ensemble_types:
        key = recipe.Q.ensemble_types[0]
        return ENSEMBLE_TYPES.get(key, "WeightedEnsemble")
    return recipe.M.name
