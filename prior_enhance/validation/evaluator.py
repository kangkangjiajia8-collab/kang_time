"""轻量 / AutoApex recipe 评估器（窗口内 validation / O_m 观测）。"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from prior_enhance.config import AutoApexConfig, DataConfig
from prior_enhance.recipe.autoapex_adapter import apply_sample_window
from prior_enhance.recipe.schema import ForecastingRecipe
from prior_enhance.schemas import ContextState
from prior_enhance.telemetry.run_logger import log
from prior_enhance.validation.objective import j_obs, j_val

PHASE = "EVAL"


def _calendar_features(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts)
    hour = ts.dt.hour.values.astype(float)
    dow = ts.dt.dayofweek.values.astype(float)
    return np.column_stack([hour, dow, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)])


def _seasonal_naive(train_y: np.ndarray, val_len: int, lag: int) -> np.ndarray:
    if len(train_y) <= lag:
        lag = min(24, max(1, len(train_y) - 1))
    if lag <= 0:
        return np.full(val_len, float(np.mean(train_y)))
    pattern = train_y[-lag:]
    reps = int(np.ceil(val_len / lag))
    return np.tile(pattern, reps)[:val_len]


def _linear_calendar_forecast(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: DataConfig,
) -> np.ndarray:
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        y = train_df[cfg.target_col].astype(float).values
        return _seasonal_naive(y, len(val_df), 168)

    x_train = _calendar_features(train_df[cfg.time_col])
    y_train = train_df[cfg.target_col].astype(float).values
    x_val = _calendar_features(val_df[cfg.time_col])
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    return np.maximum(pred, 0.0)


def _proxy_forecast(
    recipe: ForecastingRecipe,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: DataConfig,
) -> np.ndarray:
    """按模型族选择快速代理预测器（开发/回退用）。"""
    train_cut = apply_sample_window(train_df, recipe)
    y = train_cut[cfg.target_col].astype(float).values
    val_len = len(val_df)
    name = recipe.M.name

    if name in {"SeasonalNaive", "Naive", "Average"}:
        return _seasonal_naive(y, val_len, 168)
    if name in {"AutoARIMA", "ETS", "Theta", "ARIMA", "AutoETS"}:
        return _seasonal_naive(y, val_len, 168)
    if name in {"DirectTabular", "RecursiveTabular", "LightGBM", "XGBoost", "PerStepTabular"}:
        return _linear_calendar_forecast(train_cut, val_df, cfg)
    return _seasonal_naive(y, val_len, 24)


def _autoapex_forecast(
    recipe: ForecastingRecipe,
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cfg: DataConfig,
    apex_cfg: AutoApexConfig,
) -> Tuple[np.ndarray, float]:
    from prior_enhance.validation.autoapex_client import get_autoapex_client

    client = get_autoapex_client(apex_cfg)
    if not client.check_health():
        raise ConnectionError(
            f"AutoApex serve 不可达: {apex_cfg.base_url} "
            "(请确认 serve 已启动，或设置 PRIOR_ENHANCE_EVAL_BACKEND=proxy)"
        )
    pred_series, tr = client.train_and_predict(recipe, train_df, target_df, cfg)
    return pred_series.values.astype(float), tr.wall_time_sec


def _resolve_backend(
    backend: Optional[str],
    apex_cfg: Optional[AutoApexConfig],
) -> Tuple[str, AutoApexConfig]:
    apex_cfg = apex_cfg or AutoApexConfig()
    chosen = (backend or apex_cfg.eval_backend or "autoapex").lower()
    return chosen, apex_cfg


def evaluate_on_split(
    recipe: ForecastingRecipe,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ctx: ContextState,
    cfg: Optional[DataConfig] = None,
    *,
    lambda_c: float = 0.1,
    backend: Optional[str] = None,
    apex_cfg: Optional[AutoApexConfig] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    cfg = cfg or DataConfig()
    chosen, apex_cfg = _resolve_backend(backend, apex_cfg)
    wall_time = 0.0

    if chosen == "autoapex":
        try:
            pred, wall_time = _autoapex_forecast(recipe, train_df, val_df, cfg, apex_cfg)
            used = "autoapex"
        except Exception as exc:
            if not apex_cfg.fallback_proxy:
                raise
            log.log_output("autoapex_fallback", {"error": str(exc)[:200]}, phase=PHASE)
            pred = _proxy_forecast(recipe, train_df, val_df, cfg)
            used = "proxy_fallback"
    else:
        pred = _proxy_forecast(recipe, train_df, val_df, cfg)
        used = "proxy"

    y_true = val_df[cfg.target_col].astype(float).values
    metrics = j_val(recipe, y_true, pred, ctx, lambda_c=lambda_c, wall_time_sec=wall_time)
    metrics["eval_backend"] = used
    return pred, metrics


def evaluate_on_oos(
    recipe: ForecastingRecipe,
    history_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    ctx: ContextState,
    cfg: Optional[DataConfig] = None,
    *,
    lambda_c: float = 0.1,
    backend: Optional[str] = None,
    apex_cfg: Optional[AutoApexConfig] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """在 O_m 上评估：Train(r, H_m) 后预测 O_m。"""
    cfg = cfg or DataConfig()
    chosen, apex_cfg = _resolve_backend(backend, apex_cfg)
    wall_time = 0.0

    if chosen == "autoapex":
        try:
            pred, wall_time = _autoapex_forecast(recipe, history_df, oos_df, cfg, apex_cfg)
            used = "autoapex"
        except Exception as exc:
            if not apex_cfg.fallback_proxy:
                raise
            log.log_output("autoapex_fallback", {"error": str(exc)[:200]}, phase=PHASE)
            pred = _proxy_forecast(recipe, history_df, oos_df, cfg)
            used = "proxy_fallback"
    else:
        pred = _proxy_forecast(recipe, history_df, oos_df, cfg)
        used = "proxy"

    y_true = oos_df[cfg.target_col].astype(float).values
    metrics = j_obs(recipe, y_true, pred, ctx, lambda_c=lambda_c, wall_time_sec=wall_time)
    metrics["eval_backend"] = used
    return pred, metrics
