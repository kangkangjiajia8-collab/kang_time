"""J^val / J^obs 目标函数。"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from prior_enhance.recipe.catalog import MODEL_CATALOG, ModelCluster
from prior_enhance.recipe.schema import ForecastingRecipe
from prior_enhance.schemas import ContextState


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-8)
    if not mask.any():
        return 1.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def compute_cost(recipe: ForecastingRecipe, wall_time_sec: Optional[float] = None) -> float:
    """C_m(r)：计算开销；若提供 wall_time_sec 则优先使用真实训练耗时。"""
    if wall_time_sec is not None and wall_time_sec > 0:
        return float(wall_time_sec / 3600.0)
    meta = MODEL_CATALOG.get(recipe.M.name, {})
    cluster = meta.get("cluster", ModelCluster.ML_TABULAR)
    base = {
        ModelCluster.STATISTICAL: 0.05,
        ModelCluster.ML_TABULAR: 0.15,
        ModelCluster.DEEP_LEARNING: 0.8,
        ModelCluster.FOUNDATION: 1.2,
    }.get(cluster, 0.2)
    tsfresh_cost = 0.4 if recipe.F.use_tsfresh else 0.0
    ens_cost = 0.2 if recipe.Q.enable_ensemble else 0.0
    return float(base + tsfresh_cost + ens_cost + recipe.time_limit / 3600.0 * 0.1)


def mu_from_context(ctx: ContextState) -> float:
    """μ_m = μ(S_m)：风险惩罚强度（含 ξ_m）。"""
    return float(0.15 + 0.35 * ctx.d_m + 0.2 * ctx.nu_m + 0.3 * ctx.xi_m)


def validation_risk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    peak_weight: float = 0.5,
) -> float:
    """
    R^val：简化风险 — 峰值低估率 + 尾部 MAE。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return 0.0
    p90 = np.percentile(y_true, 90)
    peak_mask = y_true >= p90
    peak_under = 0.0
    if peak_mask.any():
        peak_under = float(np.mean(y_pred[peak_mask] < y_true[peak_mask]))
    tail_err = float(np.mean(np.abs(y_true - y_pred)))
    scale = float(np.mean(np.abs(y_true)) + 1e-8)
    return float(peak_weight * peak_under + (1.0 - peak_weight) * tail_err / scale)


def j_val(
    recipe: ForecastingRecipe,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: ContextState,
    *,
    lambda_c: float = 0.1,
    mu_m: Optional[float] = None,
    wall_time_sec: Optional[float] = None,
) -> Dict[str, float]:
    """J_{m,k}^{val}(r) = L^{val} + λ_C C + μ_m R^{val}"""
    mu = mu_m if mu_m is not None else mu_from_context(ctx)
    l_val = mape(y_true, y_pred)
    c = compute_cost(recipe, wall_time_sec=wall_time_sec)
    r = validation_risk(y_true, y_pred)
    total = l_val + lambda_c * c + mu * r
    return {
        "J_val": total,
        "L_val": l_val,
        "C": c,
        "R_val": r,
        "mu_m": mu,
    }


def j_obs(
    recipe: ForecastingRecipe,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ctx: ContextState,
    *,
    lambda_c: float = 0.1,
    mu_m: Optional[float] = None,
    wall_time_sec: Optional[float] = None,
) -> Dict[str, float]:
    """J_m^{obs}(r)：真实 O_m 观测目标（形式同 J^val）。"""
    out = j_val(
        recipe, y_true, y_pred, ctx,
        lambda_c=lambda_c, mu_m=mu_m, wall_time_sec=wall_time_sec,
    )
    out["J_obs"] = out.pop("J_val")
    out["L_obs"] = out.pop("L_val")
    out["R_obs"] = out.pop("R_val")
    return out
