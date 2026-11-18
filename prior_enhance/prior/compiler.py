"""z_{j,m} → φ_{j,m}(r) 先验满足度编译器。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from prior_enhance.recipe.schema import ForecastingRecipe
from prior_enhance.schemas import ContextState, PriorTargetClusters, StructuredPrior

# LLM 搜索空间名 → recipe catalog 名
WINDOW_ALIAS = {
    "full_history_window": "full_history_unweighted",
    "full_history": "full_history_unweighted",
    "recent_90d": "recent_75pct",
    "recent_50pct": "recent_50pct",
    "recent_25pct": "recent_25pct",
    "exp_decay": "exponential_decay",
    "exponential_decay": "exponential_decay",
}

FEATURE_ALIAS = {
    "none": "none",
    "calendar": "none",
    "lag": "none",
    "rolling": "none",
    "tsfresh_statistical": "tsfresh_comprehensive",
    "tsfresh_comprehensive": "tsfresh_comprehensive",
    "tsfresh_custom": "tsfresh_custom",
}

ENSEMBLE_ALIAS = {
    "none": False,
    "weighted_average": True,
    "simple_average": True,
    "median": True,
}


def _avoid_full_history_window(recipe: ForecastingRecipe) -> bool:
    return recipe.W.strategy != "full_history_unweighted" or recipe.W.recent_ratio < 0.99


def _avoid_high_dimensional_tsfresh(recipe: ForecastingRecipe) -> bool:
    return not recipe.F.use_tsfresh


def _avoid_large_ensemble(recipe: ForecastingRecipe) -> bool:
    return not recipe.Q.enable_ensemble


def _avoid_aggressive_hyperparameter_search(recipe: ForecastingRecipe) -> bool:
    return recipe.hyperparameter_tune_kwargs is None


def _avoid_high_capacity_neural_models(recipe: ForecastingRecipe) -> bool:
    neural = {
        "TemporalFusionTransformer", "PatchTST", "TiDE", "DLinear", "DeepAR",
        "Chronos", "Chronos2", "WaveNet",
    }
    return recipe.M.name not in neural


def _avoid_overfitting(recipe: ForecastingRecipe) -> bool:
    return recipe.W.recent_ratio <= 0.75 or not recipe.F.use_tsfresh


AVOID_HANDLERS = {
    "avoid_full_history_window": _avoid_full_history_window,
    "avoid_high_dimensional_tsfresh": _avoid_high_dimensional_tsfresh,
    "avoid_large_ensemble": _avoid_large_ensemble,
    "avoid_aggressive_hyperparameter_search": _avoid_aggressive_hyperparameter_search,
    "avoid_high_capacity_neural_models": _avoid_high_capacity_neural_models,
    "avoid_overfitting": _avoid_overfitting,
    "avoid_overreliance_on_long_history": _avoid_full_history_window,
    "avoid_static_feature_selection": lambda r: r.F.mode != "none",
}


def _recipe_window_key(recipe: ForecastingRecipe) -> str:
    return recipe.W.strategy


def _recipe_feature_key(recipe: ForecastingRecipe) -> str:
    return recipe.F.mode


def _overlap_score(recipe_val: str, recommended: List[str], alias: Dict[str, str]) -> float:
    if not recommended:
        return 0.0
    rv = alias.get(recipe_val, recipe_val)
    hits = 0
    for item in recommended:
        key = alias.get(item, item)
        if rv == key:
            hits += 1
    return hits / len(recommended)


def _model_overlap(model_name: str, recommended: List[str]) -> float:
    if not recommended:
        return 0.0
    if model_name in recommended:
        return 1.0
    # 同簇弱匹配
    families = {
        "stat": {"SeasonalNaive", "AutoARIMA", "ETS", "Theta", "ARIMA"},
        "ml": {"DirectTabular", "RecursiveTabular", "LightGBM", "XGBoost"},
    }
    for fam in families.values():
        if model_name in fam and any(m in fam for m in recommended):
            return 0.5
    return 0.0


def _ensemble_overlap(recipe: ForecastingRecipe, recommended: List[str]) -> float:
    if not recommended:
        return 0.0
    enabled = recipe.Q.enable_ensemble
    for item in recommended:
        want = ENSEMBLE_ALIAS.get(item)
        if want is None:
            continue
        if want == enabled:
            return 1.0
    return 0.0 if enabled else 0.5


def phi_prior_recipe(prior: StructuredPrior, recipe: ForecastingRecipe) -> float:
    """
    φ_{j,m}(r) ∈ [0,1]：recipe 对单条 prior 的满足程度。

    正向 target 重叠 + 负向 action 不违反。
    """
    t: PriorTargetClusters = prior.target
    pos_scores: List[float] = []

    w_score = _overlap_score(_recipe_window_key(recipe), t.window_strategy, WINDOW_ALIAS)
    if t.window_strategy:
        pos_scores.append(w_score)

    f_score = _overlap_score(_recipe_feature_key(recipe), t.feature_strategy, FEATURE_ALIAS)
    if t.feature_strategy:
        pos_scores.append(f_score)

    m_score = _model_overlap(recipe.M.name, t.models)
    if t.models:
        pos_scores.append(m_score)

    e_score = _ensemble_overlap(recipe, t.ensemble_strategy)
    if t.ensemble_strategy:
        pos_scores.append(e_score)

    if not pos_scores:
        pos = 0.0
    else:
        pos = sum(pos_scores) / len(pos_scores)

    avoid_ok = 1.0
    for act in prior.action:
        handler = AVOID_HANDLERS.get(act)
        if handler and not handler(recipe):
            avoid_ok = 0.0
            break

    return float(max(0.0, min(1.0, pos * avoid_ok)))


def phi_total(priors: List[StructuredPrior], recipe: ForecastingRecipe) -> float:
    """所有有效 prior 的 φ 加权平均（权重 = confidence）。"""
    valid = [p for p in priors if p.valid]
    if not valid:
        return 0.0
    num = sum(p.confidence * phi_prior_recipe(p, recipe) for p in valid)
    den = sum(p.confidence for p in valid) + 1e-8
    return float(num / den)


def applicability(prior: StructuredPrior, ctx: ContextState) -> float:
    """a_j(S_m)：cond 与当前 S_m 标签匹配度。"""
    if not prior.cond:
        return 0.5
    sm_labels = {ctx.n_label, ctx.nu_label, ctx.d_label}
    hits = sum(1 for c in prior.cond if c in sm_labels)
    return float(hits / len(prior.cond))


def init_tau_persistent(
    prior: StructuredPrior,
    historical: Sequence[Tuple[StructuredPrior, float]],
    *,
    s_min: float = 0.25,
    default_tau: float = 0.5,
    semantic_weight: float = 0.15,
) -> Tuple[float, int]:
    """
    τ_{j,m}：历史相似 prior 加权（论文 Prior Memory 检索）。

    无满足 s_min 的历史先验时返回 0.5 无信息先验。
    """
    from prior_enhance.prior.similarity import weighted_tau_from_similar_priors

    return weighted_tau_from_similar_priors(
        prior,
        historical,
        s_min=s_min,
        default_tau=default_tau,
        semantic_weight=semantic_weight,
    )


def init_tau_loc(
    priors: List[StructuredPrior],
    ctx: ContextState,
    lambda_m: float = 0.5,
    *,
    historical: Optional[Sequence[Tuple[StructuredPrior, float]]] = None,
    s_min: float = 0.25,
    epsilon_tau: float = 1e-3,
    semantic_weight: float = 0.15,
) -> Dict[str, float]:
    """
    τ_{j,m,0}^{loc} = τ_{j,m}^0（论文初始局部可靠性）。

    τ^0 = clip(a_j * [λ_m b + (1-λ_m) τ_{j,m}], ε_τ, 1-ε_τ)
    τ_{j,m} 来自 prior_store 结构相似检索；无历史时 τ_{j,m}=0.5。
    """
    tau: Dict[str, float] = {}
    hist = list(historical or [])
    for p in priors:
        if not p.valid:
            continue
        a = applicability(p, ctx)
        tau_persist, _ = init_tau_persistent(
            p, hist, s_min=s_min, semantic_weight=semantic_weight,
        )
        blended = lambda_m * p.confidence + (1.0 - lambda_m) * tau_persist
        tau[p.prior_id] = float(
            max(epsilon_tau, min(1.0 - epsilon_tau, a * blended))
        )
    return tau


def recipe_feature_vector(recipe: ForecastingRecipe) -> Dict[str, float]:
    """x_m(r)：混杂控制特征（回归用）。"""
    cluster_map = {
        "statistical": 0.0,
        "ml_tabular": 1.0,
        "deep_learning": 2.0,
        "foundation": 3.0,
    }
    return {
        "cluster_code": cluster_map.get(recipe.M.cluster.value, 1.0),
        "recent_ratio": recipe.W.recent_ratio,
        "use_tsfresh": 1.0 if recipe.F.use_tsfresh else 0.0,
        "enable_ensemble": 1.0 if recipe.Q.enable_ensemble else 0.0,
        "time_limit_log": float(recipe.time_limit) / 600.0,
    }
