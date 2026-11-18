"""探索分布 p_exp：random / low-prior / anti-prior 三类对照 recipe。"""

from __future__ import annotations

import random
from typing import List, Optional

from prior_enhance.prior.compiler import FEATURE_ALIAS, WINDOW_ALIAS, phi_prior_recipe
from prior_enhance.prior.search_space import AUTOAPEX_SEARCH_SPACE
from prior_enhance.recipe.builder import build_recipe, sample_random_recipe
from prior_enhance.recipe.catalog import TRAINABLE_MODELS
from prior_enhance.recipe.schema import ForecastingRecipe
from prior_enhance.schemas import StructuredPrior


def _complement_pick(pool: List[str], avoid: List[str]) -> Optional[str]:
    avoid_set = set(avoid)
    candidates = [x for x in pool if x not in avoid_set]
    return random.choice(candidates) if candidates else None


def build_anti_prior_recipe(
    prior: StructuredPrior,
    rng: random.Random,
    *,
    prediction_length: int = 72,
) -> ForecastingRecipe:
    """
    Anti-prior：显式违反 target 推荐或 action 规避（仍满足可行性约束）。
    """
    t = prior.target
    # 窗口：避开推荐，倾向 full_history
    w_pool = AUTOAPEX_SEARCH_SPACE.get("window_strategy", [])
    w_raw = _complement_pick(w_pool, t.window_strategy) or "full_history_window"
    w_key = WINDOW_ALIAS.get(w_raw, "full_history_unweighted")

    # 特征：若 prior 建议轻量，则强制 tsfresh
    f_pool = AUTOAPEX_SEARCH_SPACE.get("feature_strategy", [])
    if "avoid_high_dimensional_tsfresh" in prior.action:
        f_key = "tsfresh_comprehensive"
    else:
        f_raw = _complement_pick(f_pool, t.feature_strategy) or "tsfresh_statistical"
        f_key = FEATURE_ALIAS.get(f_raw, "tsfresh_comprehensive")

    # 模型：若规避神经网络，则选 TFT；否则选推荐外模型
    m_pool = AUTOAPEX_SEARCH_SPACE.get("models", [])
    if "avoid_high_capacity_neural_models" in prior.action:
        m_name = "TemporalFusionTransformer" if "TemporalFusionTransformer" in TRAINABLE_MODELS else "DirectTabular"
    else:
        m_raw = _complement_pick(m_pool, t.models) or "SeasonalNaive"
        m_name = m_raw if m_raw in TRAINABLE_MODELS else "SeasonalNaive"

    # 集成：与推荐相反
    ens_pool = AUTOAPEX_SEARCH_SPACE.get("ensemble_strategy", [])
    if t.ensemble_strategy and t.ensemble_strategy[0] != "none":
        enable_ens = False
    else:
        enable_ens = True

    return build_recipe(
        m_name,
        sample_strategy=w_key,
        feature_mode=f_key,
        enable_ensemble=enable_ens,
        ensemble_types=["weighted"] if enable_ens else [],
        prediction_length=prediction_length,
        time_limit=600,
        recipe_id=None,
    )


def build_low_prior_recipe(
    priors: List[StructuredPrior],
    rng: random.Random,
    *,
    prediction_length: int = 72,
    max_phi: float = 0.25,
    attempts: int = 12,
) -> ForecastingRecipe:
    """Low-prior：随机采样但要求对有效 prior 集合的 φ 总分偏低。"""
    valid = [p for p in priors if p.valid]
    best: Optional[ForecastingRecipe] = None
    best_phi = 1.0
    for _ in range(attempts):
        r = sample_random_recipe(rng=rng, phase=1, prediction_length=prediction_length)
        if not valid:
            return r
        phi = sum(phi_prior_recipe(p, r) for p in valid) / len(valid)
        if phi < best_phi:
            best_phi = phi
            best = r
        if phi <= max_phi:
            return r
    return best or sample_random_recipe(rng=rng, phase=1, prediction_length=prediction_length)


def sample_p_exp_batch(
    priors: List[StructuredPrior],
    rng: random.Random,
    batch_size: int,
    *,
    prediction_length: int = 72,
    evaluated_ids: set,
) -> List[ForecastingRecipe]:
    """
    p_exp 三路：random / low-prior / anti-prior（各占约 1/3）。
    """
    if batch_size <= 0:
        return []
    valid = [p for p in priors if p.valid]
    batch: List[ForecastingRecipe] = []
    kinds = ["random", "low", "anti"]
    for i in range(batch_size):
        kind = kinds[i % len(kinds)]
        if kind == "random":
            r = sample_random_recipe(rng=rng, phase=1, prediction_length=prediction_length)
        elif kind == "low":
            r = build_low_prior_recipe(valid, rng, prediction_length=prediction_length)
        else:
            base = rng.choice(valid) if valid else None
            if base:
                r = build_anti_prior_recipe(base, rng, prediction_length=prediction_length)
            else:
                r = sample_random_recipe(rng=rng, phase=1, prediction_length=prediction_length)
        if r.recipe_id in evaluated_ids:
            continue
        batch.append(r)
        evaluated_ids.add(r.recipe_id)
    return batch
