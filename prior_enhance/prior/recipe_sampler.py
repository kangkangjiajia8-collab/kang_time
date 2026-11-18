"""从 Z_m 与搜索空间采样 candidate recipe。"""

from __future__ import annotations

import random
from typing import List, Optional

from prior_enhance.prior.compiler import FEATURE_ALIAS, WINDOW_ALIAS
from prior_enhance.recipe.builder import build_recipe, sample_random_recipe
from prior_enhance.recipe.catalog import MODEL_CATALOG, TRAINABLE_MODELS
from prior_enhance.recipe.schema import ForecastingRecipe
from prior_enhance.schemas import StructuredPrior

MODEL_ALIAS = {
    "LightGBM": "RecursiveTabular",
    "XGBoost": "DirectTabular",
}


def _pick(rng: random.Random, items: List[str], fallback: List[str]) -> str:
    pool = [x for x in items if x]
    if not pool:
        pool = fallback
    return rng.choice(pool)


def prior_guided_recipe(
    priors: List[StructuredPrior],
    rng: random.Random,
    *,
    prediction_length: int = 72,
    phase: int = 1,
) -> ForecastingRecipe:
    """从有效 prior 的 target 簇中偏置采样一个 recipe。"""
    valid = [p for p in priors if p.valid]
    if not valid:
        return sample_random_recipe(rng=rng, phase=phase, prediction_length=prediction_length)

    weights = [p.confidence for p in valid]
    prior = rng.choices(valid, weights=weights, k=1)[0]
    t = prior.target

    w_raw = _pick(rng, t.window_strategy, ["recent_50pct", "exponential_decay"])
    w_key = WINDOW_ALIAS.get(w_raw, w_raw)
    if w_key not in {"full_history_unweighted", "recent_75pct", "recent_50pct", "recent_25pct", "exponential_decay"}:
        w_key = "recent_50pct"

    f_raw = _pick(rng, t.feature_strategy, ["none", "tsfresh_comprehensive"])
    f_key = FEATURE_ALIAS.get(f_raw, "none")

    m_raw = _pick(rng, t.models, ["SeasonalNaive", "AutoARIMA", "RecursiveTabular"])
    m_name = MODEL_ALIAS.get(m_raw, m_raw)
    if m_name not in TRAINABLE_MODELS:
        m_name = "SeasonalNaive"

    ens_raw = t.ensemble_strategy[0] if t.ensemble_strategy else "none"
    enable_ens = ens_raw not in ("none",)

    return build_recipe(
        m_name,
        sample_strategy=w_key,
        feature_mode=f_key,
        enable_ensemble=enable_ens,
        ensemble_types=["weighted"] if enable_ens else [],
        prediction_length=prediction_length,
        time_limit=300 if MODEL_CATALOG.get(m_name, {}).get("phase", 1) == 1 else 600,
        recipe_id=None,
    )
