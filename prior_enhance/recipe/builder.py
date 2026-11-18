"""Recipe 采样与构建。"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from prior_enhance.recipe.catalog import (
    DEFAULT_HPO_SPACES,
    ENSEMBLE_TYPES,
    FEATURE_ENGINEERING_MODES,
    MODEL_CATALOG,
    SAMPLE_WINDOW_STRATEGIES,
    ModelCluster,
    TRAINABLE_MODELS,
    models_by_cluster,
)
from prior_enhance.recipe.schema import (
    FeatureEngineeringSpec,
    ForecastingRecipe,
    HyperparameterSpec,
    ModelSpec,
    OutputAndEnsembleSpec,
    SampleWindowSpec,
    ValidationSpec,
)
from prior_enhance.telemetry.run_logger import log

PHASE = "RECIPE"


def _pick_hpo_params(model_name: str, rng: random.Random) -> Dict[str, Any]:
    """从默认搜索空间随机采一组固定超参（用于 recipe 实例化）。"""
    space = DEFAULT_HPO_SPACES.get(model_name, {})
    if not space:
        return {}

    picked: Dict[str, Any] = {}
    for key, val in space.items():
        if isinstance(val, list) and val:
            picked[key] = rng.choice(val)
        elif isinstance(val, dict) and "start" in val and "end" in val:
            lo = int(val["start"]) if isinstance(val["start"], (int, float)) else val["start"]
            hi = int(val["end"]) if isinstance(val["end"], (int, float)) else val["end"]
            if isinstance(lo, int) and isinstance(hi, int):
                picked[key] = rng.randint(lo, hi)
            else:
                picked[key] = rng.uniform(float(lo), float(hi))
        else:
            picked[key] = val
    return picked


def build_recipe(
    model_name: str,
    *,
    sample_strategy: str = "full_history_unweighted",
    feature_mode: str = "none",
    feature_config: Optional[List[Dict[str, Any]]] = None,
    hpo_params: Optional[Dict[str, Any]] = None,
    enable_ensemble: bool = True,
    ensemble_types: Optional[List[str]] = None,
    output_mode: str = "point",
    prediction_length: int = 168,
    time_limit: int = 600,
    recipe_id: Optional[str] = None,
) -> ForecastingRecipe:
    """从显式配置构造单个 recipe。"""
    if model_name not in TRAINABLE_MODELS:
        raise ValueError(f"未知或不可训练模型: {model_name}")

    meta = MODEL_CATALOG[model_name]
    w_meta = SAMPLE_WINDOW_STRATEGIES.get(
        sample_strategy, SAMPLE_WINDOW_STRATEGIES["full_history_unweighted"]
    )
    f_meta = FEATURE_ENGINEERING_MODES.get(feature_mode, FEATURE_ENGINEERING_MODES["none"])

    recipe_core = {
        "model": model_name,
        "W": sample_strategy,
        "F": feature_mode,
        "theta": hpo_params or {},
        "Q": {"output_mode": output_mode, "ensemble": ensemble_types or ["weighted"]},
    }
    rid = recipe_id or ForecastingRecipe.make_id(recipe_core)

    recipe = ForecastingRecipe(
        recipe_id=rid,
        W=SampleWindowSpec(
            strategy=sample_strategy,
            recent_ratio=float(w_meta["recent_ratio"]),
        ),
        F=FeatureEngineeringSpec(
            mode=feature_mode,
            use_tsfresh=bool(f_meta["use_tsfresh"]),
            feature_config=feature_config if feature_mode == "tsfresh_custom" else None,
        ),
        M=ModelSpec(name=model_name, cluster=meta["cluster"]),
        theta=HyperparameterSpec(
            model_name=model_name,
            params=hpo_params or {},
        ),
        V=ValidationSpec(enabled=False),
        Q=OutputAndEnsembleSpec(
            output_mode=output_mode,
            enable_ensemble=enable_ensemble,
            ensemble_types=ensemble_types or (["weighted"] if enable_ensemble else []),
        ),
        prediction_length=prediction_length,
        time_limit=time_limit,
    )
    log.log_output("recipe_built", {"summary": recipe.summary(), "recipe_id": rid}, phase=PHASE)
    return recipe


def sample_random_recipe(
    rng: Optional[random.Random] = None,
    phase: int = 1,
    clusters: Optional[List[ModelCluster]] = None,
    prediction_length: int = 168,
    time_limit: int = 600,
) -> ForecastingRecipe:
    """
    从 AutoApex recipe 空间随机采样一个方案。

    phase: 1=统计+ML, 2=+深度, 3=+大模型
    """
    rng = rng or random.Random()

    allowed_clusters = clusters or [
        ModelCluster.STATISTICAL,
        ModelCluster.ML_TABULAR,
    ]
    if phase >= 2:
        allowed_clusters = list(allowed_clusters) + [ModelCluster.DEEP_LEARNING]
    if phase >= 3:
        allowed_clusters = list(allowed_clusters) + [ModelCluster.FOUNDATION]

    candidates: List[str] = []
    for c in allowed_clusters:
        candidates.extend(
            n for n in models_by_cluster(c) if MODEL_CATALOG[n].get("phase", 1) <= phase
        )
    model_name = rng.choice(candidates)

    sample_strategy = rng.choice(list(SAMPLE_WINDOW_STRATEGIES.keys()))
    feature_mode = rng.choices(
        ["none", "tsfresh_comprehensive"],
        weights=[0.7, 0.3],
        k=1,
    )[0]
    output_mode = rng.choices(["point", "quantile_interval"], weights=[0.6, 0.4], k=1)[0]
    enable_ensemble = rng.random() < 0.7
    ensemble_types = [rng.choice(list(ENSEMBLE_TYPES.keys()))] if enable_ensemble else []

    return build_recipe(
        model_name,
        sample_strategy=sample_strategy,
        feature_mode=feature_mode,
        hpo_params=_pick_hpo_params(model_name, rng),
        enable_ensemble=enable_ensemble,
        ensemble_types=ensemble_types,
        output_mode=output_mode,
        prediction_length=prediction_length,
        time_limit=time_limit,
    )


def default_phase1_recipes(prediction_length: int = 168) -> List[ForecastingRecipe]:
    """工程建议：第一阶段必跑基线 recipe 集合。"""
    names = [
        "SeasonalNaive",
        "AutoARIMA",
        "Theta",
        "DirectTabular",
        "RecursiveTabular",
    ]
    return [
        build_recipe(n, prediction_length=prediction_length, enable_ensemble=False)
        for n in names
    ]
