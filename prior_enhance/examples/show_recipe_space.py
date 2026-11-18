#!/usr/bin/env python3
"""展示 AutoApex Recipe 空间定义与 train payload 示例。"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prior_enhance"))

from prior_enhance.recipe import (  # noqa: E402
    ModelCluster,
    build_recipe,
    default_phase1_recipes,
    models_by_cluster,
    recipe_to_train_payload,
    sample_random_recipe,
)
from prior_enhance.recipe.catalog import MODEL_CATALOG  # noqa: E402


def main():
    print("=== AutoApex Recipe 空间 r = (W, F, M, θ, V, Q) ===\n")

    print("【模型簇】")
    for cluster in ModelCluster:
        names = models_by_cluster(cluster)
        print(f"  {cluster.value:16s} ({len(names):2d}): {', '.join(names[:6])}{'...' if len(names)>6 else ''}")

    print("\n【第一阶段默认 recipe】")
    for r in default_phase1_recipes(prediction_length=168):
        print(f"  {r.summary()}")

    print("\n【随机采样 recipe (phase=2)】")
    r = sample_random_recipe(rng=random.Random(7), phase=2)
    print(f"  {r.summary()}")
    print(f"  theta.params = {json.dumps(r.theta.params, ensure_ascii=False)}")

    print("\n【TFT + 分位数 + WeightedEnsemble → train payload】")
    tft = build_recipe(
        "TemporalFusionTransformer",
        sample_strategy="recent_50pct",
        feature_mode="tsfresh_comprehensive",
        hpo_params={"hidden_dim": [64], "num_heads": [2], "batch_size": [64]},
        output_mode="quantile_interval",
        enable_ensemble=True,
        ensemble_types=["weighted"],
        prediction_length=168,
        time_limit=1200,
    )
    payload = recipe_to_train_payload(tft, task_id="demo_tft_recipe")
    print(json.dumps({k: v for k, v in payload.items() if k != "hyperparameters"}, indent=2, ensure_ascii=False))
    print("  hyperparameters:", payload["hyperparameters"])


if __name__ == "__main__":
    main()
