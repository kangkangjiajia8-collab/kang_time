"""Prior-guided 提议分布 q_{m,k}(r) ∝ exp(β_k ψ_{m,k}(r))。"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Set

from prior_enhance.prior.recipe_sampler import prior_guided_recipe
from prior_enhance.recipe.builder import sample_random_recipe
from prior_enhance.recipe.schema import ForecastingRecipe
from prior_enhance.schemas import PriorSet


def softmax_weights(scores: List[float], beta: float = 1.0) -> List[float]:
    """q(r) ∝ exp(β ψ)；数值稳定 softmax。"""
    if not scores:
        return []
    scaled = [beta * float(s) for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps) + 1e-12
    return [e / total for e in exps]


def build_candidate_pool(
    priors: PriorSet,
    rng: random.Random,
    pool_size: int,
    *,
    prediction_length: int,
    evaluated_ids: Set[str],
    phase: int = 1,
) -> List[ForecastingRecipe]:
    """
    构造候选 recipe 池（用于近似全空间 softmax 归一化）。

    混合 prior 偏置采样与随机采样，去重 recipe_id。
    """
    pool: List[ForecastingRecipe] = []
    seen: Set[str] = set(evaluated_ids)
    max_attempts = max(pool_size * 8, 32)
    attempts = 0
    while len(pool) < pool_size and attempts < max_attempts:
        attempts += 1
        if rng.random() < 0.6 and priors.valid_priors():
            r = prior_guided_recipe(
                priors.priors, rng, prediction_length=prediction_length, phase=phase,
            )
        else:
            r = sample_random_recipe(
                rng=rng, phase=phase, prediction_length=prediction_length,
            )
        if r.recipe_id in seen:
            continue
        seen.add(r.recipe_id)
        pool.append(r)
    return pool


def sample_batch_from_q(
    priors: PriorSet,
    tau_loc: Dict[str, float],
    psi_fn: Callable[[ForecastingRecipe], float],
    rng: random.Random,
    batch_size: int,
    *,
    beta: float = 1.0,
    pool_size: int = 48,
    prediction_length: int = 72,
    evaluated_ids: Set[str],
    phase: int = 1,
) -> List[ForecastingRecipe]:
    """
    从 q_{m,k}(r) ∝ exp(β ψ) 无放回采样 batch（ρ_k=0 时 p=q）。

    每轮重建候选池，过滤已评估 id，按 softmax 权重抽样。
    """
    if batch_size <= 0:
        return []

    pool = build_candidate_pool(
        priors, rng, pool_size,
        prediction_length=prediction_length,
        evaluated_ids=evaluated_ids,
        phase=phase,
    )
    if not pool:
        return []

    psi_vals = [psi_fn(r) for r in pool]
    weights = softmax_weights(psi_vals, beta=beta)
    batch: List[ForecastingRecipe] = []
    remaining = list(range(len(pool)))
    rem_weights = weights[:]

    for _ in range(min(batch_size, len(pool))):
        if not remaining:
            break
        rw = [rem_weights[i] for i in remaining]
        s = sum(rw)
        if s <= 0:
            pick_local = rng.choice(remaining)
        else:
            probs = [w / s for w in rw]
            pick_local = rng.choices(remaining, weights=probs, k=1)[0]
        batch.append(pool[pick_local])
        evaluated_ids.add(pool[pick_local].recipe_id)
        remaining.remove(pick_local)

    return batch
