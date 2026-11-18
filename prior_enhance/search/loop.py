"""Prior-guided recipe search 主循环（窗口内）。"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from prior_enhance.config import AutoApexConfig, DataConfig, SearchConfig
from prior_enhance.observation.prior_store import PriorStore
from prior_enhance.prior.compiler import init_tau_loc, phi_total
from prior_enhance.prior.recipe_sampler import prior_guided_recipe
from prior_enhance.recipe.anti_prior_builder import sample_p_exp_batch
from prior_enhance.search.proposal import sample_batch_from_q
from prior_enhance.recipe.schema import ForecastingRecipe
from prior_enhance.schemas import ContextState, ForecastTask, PriorSet, SearchResult
from prior_enhance.telemetry.run_logger import log
from prior_enhance.validation.evaluator import evaluate_on_split
from prior_enhance.validation.gain import compute_prior_gain, update_tau_logit
from prior_enhance.validation.split import make_holdout_val_split

PHASE = "SEARCH"


def _psi_score(
    recipe: ForecastingRecipe,
    priors: PriorSet,
    tau_loc: Dict[str, float],
) -> float:
    """ψ_{m,k}(r) = Σ_j τ_j^loc · φ_j(r)"""
    total = 0.0
    for p in priors.valid_priors():
        from prior_enhance.prior.compiler import phi_prior_recipe

        tau = tau_loc.get(p.prior_id, p.confidence)
        total += tau * phi_prior_recipe(p, recipe)
    return float(total)


def _sample_batch(
    prior_set: PriorSet,
    tau_loc: Dict[str, float],
    rng: random.Random,
    batch_size: int,
    cfg: SearchConfig,
    prediction_length: int,
    evaluated_ids: set,
) -> List[ForecastingRecipe]:
    """p_{m,k} = (1-ρ_k) q_{m,k} + ρ_k p_exp（论文混合采样）。"""
    rho = float(np.clip(cfg.explore_ratio, 0.0, 0.9)) if batch_size > 1 else 0.0
    n_explore = max(1, int(round(batch_size * rho))) if rho > 0 else 0
    n_guide = batch_size - n_explore

    guide_batch: List[ForecastingRecipe] = []
    if n_guide > 0:
        if cfg.use_softmax_proposal:
            psi_fn = lambda r: _psi_score(r, prior_set, tau_loc)
            guide_batch = sample_batch_from_q(
                prior_set,
                tau_loc,
                psi_fn,
                rng,
                n_guide,
                beta=cfg.beta_k,
                pool_size=cfg.proposal_pool_size,
                prediction_length=prediction_length,
                evaluated_ids=evaluated_ids,
            )
        else:
            for _ in range(n_guide):
                r = prior_guided_recipe(
                    prior_set.priors, rng, prediction_length=prediction_length,
                )
                if r.recipe_id not in evaluated_ids:
                    guide_batch.append(r)
                    evaluated_ids.add(r.recipe_id)

    explore_batch: List[ForecastingRecipe] = []
    if n_explore > 0:
        explore_batch = sample_p_exp_batch(
            prior_set.priors,
            rng,
            n_explore,
            prediction_length=prediction_length,
            evaluated_ids=evaluated_ids,
        )

    return guide_batch + explore_batch


def run_prior_guided_search(
    task: ForecastTask,
    prior_set: PriorSet,
    history_df: pd.DataFrame,
    cfg: Optional[SearchConfig] = None,
    dcfg: Optional[DataConfig] = None,
    apex_cfg: Optional[AutoApexConfig] = None,
    prior_store: Optional[PriorStore] = None,
    lambda_m: Optional[float] = None,
    *,
    seed: int = 42,
) -> SearchResult:
    """
    窗口 m 内局部搜索：评估 → ψ 采样 → τ^loc 更新 → 选 r*。

    仅使用 H_m 内部 validation；不触碰 O_m。
    """
    cfg = cfg or SearchConfig()
    dcfg = dcfg or DataConfig()
    ctx = task.context_state
    rng = random.Random(seed)

    val_split = make_holdout_val_split(history_df, cfg.val_horizon, dcfg)
    hist = prior_store.candidate_pairs() if prior_store else []
    lam = lambda_m if lambda_m is not None else (cfg.lambda_m if cfg.lambda_m is not None else 0.5)
    tau_loc = init_tau_loc(
        prior_set.valid_priors(),
        ctx,
        lambda_m=lam,
        historical=hist,
        s_min=cfg.prior_similarity_min,
        semantic_weight=cfg.prior_semantic_weight,
    )
    evaluated: List[dict] = []
    evaluated_ids: set = set()

    log.log_output("search_init", {
        "val_horizon": cfg.val_horizon,
        "tau_loc": tau_loc,
        "budget": cfg.max_trials,
        "beta_k": cfg.beta_k,
        "explore_ratio": cfg.explore_ratio,
        "use_softmax_proposal": cfg.use_softmax_proposal,
        "proposal_pool_size": cfg.proposal_pool_size,
    }, phase=PHASE)

    trials = 0
    round_idx = 0
    while trials < cfg.max_trials:
        round_idx += 1
        batch = _sample_batch(
            prior_set,
            tau_loc,
            rng,
            min(cfg.batch_size, cfg.max_trials - trials),
            cfg,
            task.horizon,
            evaluated_ids,
        )
        if not batch:
            break

        for recipe in batch:
            pred, metrics = evaluate_on_split(
                recipe,
                val_split.train_df,
                val_split.val_df,
                ctx,
                dcfg,
                lambda_c=cfg.lambda_c,
                apex_cfg=apex_cfg,
            )
            psi = _psi_score(recipe, prior_set, tau_loc)
            phi = phi_total(prior_set.valid_priors(), recipe)
            evaluated.append({
                "recipe": recipe,
                "metrics": metrics,
                "psi": psi,
                "phi": phi,
                "round": round_idx,
                "val_pred": pred.tolist(),
            })
            trials += 1
            log.log_output("recipe_eval", {
                "round": round_idx,
                "recipe_id": recipe.recipe_id,
                "model": recipe.M.name,
                "J_val": metrics["J_val"],
                "psi": psi,
                "phi": phi,
            }, phase=PHASE)

        # τ^loc 局部更新（仅窗口内，不写持久 τ）
        if len(evaluated) >= cfg.min_samples_for_gain:
            for p in prior_set.valid_priors():
                gr = compute_prior_gain(
                    p, evaluated,
                    min_samples=cfg.min_samples_for_gain,
                    chi_min=cfg.chi_min,
                    kind="val",
                    require_balance=cfg.explore_ratio > 0,
                )
                if gr.accepted:
                    old = tau_loc.get(p.prior_id, p.confidence)
                    tau_loc[p.prior_id] = update_tau_logit(
                        old, gr.delta, gr.chi,
                        omega_val=cfg.omega_val,
                        eta_val=cfg.eta_val,
                    )
                    log.log_output("tau_loc_update", {
                        "prior_id": p.prior_id,
                        "gain": gr.gain,
                        "delta": gr.delta,
                        "chi": gr.chi,
                        "tau_new": tau_loc[p.prior_id],
                    }, phase=PHASE)

    if not evaluated:
        raise RuntimeError("搜索未评估任何 recipe")

    # r_m* = argmin J^val（κ=0）
    best = min(evaluated, key=lambda x: x["metrics"]["J_val"])
    top_k = sorted(evaluated, key=lambda x: x["metrics"]["J_val"])[: cfg.retain_top_k]
    explore_pool = sorted(evaluated, key=lambda x: x["phi"])[: cfg.retain_explore_k]

    result = SearchResult(
        window_index=task.window_index,
        decision_time=task.decision_time,
        n_evaluated=len(evaluated),
        n_rounds=round_idx,
        best_recipe_id=best["recipe"].recipe_id,
        best_j_val=best["metrics"]["J_val"],
        tau_loc_final=tau_loc,
        evaluated_summary=[
            {
                "recipe_id": e["recipe"].recipe_id,
                "model": e["recipe"].M.name,
                "W": e["recipe"].W.strategy,
                "J_val": e["metrics"]["J_val"],
                "L_val": e["metrics"]["L_val"],
                "psi": e["psi"],
                "phi": e["phi"],
            }
            for e in evaluated
        ],
        retain_ids=[e["recipe"].recipe_id for e in top_k] + [
            e["recipe"].recipe_id for e in explore_pool
            if e["recipe"].recipe_id not in {t["recipe"].recipe_id for t in top_k}
        ],
    )
    result._evaluated_records = evaluated  # type: ignore[attr-defined]
    result._best_recipe = best["recipe"]  # type: ignore[attr-defined]
    result._retain_recipes = list({e["recipe"].recipe_id: e["recipe"] for e in top_k + explore_pool}.values())

    log.log_output("search_done", {
        "best_recipe_id": result.best_recipe_id,
        "best_j_val": result.best_j_val,
        "n_evaluated": result.n_evaluated,
    }, phase=PHASE)
    return result
