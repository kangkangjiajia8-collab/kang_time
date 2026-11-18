"""O_m 解锁后的 Observation Feedback。"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from prior_enhance.config import AutoApexConfig, DataConfig, SearchConfig
from prior_enhance.observation.memory import PriorMemory, update_persistent_memory
from prior_enhance.observation.prior_store import PriorStore
from prior_enhance.prior.compiler import phi_prior_recipe
from prior_enhance.recipe.schema import ForecastingRecipe
from prior_enhance.schemas import ContextState, ObservationResult, PriorSet, SearchResult
from prior_enhance.telemetry.run_logger import log
from prior_enhance.validation.evaluator import evaluate_on_oos
import os

from prior_enhance.validation.gain import compute_contrastive_obs_gain, compute_prior_gain

PHASE = "OBSERVE"


def run_observation_feedback(
    search_result: SearchResult,
    prior_set: PriorSet,
    history_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    ctx: ContextState,
    memory: Optional[PriorMemory] = None,
    cfg: Optional[SearchConfig] = None,
    dcfg: Optional[DataConfig] = None,
    apex_cfg: Optional[AutoApexConfig] = None,
    prior_store: Optional[PriorStore] = None,
) -> ObservationResult:
    """
    O_m 真实值揭晓后：对 E_m^{retain} 算 J^obs、G^obs，慢速更新持久 τ 与 λ。
    """
    cfg = cfg or SearchConfig()
    dcfg = dcfg or DataConfig()
    memory = memory or PriorMemory()

    evaluated_records = getattr(search_result, "_evaluated_records", [])
    id_to_recipe = {e["recipe"].recipe_id: e["recipe"] for e in evaluated_records}
    retain_ids = search_result.retain_ids or list(id_to_recipe.keys())[:5]
    retain_recipes = [id_to_recipe[rid] for rid in retain_ids if rid in id_to_recipe]

    obs_records: List[dict] = []
    for recipe in retain_recipes:
        pred, metrics = evaluate_on_oos(recipe, history_df, oos_df, ctx, dcfg, lambda_c=cfg.lambda_c, apex_cfg=apex_cfg)
        obs_records.append({
            "recipe": recipe,
            "metrics": metrics,
            "oos_pred": pred.tolist(),
        })
        log.log_output("obs_eval", {
            "recipe_id": recipe.recipe_id,
            "J_obs": metrics["J_obs"],
            "L_obs": metrics["L_obs"],
        }, phase=PHASE)

    gain_method = os.getenv("PRIOR_ENHANCE_OBS_GAIN_METHOD", "regression").strip().lower()
    gain_results = []
    for p in prior_set.valid_priors():
        if gain_method == "contrastive":
            gr = compute_contrastive_obs_gain(
                p, obs_records,
                chi_min=cfg.chi_min,
            )
        else:
            gr = compute_prior_gain(
                p, obs_records,
                min_samples=max(3, cfg.min_samples_for_gain - 1),
                chi_min=cfg.chi_min,
                kind="obs",
                require_balance=cfg.explore_ratio > 0,
            )
            if not gr.accepted and len(obs_records) >= 3:
                gr_alt = compute_contrastive_obs_gain(p, obs_records, chi_min=cfg.chi_min)
                if gr_alt.accepted:
                    gr = gr_alt
        gain_results.append(gr)

    memory = update_persistent_memory(
        memory, prior_set, gain_results, ctx,
        omega_max=cfg.omega_obs_max,
        eta_obs=cfg.eta_obs,
        eta_lambda=cfg.eta_lambda,
    )

    if prior_store is not None:
        prior_store.append_window(
            window_index=search_result.window_index,
            decision_time=search_result.decision_time,
            priors=prior_set.valid_priors(),
            tau_map=memory.tau,
            scenario=ctx.scenario,
        )
        prior_store.save()
        log.log_output("prior_store_append", {
            "path": prior_store.path,
            "total_records": len(prior_store),
        }, phase=PHASE)

    best_obs = min(obs_records, key=lambda x: x["metrics"]["J_obs"]) if obs_records else None
    return ObservationResult(
        window_index=search_result.window_index,
        decision_time=search_result.decision_time,
        n_retain=len(obs_records),
        observation_summary=[
            {
                "recipe_id": r["recipe"].recipe_id,
                "J_obs": r["metrics"]["J_obs"],
                "L_obs": r["metrics"]["L_obs"],
            }
            for r in obs_records
        ],
        gain_summary=[
            {
                "prior_id": g.prior_id,
                "gain": g.gain,
                "delta": g.delta,
                "chi": g.chi,
                "accepted": g.accepted,
                "reason": g.reason,
            }
            for g in gain_results
        ],
        best_obs_recipe_id=best_obs["recipe"].recipe_id if best_obs else None,
        best_j_obs=best_obs["metrics"]["J_obs"] if best_obs else None,
        memory_tau=memory.tau,
        lambda_m=memory.lambda_m,
        oos_forecasts=[
            {
                "recipe_id": r["recipe"].recipe_id,
                "pred": r["oos_pred"],
                "J_obs": r["metrics"]["J_obs"],
            }
            for r in obs_records
        ],
    )
