"""持久 Prior Memory：τ_{j,m} 与全局 λ_m。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from prior_enhance.schemas import ContextState, PriorSet
from prior_enhance.validation.gain import GainResult, update_tau_logit


@dataclass
class PriorMemory:
    """跨窗口传递的先验记忆。"""

    tau: Dict[str, float] = field(default_factory=dict)
    lambda_m: float = 0.5


def omega_forget_rate(ctx: ContextState, omega_max: float = 0.4) -> float:
    """
    ω_m：非平稳/高风险时加速遗忘。

    ω_m = ω_max · σ(α_0 - α_n n_m + α_ν ν_m + α_d d_m + α_ξ ξ_m)
    """
    z = 1.2 - 1.0 * ctx.n_m + 0.8 * ctx.nu_m + 1.0 * ctx.d_m + 0.6 * ctx.xi_m
    sig = 1.0 / (1.0 + np.exp(-z))
    return float(omega_max * sig)


def update_persistent_memory(
    memory: PriorMemory,
    prior_set: PriorSet,
    gain_results: List[GainResult],
    ctx: ContextState,
    *,
    omega_max: float = 0.4,
    eta_obs: float = 0.1,
    eta_lambda: float = 0.05,
) -> PriorMemory:
    """l_{j,m+1} = (1-ω_m) l_{j,m} + η_obs χ Δ^{obs}"""
    omega = omega_forget_rate(ctx, omega_max)
    gain_map = {g.prior_id: g for g in gain_results}

    for p in prior_set.valid_priors():
        pid = p.prior_id
        old_tau = memory.tau.get(pid, 0.5)
        gr = gain_map.get(pid)
        if gr and gr.accepted:
            memory.tau[pid] = update_tau_logit(
                old_tau, gr.delta, gr.chi,
                omega_val=omega,
                eta_val=eta_obs,
            )
        else:
            # 证据不足：仅衰减向 0.5
            memory.tau[pid] = update_tau_logit(old_tau, 0.0, 0.0, omega_val=omega, eta_val=0.0)

    accepted = [g for g in gain_results if g.accepted]
    if accepted:
        num = sum(g.chi * memory.tau.get(g.prior_id, 0.5) for g in accepted)
        den = sum(g.chi for g in accepted) + 1e-8
        target_lambda = num / den
        memory.lambda_m = float(
            (1.0 - eta_lambda) * memory.lambda_m + eta_lambda * target_lambda
        )
    return memory
