"""Validation / Observation Gain 回归与证据门控。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from prior_enhance.prior.compiler import phi_prior_recipe, recipe_feature_vector
from prior_enhance.schemas import StructuredPrior


@dataclass
class GainResult:
    prior_id: str
    beta: float
    gain: float
    delta: float
    se: float
    chi: float
    accepted: bool
    reason: str = ""


def _ols_beta(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """简单一元 OLS：y ~ x，返回 (beta, se_beta)。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-8:
        return 0.0, np.inf
    x_mean = x.mean()
    y_mean = y.mean()
    cov = np.mean((x - x_mean) * (y - y_mean))
    var = np.mean((x - x_mean) ** 2) + 1e-12
    beta = cov / var
    resid = y - (y_mean + beta * (x - x_mean))
    se = float(np.sqrt(np.mean(resid ** 2) / (len(x) * var)))
    return float(beta), se


def _multivariate_ols_beta(
    y: np.ndarray,
    phi: np.ndarray,
    controls: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    多元 OLS：y ~ φ + γ^T x + ε，返回 (β_φ, se_β_φ)。
    controls 形状 (n, k)。
    """
    y = np.asarray(y, dtype=float)
    phi = np.asarray(phi, dtype=float).reshape(-1, 1)
    n = len(y)
    if n < 4 or np.std(phi) < 1e-8:
        return _ols_beta(y, phi.ravel())

    if controls is not None and controls.size > 0:
        X = np.column_stack([np.ones(n), phi.ravel(), np.asarray(controls, dtype=float)])
    else:
        X = np.column_stack([np.ones(n), phi.ravel()])

    if X.shape[0] <= X.shape[1]:
        return _ols_beta(y, phi.ravel())

    try:
        coef, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        beta_phi = float(coef[1])
        if residuals.size > 0:
            rss = float(residuals[0])
        else:
            rss = float(np.sum((y - X @ coef) ** 2))
        dof = max(n - rank, 1)
        var_beta = rss / dof * np.linalg.pinv(X.T @ X)[1, 1]
        se = float(np.sqrt(max(var_beta, 1e-12)))
        return beta_phi, se
    except np.linalg.LinAlgError:
        return _ols_beta(y, phi.ravel())


def evidence_quality(
    phi_values: List[float],
    j_values: List[float],
    *,
    min_samples: int = 4,
    chi_min: float = 0.3,
    require_balance: bool = False,
    round_ids: Optional[List[int]] = None,
) -> Tuple[float, str]:
    """
    χ = q_sample · q_balance · q_quality · q_stability。
    """
    n = len(j_values)
    if n < min_samples:
        return 0.0, "insufficient_sample"

    q_sample = min(1.0, n / max(min_samples, 1))

    if require_balance:
        low = sum(1 for p in phi_values if p < 0.3)
        high = sum(1 for p in phi_values if p >= 0.5)
        q_balance = 1.0 if low >= 1 and high >= 1 else 0.0
    else:
        low = sum(1 for p in phi_values if p < 0.35)
        high = sum(1 for p in phi_values if p >= 0.45)
        q_balance = 1.0 if (low >= 1 and high >= 1) else 0.5

    if np.std(j_values) < 1e-8 or np.std(phi_values) < 1e-8:
        q_quality = 0.2
    else:
        corr = abs(float(np.corrcoef(phi_values, j_values)[0, 1]))
        q_quality = float(np.clip(corr, 0.2, 1.0))

    if round_ids and len(set(round_ids)) >= 2:
        by_round: Dict[int, List[float]] = {}
        for rid, jv in zip(round_ids, j_values):
            by_round.setdefault(rid, []).append(jv)
        means = [float(np.mean(v)) for v in by_round.values() if v]
        if len(means) >= 2 and np.mean(np.abs(means)) > 1e-8:
            cv = float(np.std(means) / (np.mean(np.abs(means)) + 1e-8))
            q_stability = float(np.clip(1.0 - cv, 0.3, 1.0))
        else:
            q_stability = 0.6
    else:
        q_stability = 0.7

    chi = float(q_sample * q_balance * q_quality * q_stability)
    if chi < chi_min:
        return chi, "chi_below_min"
    return chi, "ok"


def compute_prior_gain(
    prior: StructuredPrior,
    records: List[dict],
    *,
    delta_max: float = 2.0,
    eps_g: float = 0.1,
    min_samples: int = 4,
    chi_min: float = 0.3,
    kind: str = "val",
    require_balance: bool = False,
    use_controls: bool = True,
) -> GainResult:
    """
    对单条 prior 拟合 J ~ φ_j + γ^T x_m(r) + ε，G = -β_φ。
    """
    j_key = "J_val" if kind == "val" else "J_obs"
    phi_vals: List[float] = []
    j_vals: List[float] = []
    control_rows: List[List[float]] = []
    round_ids: List[int] = []

    for rec in records:
        recipe = rec["recipe"]
        phi_vals.append(phi_prior_recipe(prior, recipe))
        j_vals.append(rec["metrics"][j_key])
        if use_controls:
            fv = recipe_feature_vector(recipe)
            control_rows.append([
                fv["cluster_code"],
                fv["recent_ratio"],
                fv["use_tsfresh"],
                fv["enable_ensemble"],
            ])
        round_ids.append(int(rec.get("round", 0)))

    y = np.array(j_vals, dtype=float)
    phi = np.array(phi_vals, dtype=float)
    controls = np.array(control_rows, dtype=float) if control_rows else None
    beta, se = _multivariate_ols_beta(y, phi, controls)
    gain = -beta
    delta = float(np.clip(gain / (se + eps_g), -delta_max, delta_max))
    chi, reason = evidence_quality(
        phi_vals,
        j_vals,
        min_samples=min_samples,
        chi_min=chi_min,
        require_balance=require_balance,
        round_ids=round_ids,
    )
    accepted = reason == "ok"

    return GainResult(
        prior_id=prior.prior_id,
        beta=beta,
        gain=gain,
        delta=delta,
        se=se,
        chi=chi,
        accepted=accepted,
        reason=reason,
    )


def _twin_match_score(r_a: object, r_b: object) -> float:
    """非核心变量孪生匹配度（模型簇、成本、集成）。"""
    from prior_enhance.prior.compiler import recipe_feature_vector

    fa = recipe_feature_vector(r_a)
    fb = recipe_feature_vector(r_b)
    keys = ("cluster_code", "use_tsfresh", "enable_ensemble")
    diffs = [abs(fa[k] - fb[k]) for k in keys]
    return float(1.0 - min(1.0, sum(diffs) / max(len(keys), 1)))


def compute_contrastive_obs_gain(
    prior: StructuredPrior,
    records: List[dict],
    *,
    phi_gap: float = 0.15,
    delta_max: float = 2.0,
    eps_g: float = 0.1,
    min_pairs: int = 2,
    chi_min: float = 0.3,
) -> GainResult:
    """
    配对断层扫描（Contrastive Recipe Pairs）：
    G^{obs} = E[J(r^-) - J(r^+)]，其中 φ(r^+) - φ(r^-) >= phi_gap。
    """
    j_key = "J_obs"
    pairs: List[tuple[float, float]] = []
    phi_vals = [phi_prior_recipe(prior, rec["recipe"]) for rec in records]
    j_vals = [rec["metrics"][j_key] for rec in records]

    for i, rec_i in enumerate(records):
        for j, rec_j in enumerate(records):
            if i == j:
                continue
            phi_i, phi_j = phi_vals[i], phi_vals[j]
            if phi_i - phi_j < phi_gap:
                continue
            twin = _twin_match_score(rec_i["recipe"], rec_j["recipe"])
            if twin < 0.5:
                continue
            pairs.append((j_vals[j], j_vals[i]))  # (r^-, r^+)

    if len(pairs) < min_pairs:
        return GainResult(
            prior_id=prior.prior_id,
            beta=0.0,
            gain=0.0,
            delta=0.0,
            se=float("inf"),
            chi=0.0,
            accepted=False,
            reason="insufficient_pairs",
        )

    diffs = np.array([a - b for a, b in pairs], dtype=float)
    gain = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("inf")
    delta = float(np.clip(gain / (se + eps_g), -delta_max, delta_max))

    pair_phi = []
    pair_j = []
    for i, rec in enumerate(records):
        pair_phi.append(phi_vals[i])
        pair_j.append(j_vals[i])
    chi, reason = evidence_quality(
        pair_phi, pair_j, min_samples=min_pairs, chi_min=chi_min, require_balance=True,
    )
    return GainResult(
        prior_id=prior.prior_id,
        beta=-gain,
        gain=gain,
        delta=delta,
        se=se,
        chi=chi,
        accepted=reason == "ok",
        reason=reason if reason != "ok" else "contrastive_ok",
    )


def update_tau_logit(
    tau_loc: float,
    delta: float,
    chi: float,
    *,
    omega_val: float = 0.2,
    eta_val: float = 0.15,
) -> float:
    """l_{k+1} = (1-ω) l_k + η χ Δ；返回新 τ^loc。"""
    eps = 1e-6
    tau_loc = float(np.clip(tau_loc, eps, 1.0 - eps))
    logit = float(np.log(tau_loc / (1.0 - tau_loc)))
    logit_new = (1.0 - omega_val) * logit + eta_val * chi * delta
    return float(1.0 / (1.0 + np.exp(-logit_new)))
