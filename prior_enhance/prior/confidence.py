"""先验置信度：b_{j,m} = λ b^LLM + (1-λ) b^evidence。"""

from __future__ import annotations


def compute_b_evidence(
    n_m: float,
    nu_m: float,
    episode_count: int,
    episode_ref_k: int = 10,
    *,
    w_n: float = 0.4,
    w_nu: float = 0.3,
    w_ep: float = 0.3,
) -> float:
    """
    b^{evidence} = 0.4 n_m + 0.3 (1-ν_m) + 0.3 min(|D_m|/K, 1)
    """
    ep_term = min(max(episode_count, 0) / max(episode_ref_k, 1), 1.0)
    b = w_n * n_m + w_nu * (1.0 - nu_m) + w_ep * ep_term
    return float(max(0.0, min(1.0, b)))


def blend_confidence(b_llm: float, b_evidence: float, lambda_blend: float = 0.5) -> float:
    lam = max(0.0, min(1.0, lambda_blend))
    b = lam * b_llm + (1.0 - lam) * b_evidence
    return float(max(0.0, min(1.0, b)))
