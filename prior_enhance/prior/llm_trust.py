"""全局 LLM 信任权重 λ_m 冷启动（公开排行榜映射）。"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from prior_enhance.prior.llm_leaderboard import (
    SCORE_MAX,
    SCORE_MIN,
    leaderboard_bounds,
    lookup_leaderboard_score,
)


def lambda_m_init_from_leaderboard(
    score: float,
    score_max: float,
    score_min: float,
    *,
    epsilon: float = 1e-8,
) -> float:
    """
    λ_m^{init} = (score_m - score_min) / (score_max - score_min + ε)

    论文：外部公开评测综合能力分归一化至 [0,1]。
    """
    denom = score_max - score_min + epsilon
    if denom <= 0:
        return 0.0
    lam = (score - score_min) / denom
    return float(max(0.0, min(1.0, lam)))


def resolve_lambda_m_init(
    *,
    configured: Optional[float] = None,
    model_name: Optional[str] = None,
    default: Optional[float] = None,
) -> Tuple[float, dict]:
    """
    解析 λ_m 初值：显式配置 > 环境变量 > 排行榜 CSV 查表。

    未命中排行榜时使用 score_min → λ=0（低信任）。

    返回 (lambda_m, meta)。
    """
    meta = {"source": "default", "score": None, "matched": False}

    if configured is not None:
        meta["source"] = "config"
        return float(max(0.0, min(1.0, configured))), meta

    if score_env := os.getenv("PRIOR_ENHANCE_LLM_LEADERBOARD_SCORE"):
        score = float(score_env)
        score_max = float(os.getenv("PRIOR_ENHANCE_LLM_LEADERBOARD_MAX", str(SCORE_MAX)))
        score_min = float(os.getenv("PRIOR_ENHANCE_LLM_LEADERBOARD_MIN", str(SCORE_MIN)))
        meta.update({"source": "env", "score": score, "score_max": score_max, "score_min": score_min})
        return lambda_m_init_from_leaderboard(score, score_max, score_min), meta

    score_max, score_min = leaderboard_bounds()
    score, matched = lookup_leaderboard_score(model_name)
    meta.update({
        "source": "leaderboard_csv" if matched else "leaderboard_fallback_min",
        "score": score,
        "score_max": score_max,
        "score_min": score_min,
        "matched": matched,
        "model_name": model_name,
    })
    lam = lambda_m_init_from_leaderboard(score, score_max, score_min)
    if default is not None and not matched and os.getenv("PRIOR_ENHANCE_LLM_LEADERBOARD_STRICT", "1") not in ("1", "true", "yes"):
        meta["source"] = "default_unmatched"
        return float(default), meta
    return lam, meta
