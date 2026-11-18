"""结构化 prior 多维相似度 s(z_{j,m}, z_i)（论文 Prior Memory 检索用）。"""

from __future__ import annotations

from typing import List, Optional, Sequence

from prior_enhance.context.text_encoder import ChineseTextEncoder, get_text_encoder
from prior_enhance.schemas import PriorTargetClusters, StructuredPrior


def _overlap_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    """集合重叠度 |A∩B| / max(|A|,|B|,1)。"""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb), 1)


def _target_similarity(t1: PriorTargetClusters, t2: PriorTargetClusters) -> float:
    dims: List[float] = []
    if t1.window_strategy or t2.window_strategy:
        dims.append(_overlap_ratio(t1.window_strategy, t2.window_strategy))
    if t1.feature_strategy or t2.feature_strategy:
        dims.append(_overlap_ratio(t1.feature_strategy, t2.feature_strategy))
    if t1.models or t2.models:
        dims.append(_overlap_ratio(t1.models, t2.models))
    if t1.hyperparameter_policy or t2.hyperparameter_policy:
        dims.append(_overlap_ratio(t1.hyperparameter_policy, t2.hyperparameter_policy))
    if t1.ensemble_strategy or t2.ensemble_strategy:
        dims.append(_overlap_ratio(t1.ensemble_strategy, t2.ensemble_strategy))
    return sum(dims) / len(dims) if dims else 0.0


def prior_to_embed_text(prior: StructuredPrior) -> str:
    """将 prior 序列化为文本，供 g(·) 语义相似度（自然语言字段补充）。"""
    t = prior.target
    parts = [
        " ".join(prior.cond),
        " ".join(t.window_strategy),
        " ".join(t.feature_strategy),
        " ".join(t.models),
        " ".join(t.hyperparameter_policy),
        " ".join(t.ensemble_strategy),
        " ".join(prior.action),
    ]
    return " | ".join(p for p in parts if p.strip())


def prior_structural_similarity(
    z_a: StructuredPrior,
    z_b: StructuredPrior,
    *,
    encoder: Optional[ChineseTextEncoder] = None,
    semantic_weight: float = 0.15,
) -> float:
    """
    s(z_a, z_b) ∈ [0,1]：离散字段规则重叠 + 可选文本编码余弦相似度。

    论文：模型/窗口/特征/超参/集成等离散字段用集合重叠；
    自然语言或难规则化字段用 g(·) 余弦相似度；各维平均。
    """
    dims: List[float] = []
    if z_a.cond or z_b.cond:
        dims.append(_overlap_ratio(z_a.cond, z_b.cond))
    dims.append(_target_similarity(z_a.target, z_b.target))
    if z_a.action or z_b.action:
        dims.append(_overlap_ratio(z_a.action, z_b.action))

    struct_score = sum(dims) / len(dims) if dims else 0.0

    if semantic_weight <= 0.0:
        return float(struct_score)

    enc = encoder or get_text_encoder()
    text_a = prior_to_embed_text(z_a)
    text_b = prior_to_embed_text(z_b)
    if not text_a.strip() or not text_b.strip():
        return float(struct_score)

    vec_a = enc.encode(text_a)
    vec_b = enc.encode(text_b)
    sem = max(0.0, enc.cosine_similarity(vec_a, vec_b))
    w = max(0.0, min(1.0, semantic_weight))
    return float((1.0 - w) * struct_score + w * sem)


def weighted_tau_from_similar_priors(
    query: StructuredPrior,
    candidates: Sequence[tuple[StructuredPrior, float]],
    *,
    s_min: float = 0.25,
    default_tau: float = 0.5,
    epsilon: float = 1e-8,
    encoder: Optional[ChineseTextEncoder] = None,
    semantic_weight: float = 0.15,
) -> tuple[float, int]:
    """
    τ_{j,m} = Σ s·τ_i / (Σ s + ε)；无候选 ≥ s_min 时返回 default_tau。

    candidates: (historical_prior, tau_i) 列表。
    返回 (tau, n_matched)。
    """
    num = 0.0
    den = 0.0
    n_matched = 0
    for hist, tau_i in candidates:
        s = prior_structural_similarity(
            query, hist, encoder=encoder, semantic_weight=semantic_weight,
        )
        if s < s_min:
            continue
        num += s * tau_i
        den += s
        n_matched += 1

    if n_matched == 0 or den < epsilon:
        return float(default_tau), 0
    return float(num / (den + epsilon)), n_matched
