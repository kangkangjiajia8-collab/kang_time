"""先验合法性检查 Valid(z_{j,m})。"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from prior_enhance.config import PriorConfig
from prior_enhance.prior.confidence import blend_confidence, compute_b_evidence
from prior_enhance.prior.search_space import AUTOAPEX_SEARCH_SPACE
from prior_enhance.schemas import ForecastTask, PriorTargetClusters, StructuredPrior
from prior_enhance.telemetry.run_logger import log

PHASE = "PRIOR"

VALID_COND_LABELS: Set[str] = {
    "low_sample",
    "medium_sample",
    "high_sample",
    "familiar_event",
    "novel_event",
    "stable",
    "moderate_drift",
    "severe_drift",
}

TARGET_FIELD_ALIASES = {
    "window_strategy": "window_strategy",
    "window": "window_strategy",
    "w": "window_strategy",
    "feature_strategy": "feature_strategy",
    "feature": "feature_strategy",
    "f": "feature_strategy",
    "models": "models",
    "model": "models",
    "m": "models",
    "hyperparameter_policy": "hyperparameter_policy",
    "hyperparameter": "hyperparameter_policy",
    "theta": "hyperparameter_policy",
    "ensemble_strategy": "ensemble_strategy",
    "ensemble": "ensemble_strategy",
    "q": "ensemble_strategy",
}

LEAKAGE_KEYWORDS = (
    "future label",
    "use oos",
    "peek future",
    "使用未来真实值",
    "偷看未来",
    "泄露",
)


def _get_field_ci(raw: dict, *names: str):
    if not isinstance(raw, dict):
        return None
    lower_map = {str(k).lower(): v for k, v in raw.items()}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _as_str_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def _parse_cond(raw: dict, default_cond: List[str]) -> List[str]:
    cond_raw = _get_field_ci(raw, "cond") or raw.get("cond")
    if cond_raw is None:
        return list(default_cond)
    if isinstance(cond_raw, str):
        # 兼容逗号分隔或单标签
        parts = [p.strip() for p in cond_raw.replace("，", ",").split(",") if p.strip()]
        return parts or list(default_cond)
    if isinstance(cond_raw, list):
        return [str(x).strip() for x in cond_raw if str(x).strip()]
    return list(default_cond)


def _parse_target(raw: dict) -> PriorTargetClusters:
    target_raw = _get_field_ci(raw, "target") or raw.get("target") or {}
    if not isinstance(target_raw, dict):
        return PriorTargetClusters()

    normalized = {}
    for key, val in target_raw.items():
        field = TARGET_FIELD_ALIASES.get(str(key).lower().strip())
        if field:
            normalized[field] = _as_str_list(val)

    return PriorTargetClusters(**normalized)


def _parse_action(raw: dict) -> List[str]:
    action_raw = _get_field_ci(raw, "action") or raw.get("action")
    if action_raw is None:
        return []
    if isinstance(action_raw, str):
        return [action_raw.strip()] if action_raw.strip() else []
    return _as_str_list(action_raw)


def _in_search_space(field: str, values: List[str]) -> bool:
    allowed = set(AUTOAPEX_SEARCH_SPACE.get(field, []))
    if not allowed:
        return True
    return all(v in allowed for v in values)


def _target_cluster_count(target: PriorTargetClusters) -> int:
    fields = [
        target.window_strategy,
        target.feature_strategy,
        target.models,
        target.hyperparameter_policy,
        target.ensemble_strategy,
    ]
    return sum(1 for f in fields if f)


def validate_prior(
    raw: dict,
    task: ForecastTask,
    prior_cfg: PriorConfig,
    index: int = 0,
) -> Tuple[Optional[StructuredPrior], Optional[str]]:
    """解析并验证单条 prior。"""
    try:
        cond = _parse_cond(raw, task.context_state.cond_labels)
        target = _parse_target(raw)
        action = _parse_action(raw)
        b_llm = float(_get_field_ci(raw, "b_llm", "confidence", "b") or raw.get("b_llm", 0.5))

        b_evidence = compute_b_evidence(
            task.context_state.n_m,
            task.context_state.nu_m,
            task.episode_count,
            prior_cfg.episode_ref_k,
        )
        confidence = blend_confidence(b_llm, b_evidence, prior_cfg.confidence_lambda)

        prior = StructuredPrior(
            prior_id=str(raw.get("prior_id") or f"prior_{index + 1:03d}"),
            cond=cond,
            target=target,
            action=action,
            b_llm=round(b_llm, 4),
            b_evidence=round(b_evidence, 4),
            confidence=round(confidence, 4),
        )
    except Exception as exc:
        return None, f"schema parse error: {exc}"

    if not prior.cond:
        prior.valid = False
        prior.reject_reason = "empty cond"
        return prior, prior.reject_reason

    invalid_cond = [c for c in prior.cond if c not in VALID_COND_LABELS]
    if invalid_cond:
        prior.valid = False
        prior.reject_reason = f"invalid cond labels: {invalid_cond}"
        return prior, prior.reject_reason

    if _target_cluster_count(prior.target) < 1:
        prior.valid = False
        prior.reject_reason = "target 至少需要一个非空配置簇"
        return prior, prior.reject_reason

    for field_name, values in prior.target.model_dump().items():
        if values and not _in_search_space(field_name, values):
            prior.valid = False
            prior.reject_reason = f"target.{field_name} 含非法候选: {values}"
            return prior, prior.reject_reason

    if not prior.action:
        prior.valid = False
        prior.reject_reason = "action 至少一条 avoid_*"
        return prior, prior.reject_reason

    for act in prior.action:
        if not act.startswith("avoid_"):
            prior.valid = False
            prior.reject_reason = f"action 必须以 avoid_ 开头: {act!r}"
            return prior, prior.reject_reason

    combined = " ".join(prior.cond + prior.action)
    if any(kw in combined.lower() for kw in LEAKAGE_KEYWORDS):
        prior.valid = False
        prior.reject_reason = "potential leakage language"
        return prior, prior.reject_reason

    if prior.b_llm < 0.0 or prior.b_llm > 1.0:
        prior.valid = False
        prior.reject_reason = "b_llm out of range"
        return prior, prior.reject_reason

    return prior, None


def validate_prior_batch(
    raw_priors: List[dict],
    task: ForecastTask,
    prior_cfg: Optional[PriorConfig] = None,
) -> List[StructuredPrior]:
    """批量验证并校准置信度。"""
    prior_cfg = prior_cfg or PriorConfig()
    results: List[StructuredPrior] = []
    seen_ids: Set[str] = set()

    for i, raw in enumerate(raw_priors):
        prior, reason = validate_prior(raw, task, prior_cfg, index=i)
        if prior is None:
            log.debug(f"prior[{i}] 丢弃: {reason}", phase=PHASE, event="prior_rejected")
            continue
        if prior.prior_id in seen_ids:
            prior.valid = False
            prior.reject_reason = "duplicate prior_id"
        seen_ids.add(prior.prior_id)
        if not prior.valid:
            log.debug(
                f"prior[{prior.prior_id}] 无效: {prior.reject_reason}",
                phase=PHASE,
                event="prior_invalid",
            )
        results.append(prior)
    return results
