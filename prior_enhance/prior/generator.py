"""Z_m 生成：LLM → JSON 解析 → Valid 过滤 → 证据校准 b_{j,m}。"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from prior_enhance.config import LLMConfig, PriorConfig
from prior_enhance.prior.coverage_rules import coverage_issues
from prior_enhance.prior.llm_client import DashScopeLLMClient
from prior_enhance.prior.prompt_builder import build_prior_generation_prompt
from prior_enhance.prior.validator import validate_prior_batch
from prior_enhance.schemas import ForecastTask, PriorSet, StructuredPrior
from prior_enhance.telemetry.run_logger import log

PHASE = "PRIOR"

SYSTEM_PROMPT = (
    "你是结构化 recipe 先验生成器。"
    "禁止输出预测数值。"
    "输出 {min_priors}~{max_priors} 条 prior，格式为 cond/target/action/b_llm。"
    "target 为 AutoApex 搜索空间内的配置簇；action 为 avoid_* 规避列表。"
    "最终回复只能是单个 JSON 对象，不要 markdown。"
)


def _sanitize_json_text(text: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f]", " ", text)


def _relax_json(text: str) -> str:
    text = _sanitize_json_text(text)
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_json_object(text: str) -> dict:
    text = _sanitize_json_text(text.strip())
    candidates: List[str] = []

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())
    elif text.startswith("```"):
        candidates.append(re.sub(r"\s*```\s*$", "", re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)).strip())

    start = text.find("{")
    if start >= 0:
        candidates.append(text[start:])

    priors_match = re.search(r'\{\s*"priors"\s*:\s*\[[\s\S]*\]\s*\}', text)
    if priors_match:
        candidates.insert(0, priors_match.group(0))

    last_err: Optional[Exception] = None
    for raw in candidates:
        relaxed = _relax_json(raw)
        try:
            obj, _ = json.JSONDecoder().raw_decode(relaxed)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as exc:
            last_err = exc
            continue

    if last_err:
        raise last_err
    raise json.JSONDecodeError("未找到 JSON 对象", text, 0)


def _build_retry_hint(
    valid: List[StructuredPrior],
    raw_count: int,
    llm_cfg: LLMConfig,
    reject_summary: Optional[List[str]] = None,
) -> str:
    issues = coverage_issues(valid, llm_cfg.min_priors, llm_cfg.max_priors)
    reject_part = ""
    if reject_summary:
        reject_part = f" 拒绝原因: {'; '.join(reject_summary[:5])}。"
    return (
        f"上次返回 {raw_count} 条 raw / {len(valid)} 条有效。"
        f"问题: {'; '.join(issues) if issues else '格式不合格'}。"
        f"{reject_part}"
        "请输出 cond 标签数组、target 配置簇、action avoid_* 列表、b_llm。"
    )


def _parse_and_validate(
    raw_response: str,
    task: ForecastTask,
    prior_cfg: PriorConfig,
) -> Tuple[List[dict], List[StructuredPrior], List[StructuredPrior]]:
    payload = _extract_json_object(raw_response)
    raw_list = payload.get("priors") or payload.get("Z_m") or []
    if isinstance(raw_list, dict):
        raw_list = list(raw_list.values())
    if not isinstance(raw_list, list) or not raw_list:
        raise ValueError("LLM JSON 中缺少非空的 priors 数组")

    priors = validate_prior_batch(raw_list, task, prior_cfg)
    valid = [p for p in priors if p.valid]
    return raw_list, priors, valid


def generate_priors_for_task(
    task: ForecastTask,
    llm_client: Optional[DashScopeLLMClient] = None,
    llm_config: Optional[LLMConfig] = None,
    prior_config: Optional[PriorConfig] = None,
) -> PriorSet:
    """对单个 T_m 调用大模型生成 Z_m。"""
    llm_cfg = llm_config or LLMConfig()
    prior_cfg = prior_config or PriorConfig()
    client = llm_client or DashScopeLLMClient(llm_cfg)

    system = SYSTEM_PROMPT.format(
        min_priors=llm_cfg.min_priors,
        max_priors=llm_cfg.max_priors,
    )
    messages: List[dict] = [{"role": "system", "content": system}]

    raw_response = ""
    prompt = ""
    raw_list: List[dict] = []
    priors: List[StructuredPrior] = []
    valid: List[StructuredPrior] = []

    for attempt in range(llm_cfg.max_retries + 1):
        retry_hint = None
        if attempt > 0:
            reject_summary = [
                f"{p.prior_id}:{p.reject_reason}"
                for p in priors
                if not p.valid and p.reject_reason
            ]
            retry_hint = _build_retry_hint(valid, len(raw_list), llm_cfg, reject_summary)

        prompt = build_prior_generation_prompt(
            task, llm_cfg, prior_cfg, retry_hint=retry_hint
        )
        log.log_output("llm_prompt_chars", {"chars": len(prompt), "attempt": attempt}, phase=PHASE)

        if attempt == 0:
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": "user", "content": retry_hint or prompt})

        raw_response = client.chat(messages)
        log.log_output(
            "llm_raw_response_chars",
            {"chars": len(raw_response or ""), "attempt": attempt},
            phase=PHASE,
        )

        if not raw_response or not raw_response.strip():
            raise ValueError("LLM 返回空响应，请检查 API Key 与模型配置")

        try:
            raw_list, priors, valid = _parse_and_validate(raw_response, task, prior_cfg)
        except json.JSONDecodeError as exc:
            if attempt < llm_cfg.max_retries:
                log.warning(f"JSON 解析失败，重试: {exc}", phase=PHASE, event="llm_retry")
                continue
            raise ValueError(f"LLM 响应无法解析为 JSON: {exc}") from exc

        log.log_transform(
            "llm_json → validated_priors",
            before={"raw_count": len(raw_list), "attempt": attempt},
            after={"validated_count": len(priors), "valid_count": len(valid)},
            phase=PHASE,
        )

        if valid and not coverage_issues(valid, llm_cfg.min_priors, llm_cfg.max_priors):
            break
        if attempt < llm_cfg.max_retries:
            log.warning(
                f"prior 数量不足，重试 ({attempt + 1}/{llm_cfg.max_retries})",
                phase=PHASE,
                event="llm_retry",
            )

    if not valid:
        _save_failed_response(raw_response)
        raise ValueError(
            f"LLM JSON 含 {len(raw_list)} 条 prior，校验后无有效条目。"
            " 常见原因: cond 标签非法、target 不在搜索空间、action 非 avoid_*。"
        )

    remaining = coverage_issues(valid, llm_cfg.min_priors, llm_cfg.max_priors)
    if remaining:
        _save_failed_response(raw_response)
        raise ValueError(f"有效 prior {len(valid)} 条，未达标: {remaining}")

    return PriorSet(
        window_index=task.window_index,
        decision_time=task.decision_time,
        priors=priors,
        raw_llm_response=raw_response,
        prompt_snapshot=prompt,
    )


def _save_failed_response(raw_response: str) -> None:
    fail_log = Path(__file__).resolve().parents[1] / "outputs" / "logs" / "llm_last_failed_response.txt"
    try:
        os.makedirs(fail_log.parent, exist_ok=True)
        with open(fail_log, "w", encoding="utf-8") as f:
            f.write(raw_response)
        log.log_artifact("llm_last_failed_response", str(fail_log), phase=PHASE)
    except OSError:
        pass
