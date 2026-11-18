"""从 T_m 构造 LLM 结构化 prior 生成 prompt（新版论文四元组）。"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Optional

from prior_enhance.config import LLMConfig, PriorConfig
from prior_enhance.prior.search_space import AUTOAPEX_SEARCH_SPACE
from prior_enhance.schemas import ForecastTask


PRIOR_JSON_EXAMPLE = {
    "priors": [
        {
            "prior_id": "prior_001",
            "cond": ["high_sample", "familiar_event", "moderate_drift"],
            "target": {
                "window_strategy": ["recent_90d", "exp_decay"],
                "feature_strategy": ["calendar", "lag"],
                "models": ["LightGBM", "RecursiveTabular"],
                "hyperparameter_policy": ["default"],
                "ensemble_strategy": ["weighted_average"],
            },
            "action": [
                "avoid_full_history_window",
                "avoid_high_dimensional_tsfresh",
            ],
            "b_llm": 0.72,
        },
        {
            "prior_id": "prior_002",
            "cond": ["high_sample", "familiar_event", "moderate_drift"],
            "target": {
                "window_strategy": ["exp_decay"],
                "feature_strategy": ["tsfresh_statistical"],
                "models": ["AutoARIMA", "ETS"],
                "hyperparameter_policy": ["conservative"],
                "ensemble_strategy": ["none"],
            },
            "action": [
                "avoid_aggressive_hyperparameter_search",
                "avoid_high_capacity_neural_models",
            ],
            "b_llm": 0.68,
        },
    ]
}


def build_prior_generation_prompt(
    task: ForecastTask,
    llm_config: Optional[LLMConfig] = None,
    prior_config: Optional[PriorConfig] = None,
    retry_hint: Optional[str] = None,
) -> str:
    """构造让 LLM 输出 Z_m 的 user prompt。"""
    llm_cfg = llm_config or LLMConfig()
    prior_cfg = prior_config or PriorConfig()
    s = task.stats
    c = task.context_state

    cov_text = (
        ", ".join(task.known_covariates) if task.known_covariates else "（无）"
    )
    event_text = (
        task.event_text.strip()
        if task.event_text
        else (
            task.event_context.to_prompt_text()
            if task.event_context
            else "（无事件文本 E_m）"
        )
    )

    search_space_json = json.dumps(AUTOAPEX_SEARCH_SPACE, ensure_ascii=False, indent=2)
    cond_default = json.dumps(c.cond_labels, ensure_ascii=False)

    stats_block = dedent(
        f"""
        ## 历史统计（截至 t_m，禁止使用 O_m）
        - t_m: {task.decision_time}
        - 历史点数: {s.n_points}, 频率: {s.freq_code or 'unknown'}
        - 均值/标准差: {s.mean:.4f} / {s.std:.4f}
        - 缺失率: {s.missing_rate:.4f}
        """
    ).strip()

    context_block = dedent(
        f"""
        ## 上下文状态 S_m = (n_m, ν_m, d_m)
        - n_m={c.n_m:.3f} → `{c.n_label}` (N_eff={c.n_eff}, N_ref={c.n_ref}, 场景={c.scenario})
        - ν_m={c.nu_m:.3f} → `{c.nu_label}` (s_max={c.s_max:.3f})
        - d_m={c.d_m:.3f} (μ={c.d_m_mu:.3f}, σ={c.d_m_sigma:.3f}) → `{c.d_label}`
        - **默认 cond 标签**（每条 prior 的 cond 必须从此集合选取，可全选或选子集）: {cond_default}
        - 检索 episode 数 |D_m|: {task.episode_count}
        """
    ).strip()

    if c.analysis:
        a = c.analysis
        context_block += dedent(
            f"""

        ### 辅助诊断（可选参考）
        - periodicity={a.periodicity:.3f}, trend={a.trend:.3f}, stationarity={a.stationarity:.3f}
        """
        ).rstrip()

    task_block = dedent(
        f"""
        ## 预测任务 T_m
        - H={task.horizon}, B={task.search_budget}, loss={task.loss_name}
        - X_m+: {cov_text}
        - E_m: {event_text}
        - K_m: {task.domain_knowledge}
        """
    ).strip()

    retry_block = f"\n## ⚠️ 上次不合格\n{retry_hint}\n" if retry_hint else ""

    instructions = dedent(
        f"""
        你是 Forecasting Recipe Search 的结构化先验生成器。

        **核心立场**：LLM generates testable priors, not forecasts.
        - 不输出预测数值；
        - 不指定唯一最终 recipe；
        - 只输出结构化 prior，用于缩小 AutoApex 搜索空间。

        ## 每条 prior 格式 z_{{j,m}} = (cond, target, action, b^LLM)

        1. **cond**：字符串数组，元素只能来自:
           low_sample | medium_sample | high_sample |
           familiar_event | novel_event |
           stable | moderate_drift | severe_drift

        2. **target**：推荐配置簇（正向先验），字段均可为数组，值必须来自下方搜索空间:
           - window_strategy, feature_strategy, models,
             hyperparameter_policy, ensemble_strategy
           每个字段至少 1 个候选。

        3. **action**：负向规避列表（avoid_* 英文 snake_case），描述应**避免**的配置，例如:
           avoid_full_history_window, avoid_high_dimensional_tsfresh,
           avoid_large_ensemble, avoid_aggressive_hyperparameter_search

        4. **b_llm**：LLM 自报置信度 ∈ [0,1]（系统会用 S_m 证据另行校准，你只需报 b_llm）

        ## AutoApex 搜索空间
        {search_space_json}

        ## 输出要求
        - 只输出 JSON 对象 {{"priors": [...]}}，不要 markdown，不要解释；
        - 输出 {llm_cfg.min_priors}~{llm_cfg.max_priors} 条 prior；
        - prior_id 从 prior_001 连续编号；
        - 每条 prior 的 target 至少覆盖 2 类组件（如有 window+models）；
        - action 至少 1 条 avoid_*；
        - 概率校准 Q 模块当前阶段暂不启用，ensemble_strategy 可推荐 weighted_average 或 none。

        JSON 示例:
        {json.dumps(PRIOR_JSON_EXAMPLE, ensure_ascii=False, indent=2)}
        """
    ).strip()

    parts = [task_block, stats_block, context_block]
    if retry_block:
        parts.append(retry_block.strip())
    parts.append(instructions)
    return "\n\n".join(parts)


