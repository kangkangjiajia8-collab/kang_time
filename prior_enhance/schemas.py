"""与论文符号对齐的核心数据结构。"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class EventItem(BaseModel):
    """单条事件（E_m.events 元素）。"""

    event_id: str
    event_time: Optional[datetime] = None
    source: str = "unknown"
    event_type: str = ""
    title: str = ""
    content: str = ""
    region: Optional[str] = None
    confidence: Optional[float] = None


class EventContext(BaseModel):
    """事件上下文 E_m。"""

    events: List[EventItem] = Field(default_factory=list)
    event_summary: Optional[str] = None

    def to_prompt_text(self) -> str:
        if self.event_summary:
            return self.event_summary
        parts = []
        for e in self.events:
            line = e.title.strip()
            if e.content:
                line = f"{line}：{e.content.strip()}" if line else e.content.strip()
            if line:
                parts.append(line)
        return "\n".join(parts) if parts else ""


class PriorTargetClusters(BaseModel):
    """
    target_{j,m}：推荐的 recipe 配置簇（正向先验）。

    对应论文 window / feature / models / hyperparameter / ensemble 五类候选集合。
    """

    window_strategy: List[str] = Field(default_factory=list)
    feature_strategy: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    hyperparameter_policy: List[str] = Field(default_factory=list)
    ensemble_strategy: List[str] = Field(default_factory=list)


class StructuredPrior(BaseModel):
    """
    结构化先验 z_{j,m} = (cond, target, action, b)。

    当前阶段不含 effect / failure / feature（后续 φ 编译阶段再扩展）。
    """

    prior_id: str
    cond: List[str] = Field(..., description="适用条件：S_m 离散标签组合")
    target: PriorTargetClusters = Field(..., description="推荐配置簇")
    action: List[str] = Field(..., description="负向规避动作 avoid_* 列表")
    b_llm: float = Field(..., ge=0.0, le=1.0, description="LLM 自报置信度")
    b_evidence: float = Field(0.0, ge=0.0, le=1.0, description="上下文证据置信度")
    confidence: float = Field(..., ge=0.0, le=1.0, description="校准后置信度 b_{j,m}")
    valid: bool = True
    reject_reason: Optional[str] = None

    @property
    def b(self) -> float:
        return self.confidence


class AnalysisDimensions(BaseModel):
    """辅助统计诊断（非 S_m 核心三维，供 prompt 参考）。"""

    freq_code: str = "h"
    seasonal_lag: int = 24
    sub_seasonal_lag: Optional[int] = 168
    periodicity: float = 0.0
    sub_periodicity: float = 0.0
    trend: float = 0.0
    stationarity: float = 0.0
    persistence: float = 0.0
    volatility_cluster: float = 0.0
    outlier_rate: float = 0.0
    jump_intensity: float = 0.0
    cv: float = 0.0


class ContextState(BaseModel):
    """
    上下文状态 S_m = (n_m, ν_m, d_m, ξ_m) ∈ [0,1]^4。

    ξ_m 由序列风险诊断（波动聚集、异常率、跳变强度）估计。
    """

    n_m: float = Field(..., ge=0.0, le=1.0, description="有效样本量")
    nu_m: float = Field(..., ge=0.0, le=1.0, description="事件新颖性")
    d_m: float = Field(..., ge=0.0, le=1.0, description="分布漂移强度")
    xi_m: float = Field(0.5, ge=0.0, le=1.0, description="运行风险等级")
    d_m_mu: float = Field(0.0, ge=0.0, le=1.0, description="均值漂移分量")
    d_m_sigma: float = Field(0.0, ge=0.0, le=1.0, description="波动漂移分量")
    n_label: str = Field("", description="low_sample | medium_sample | high_sample")
    nu_label: str = Field("", description="familiar_event | novel_event")
    d_label: str = Field("", description="stable | moderate_drift | severe_drift")
    cond_labels: List[str] = Field(default_factory=list, description="cond 默认标签三元组")
    scenario: str = Field("", description="当前场景标签 c_m")
    n_eff: int = Field(0, description="N_m^eff 有效同类场景样本数")
    n_ref: int = Field(0, description="N_m^ref 参考样本数")
    s_max: float = Field(0.0, ge=0.0, le=1.0, description="事件最大语义相似度")
    analysis: Optional[AnalysisDimensions] = None


class RetrievedEpisodeSummary(BaseModel):
    """Retrieve(T_m, D) 返回的 episode 摘要（不含 embedding）。"""

    episode_id: str
    scenario: str
    decision_time: datetime
    event_summary: str = ""
    retrieval_score: Optional[float] = None


class SeriesStats(BaseModel):
    """从历史 H_m 提取的统计摘要。"""

    n_points: int
    start_time: datetime
    end_time: datetime
    freq: Optional[str] = None
    freq_code: Optional[str] = None
    mean: float
    std: float
    min_val: float
    max_val: float
    missing_rate: float
    periodicity: float = 0.0
    trend: float = 0.0
    stationarity: float = 0.0
    recent_mean: float = 0.0
    recent_std: float = 0.0
    baseline_mean: float = 0.0
    baseline_std: float = 0.0


class ForecastTask(BaseModel):
    """
    第 m 个窗口的预测任务 T_m = (H_m, X_m+, E_m, K_m, H, B, l)。
    """

    window_index: int
    decision_time: datetime
    horizon: int
    search_budget: int
    loss_name: str
    history_path: str
    history_rows: int
    oos_path: Optional[str] = None
    oos_rows: int = 0
    known_covariates: List[str] = Field(default_factory=list)
    event_text: Optional[str] = None
    event_context: Optional[EventContext] = None
    domain_knowledge: str = ""
    target_col: str = "value"
    item_id: str = "default_item"
    episode_count: int = Field(0, description="|D_m| 检索 episode 数量")
    retrieved_episodes: List[RetrievedEpisodeSummary] = Field(
        default_factory=list,
        description="D_m = Retrieve(T_m, D) 检索结果摘要",
    )
    stats: SeriesStats
    context_state: ContextState


class PriorSet(BaseModel):
    """Z_m = {z_{j,m}}。"""

    window_index: int
    decision_time: datetime
    priors: List[StructuredPrior] = Field(default_factory=list)
    raw_llm_response: Optional[str] = None
    prompt_snapshot: Optional[str] = None

    @field_validator("priors", mode="before")
    @classmethod
    def _ensure_list(cls, v):
        return v or []

    def valid_priors(self) -> List[StructuredPrior]:
        return [p for p in self.priors if p.valid]


class SearchResult(BaseModel):
    """窗口 m 局部搜索输出：r_m* 与 τ^loc（不写持久 τ）。"""

    window_index: int
    decision_time: datetime
    n_evaluated: int
    n_rounds: int
    best_recipe_id: str
    best_j_val: float
    tau_loc_final: dict = Field(default_factory=dict)
    evaluated_summary: List[dict] = Field(default_factory=list)
    retain_ids: List[str] = Field(default_factory=list)


class ObservationResult(BaseModel):
    """O_m 解锁后的 Observation Feedback 与持久记忆更新。"""

    window_index: int
    decision_time: datetime
    n_retain: int
    observation_summary: List[dict] = Field(default_factory=list)
    gain_summary: List[dict] = Field(default_factory=list)
    best_obs_recipe_id: Optional[str] = None
    best_j_obs: Optional[float] = None
    memory_tau: dict = Field(default_factory=dict)
    lambda_m: float = 0.5
    oos_forecasts: List[dict] = Field(default_factory=list)
