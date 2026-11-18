"""Forecasting Recipe r = (W, F, M, θ, V, Q) 数据结构。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from prior_enhance.recipe.catalog import ModelCluster


class SampleWindowSpec(BaseModel):
    """W: 样本选择 / 历史窗口策略。"""

    strategy: str = "full_history_unweighted"
    recent_ratio: float = Field(default=1.0, ge=0.05, le=1.0)
    min_history_points: int = Field(default=48, ge=1)


class FeatureEngineeringSpec(BaseModel):
    """F: 特征工程（AutoApex tsfresh + fe_template）。"""

    mode: str = "none"  # none | tsfresh_comprehensive | tsfresh_custom
    use_tsfresh: bool = False
    feature_config: Optional[List[Dict[str, Any]]] = None
    # 示例: [{"name": "autocorrelation", "params": {"lag": {"start": 1, "end": 10}}}]


class ModelSpec(BaseModel):
    """M: 底层模型族 + 簇。"""

    name: str
    cluster: ModelCluster


class HyperparameterSpec(BaseModel):
    """θ: 模型超参数（对应 hyperparameters JSON 中单个模型的 params）。"""

    model_name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ValidationSpec(BaseModel):
    """V: 内部验证策略（当前预留，默认 AutoApex num_val_windows）。"""

    enabled: bool = False
    num_val_windows: Optional[int] = None


class OutputAndEnsembleSpec(BaseModel):
    """
    Q: 概率预测模式 + 集成策略。

    对应 train API: use_quantile, enable_ensemble, ensemble_types, best_model
    """

    output_mode: str = "point"  # point | quantile_interval
    enable_ensemble: bool = True
    ensemble_types: List[str] = Field(default_factory=lambda: ["weighted"])
    best_model_policy: str = "ensemble_if_enabled"  # single | ensemble_if_enabled
    preferred_single_model: Optional[str] = None


class ForecastingRecipe(BaseModel):
    """
    预测方案 r = (W, F, M, θ, V, Q) ∈ R_m。

    每个 recipe 可编译为 AutoApex /api/train 请求 payload。
    """

    recipe_id: str
    W: SampleWindowSpec = Field(default_factory=SampleWindowSpec)
    F: FeatureEngineeringSpec = Field(default_factory=FeatureEngineeringSpec)
    M: ModelSpec
    theta: HyperparameterSpec
    V: ValidationSpec = Field(default_factory=ValidationSpec)
    Q: OutputAndEnsembleSpec = Field(default_factory=OutputAndEnsembleSpec)

    # 训练任务级参数（非 r 六元组内部，但 recipe 执行时需要）
    prediction_length: int = 168
    freq: Optional[str] = None
    eval_metric: Optional[str] = None
    time_limit: int = 600
    hyperparameter_tune_kwargs: Optional[Dict[str, Any]] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_id(payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def summary(self) -> str:
        ens = ",".join(self.Q.ensemble_types) if self.Q.enable_ensemble else "none"
        return (
            f"[{self.recipe_id}] M={self.M.name}({self.M.cluster.value}) "
            f"W={self.W.strategy} F={self.F.mode} Q={self.Q.output_mode}+{ens}"
        )
