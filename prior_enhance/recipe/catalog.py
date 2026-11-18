"""
AutoApex Recipe 目录：模型簇、特征工程、超参、集成策略。

Recipe r 来自 AutoApex 可执行配置空间，而非 LLM 臆造。
参考:
  - serve/configs/hpo_template.py
  - serve/configs/fe_template.py
  - serve/api/services/train_service.py (ensemble)
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVE_ROOT = _REPO_ROOT / "serve"
if str(_SERVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVE_ROOT))

try:
    from configs.fe_template import FEATURE_HANDLERS
except ImportError:
    FEATURE_HANDLERS = {}


class ModelCluster(str, Enum):
    """模型簇：在 AutoApex 无原生概念，prior_enhance 统一归类。"""

    STATISTICAL = "statistical"  # 统计模型
    ML_TABULAR = "ml_tabular"  # 机器学习表格化
    DEEP_LEARNING = "deep_learning"  # 深度学习
    FOUNDATION = "foundation"  # 时序大模型
    ENSEMBLE_META = "ensemble_meta"  # 集成元模型（非单模型训练项）


# ---------------------------------------------------------------------------
# M: 模型族目录（name → cluster + 说明）
# ---------------------------------------------------------------------------
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    # --- 统计模型 ---
    "Naive": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "SeasonalNaive": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "Average": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "SeasonalAverage": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "Zero": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "ARIMA": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "AutoARIMA": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "ETS": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "AutoETS": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "AutoCES": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "Theta": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "DynamicOptimizedTheta": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "ADIDA": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "Croston": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "CrostonSBA": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "IMAPA": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "MovingAverageInterpolation": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    "EWMA": {"cluster": ModelCluster.STATISTICAL, "probabilistic": False, "phase": 1},
    # --- 机器学习 ---
    "DirectTabular": {"cluster": ModelCluster.ML_TABULAR, "probabilistic": False, "phase": 1},
    "RecursiveTabular": {"cluster": ModelCluster.ML_TABULAR, "probabilistic": False, "phase": 1},
    "PerStepTabular": {"cluster": ModelCluster.ML_TABULAR, "probabilistic": False, "phase": 1},
    # --- 深度学习 ---
    "DeepAR": {"cluster": ModelCluster.DEEP_LEARNING, "probabilistic": True, "phase": 2},
    "TemporalFusionTransformer": {"cluster": ModelCluster.DEEP_LEARNING, "probabilistic": True, "phase": 2},
    "PatchTST": {"cluster": ModelCluster.DEEP_LEARNING, "probabilistic": False, "phase": 2},
    "TiDE": {"cluster": ModelCluster.DEEP_LEARNING, "probabilistic": True, "phase": 2},
    "DLinear": {"cluster": ModelCluster.DEEP_LEARNING, "probabilistic": False, "phase": 2},
    "WaveNet": {"cluster": ModelCluster.DEEP_LEARNING, "probabilistic": True, "phase": 2},
    "SimpleFeedForward": {"cluster": ModelCluster.DEEP_LEARNING, "probabilistic": False, "phase": 2},
    # --- 时序大模型（AutoApex 运行时常见 Chronos / Chronos2 变体）---
    "Chronos": {"cluster": ModelCluster.FOUNDATION, "probabilistic": True, "phase": 3},
    "Chronos2": {"cluster": ModelCluster.FOUNDATION, "probabilistic": True, "phase": 3},
    # --- 集成元模型（Q 层，不单独作为 M 训练入口）---
    "WeightedEnsemble": {"cluster": ModelCluster.ENSEMBLE_META, "probabilistic": True, "phase": 2},
    "PerItemGreedyEnsemble": {"cluster": ModelCluster.ENSEMBLE_META, "probabilistic": True, "phase": 2},
    "SimpleAverage": {"cluster": ModelCluster.ENSEMBLE_META, "probabilistic": True, "phase": 2},
    "MedianEnsemble": {"cluster": ModelCluster.ENSEMBLE_META, "probabilistic": True, "phase": 2},
}

TRAINABLE_MODELS: Set[str] = {
    n for n, m in MODEL_CATALOG.items() if m["cluster"] != ModelCluster.ENSEMBLE_META
}

# ---------------------------------------------------------------------------
# W: 样本窗口 / 加权策略（prior 层抽象；训练前可截断 history CSV）
# ---------------------------------------------------------------------------
SAMPLE_WINDOW_STRATEGIES = {
    "full_history_unweighted": {
        "description": "使用全部可见历史，等权",
        "recent_ratio": 1.0,
    },
    "recent_75pct": {
        "description": "仅使用最近 75% 历史",
        "recent_ratio": 0.75,
    },
    "recent_50pct": {
        "description": "仅使用最近 50% 历史（漂移场景）",
        "recent_ratio": 0.5,
    },
    "recent_25pct": {
        "description": "仅使用最近 25% 历史（强漂移）",
        "recent_ratio": 0.25,
    },
    "exponential_decay": {
        "description": "概念性指数衰减加权（prior 建议项；AutoApex 训练 API 无原生权重）",
        "recent_ratio": 0.8,
    },
}

# ---------------------------------------------------------------------------
# F: 特征工程（fe_template.FEATURE_HANDLERS + tsfresh comprehensive）
# ---------------------------------------------------------------------------
FEATURE_ENGINEERING_MODES = {
    "none": {"use_tsfresh": False, "feature_config": None},
    "tsfresh_comprehensive": {"use_tsfresh": True, "feature_config": "comprehensive"},
    "tsfresh_custom": {"use_tsfresh": True, "feature_config": "custom"},
}

# 常用特征子集（完整列表见 fe_template.FEATURE_HANDLERS）
DEFAULT_FEATURE_CANDIDATES: List[str] = sorted(FEATURE_HANDLERS.keys()) if FEATURE_HANDLERS else [
    "autocorrelation",
    "partial_autocorrelation",
    "agg_autocorrelation",
    "linear_trend",
    "augmented_dickey_fuller",
    "fft_aggregated",
    "quantile",
    "number_peaks",
]

# ---------------------------------------------------------------------------
# Q: 概率输出 + 集成策略（train_service.get_ensemble_configs）
# ---------------------------------------------------------------------------
ENSEMBLE_TYPES = {
    "weighted": "WeightedEnsemble",
    "per_item": "PerItemGreedyEnsemble",
    "simple": "SimpleAverage",
    "median": "MedianEnsemble",
}

OUTPUT_MODES = {
    "point": {"use_quantile": None, "quantile_levels": [0.5]},
    "quantile_interval": {"use_quantile": "interval", "quantile_levels": [0.1, 0.5, 0.9]},
}

# ---------------------------------------------------------------------------
# theta: 默认 HPO 搜索空间（与 post.py / 用户文档对齐的推荐传参）
# ---------------------------------------------------------------------------
DEFAULT_HPO_SPACES: Dict[str, Dict[str, Any]] = {
    "ARIMA": {
        "order": [(1, 0, 0), (1, 1, 1), (2, 1, 0), (0, 1, 1), (2, 1, 2)],
        "seasonal_order": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
        "method": ["CSS-ML", "CSS", "ML"],
    },
    "AutoARIMA": {
        "max_p": {"start": 3, "end": 8},
        "max_q": {"start": 3, "end": 8},
        "max_P": {"start": 1, "end": 4},
        "max_Q": {"start": 1, "end": 4},
        "max_d": {"start": 0, "end": 2},
        "max_D": {"start": 0, "end": 2},
        "start_p": {"start": 1, "end": 3},
        "seasonal": [True, False],
    },
    "ETS": {"model": ["AAA", "ANA", "MAM", "MNM"], "damped": [True, False]},
    "AutoCES": {"model": ["Z", "N", "S", "P", "F"]},
    "Theta": {"decomposition_type": ["multiplicative", "additive"]},
    "Croston": {"variant": ["SBA", "classic", "optimized"]},
    "MovingAverageInterpolation": {
        "window_size": {"start": 24, "end": 60},
        "max_growth": {"start": 0.15, "end": 0.35},
        "min_growth": {"start": -0.25, "end": -0.05},
        "diff_clip_q": [[0.05, 0.95], [0.1, 0.9]],
        "base_periods": [[1, 2], [1, 2, 3]],
    },
    "TemporalFusionTransformer": {
        "hidden_dim": [32, 64, 128],
        "num_heads": [1, 2, 4],
        "dropout_rate": {"start": 0.1, "end": 0.3},
        "batch_size": [32, 64, 128],
        "lr": {"start": 0.0001, "end": 0.01, "log": True},
    },
    "PatchTST": {
        "patch_len": [8, 16, 24],
        "stride": [4, 8, 16],
        "d_model": [32, 64, 128, 512],
        "nhead": [2, 4, 8],
        "num_encoder_layers": {"start": 2, "end": 4},
        "scaling": ["mean", "std", None],
        "batch_size": [32, 64, 128],
        "lr": {"start": 0.0001, "end": 0.01, "log": True},
    },
    "TiDE": {
        "encoder_hidden_dim": [64, 128, 256],
        "decoder_hidden_dim": [64, 128, 256],
        "temporal_hidden_dim": [64, 128],
        "distr_hidden_dim": [64, 128],
        "num_layers_encoder": {"start": 1, "end": 4},
        "num_layers_decoder": {"start": 1, "end": 4},
        "dropout_rate": {"start": 0.1, "end": 0.3},
        "layer_norm": [True, False],
        "batch_size": [128, 256, 512],
        "lr": {"start": 0.0001, "end": 0.01, "log": True},
    },
    "DLinear": {"hidden_dimension": {"start": 10, "end": 50}},
    "WaveNet": {
        "num_bins": {"start": 512, "end": 1024},
        "use_log_scale_feature": [True, False],
        "negative_data": [True, False],
        "batch_size": [32, 64, 128],
        "lr": {"start": 0.0001, "end": 0.01, "log": True},
        "weight_decay": {"start": 1e-8, "end": 0.0001, "log": True},
    },
    "SimpleFeedForward": {
        "batch_normalization": [True, False],
        "mean_scaling": [True, False],
        "batch_size": [32, 64, 128],
        "lr": {"start": 0.0001, "end": 0.01, "log": True},
    },
    "DirectTabular": {},
    "RecursiveTabular": {},
    "PerStepTabular": {},
    "DeepAR": {},
    "Chronos": {},
    "Chronos2": {},
}


def models_by_cluster(cluster: ModelCluster) -> List[str]:
    return [n for n, m in MODEL_CATALOG.items() if m["cluster"] == cluster]
