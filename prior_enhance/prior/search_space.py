"""AutoApex 可执行搜索空间词汇（供 LLM target 推荐）。"""

from __future__ import annotations

AUTOAPEX_SEARCH_SPACE = {
    "window_strategy": [
        "full_history_unweighted",
        "recent_90d",
        "recent_50pct",
        "recent_25pct",
        "exp_decay",
        "exponential_decay",
    ],
    "feature_strategy": [
        "none",
        "calendar",
        "lag",
        "rolling",
        "tsfresh_statistical",
        "tsfresh_comprehensive",
        "tsfresh_custom",
    ],
    "models": [
        "SeasonalNaive",
        "AutoARIMA",
        "ETS",
        "DirectTabular",
        "RecursiveTabular",
        "LightGBM",
        "XGBoost",
        "TemporalFusionTransformer",
        "PatchTST",
        "TiDE",
        "DLinear",
        "DeepAR",
        "Chronos2",
        "TABPFNV2",
    ],
    "hyperparameter_policy": [
        "default",
        "conservative",
        "aggressive",
    ],
    "ensemble_strategy": [
        "none",
        "weighted_average",
        "simple_average",
        "median",
    ],
}
