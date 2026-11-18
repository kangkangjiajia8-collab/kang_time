"""时间频率推断与统计分析参数匹配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

# 标准频率码（与 pandas offset 对齐）
STANDARD_FREQ_CODES = ("5T", "15T", "h", "D", "W-SUN", "M")

STANDARD_DELTAS: Dict[str, pd.Timedelta] = {
    "5T": pd.Timedelta("5min"),
    "15T": pd.Timedelta("15min"),
    "h": pd.Timedelta("1h"),
    "D": pd.Timedelta("1D"),
    "W-SUN": pd.Timedelta("7D"),
    "M": pd.Timedelta("30D"),
}


@dataclass(frozen=True)
class FreqProfile:
    """某一标准频率下的分析参数。"""

    code: str
    label: str
    delta: pd.Timedelta
    seasonal_lag: int  # 主周期 lag（如日周期）
    sub_seasonal_lag: Optional[int]  # 次周期 lag（如周周期）
    recent_window: int  # 近期段长度（点数）
    ref_n: float  # n_m 归一化参考样本量
    drift_recent_ratio: float = 0.1


FREQ_PROFILES: Dict[str, FreqProfile] = {
    "5T": FreqProfile(
        code="5T",
        label="5分钟",
        delta=pd.Timedelta("5min"),
        seasonal_lag=288,  # 1 天
        sub_seasonal_lag=288 * 7,
        recent_window=288 * 7,
        ref_n=10_000.0,
    ),
    "15T": FreqProfile(
        code="15T",
        label="15分钟",
        delta=pd.Timedelta("15min"),
        seasonal_lag=96,
        sub_seasonal_lag=96 * 7,
        recent_window=96 * 7,
        ref_n=5_000.0,
    ),
    "h": FreqProfile(
        code="h",
        label="小时",
        delta=pd.Timedelta("1h"),
        seasonal_lag=24,
        sub_seasonal_lag=168,
        recent_window=168,
        ref_n=2_000.0,
    ),
    "D": FreqProfile(
        code="D",
        label="天",
        delta=pd.Timedelta("1D"),
        seasonal_lag=7,
        sub_seasonal_lag=30,
        recent_window=30,
        ref_n=365.0,
    ),
    "W-SUN": FreqProfile(
        code="W-SUN",
        label="周",
        delta=pd.Timedelta("7D"),
        seasonal_lag=52,
        sub_seasonal_lag=104,
        recent_window=26,
        ref_n=104.0,
    ),
    "M": FreqProfile(
        code="M",
        label="月",
        delta=pd.Timedelta("30D"),
        seasonal_lag=12,
        sub_seasonal_lag=24,
        recent_window=12,
        ref_n=60.0,
    ),
}


def infer_freq_code(
    ts: pd.Series,
    override: Optional[str] = None,
) -> str:
    """
    从时间戳推断标准频率码。

    支持: 5T, 15T, h, D, W-SUN, M
    override 可显式指定（来自配置或 CLI）。
    """
    if override and override in FREQ_PROFILES:
        return override

    diffs = pd.to_datetime(ts).diff().dropna()
    if diffs.empty:
        return "h"

    median = diffs.median()
    best_code = "h"
    best_err = float("inf")
    for code, delta in STANDARD_DELTAS.items():
        err = abs((median - delta).total_seconds())
        if err < best_err:
            best_err = err
            best_code = code

    expected_sec = STANDARD_DELTAS[best_code].total_seconds()
    if expected_sec > 0 and best_err / expected_sec > 0.20:
        return "h"
    return best_code


def get_freq_profile(code: str) -> FreqProfile:
    return FREQ_PROFILES.get(code, FREQ_PROFILES["h"])


def recent_window_size(n: int, profile: FreqProfile) -> int:
    """近期段长度：频率默认窗口与序列长度比例取较大者。"""
    ratio_k = max(1, int(n * profile.drift_recent_ratio))
    return min(n - 1, max(profile.recent_window, ratio_k)) if n > 1 else 1
