"""上下文状态 S_m = Gamma(T_m, D_m) 的统计估计。"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from prior_enhance.config import DataConfig
from prior_enhance.context.event_novelty import compute_event_novelty
from prior_enhance.context.frequency import (
    FreqProfile,
    get_freq_profile,
    infer_freq_code,
    recent_window_size,
)
from prior_enhance.context.labels import build_cond_labels
from prior_enhance.context.scenario import (
    scenario_label,
    scenario_labels_for_series,
    scenario_ref_count,
)
from prior_enhance.context.text_encoder import ChineseTextEncoder, get_text_encoder
from prior_enhance.data.episode_store import (
    Episode,
    historical_event_texts_for_novelty,
)
from prior_enhance.schemas import AnalysisDimensions, ContextState, EventContext, SeriesStats
from prior_enhance.telemetry.run_logger import log

PHASE = "CONTEXT"


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _drift_component(a: np.ndarray, b: np.ndarray, delta: float = 1.0) -> float:
    """标准化后均值差异，clip 到 [0,1]。"""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    return _clip01(abs(float(np.mean(a) - np.mean(b))) / delta)


def _vol_drift_component(a: np.ndarray, b: np.ndarray, delta: float = 1.0) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    return _clip01(abs(float(np.std(a, ddof=1) - np.std(b, ddof=1))) / delta)


def _acf_at_lag(y: np.ndarray, lag: int) -> float:
    """绝对自相关，归一化到 [0,1]。"""
    if lag <= 0 or len(y) <= lag + 2:
        return 0.0
    a = y[lag:].astype(float)
    b = y[:-lag].astype(float)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-8
    return float(np.clip(abs(np.sum(a * b) / denom), 0.0, 1.0))


def _linear_trend_strength(y: np.ndarray) -> float:
    n = len(y)
    if n < 4:
        return 0.0
    x = np.arange(n, dtype=float)
    coef = np.polyfit(x, y.astype(float), 1)
    trend_line = np.polyval(coef, x)
    ss_res = np.sum((y - trend_line) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-8
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


def _variance_stationarity(y: np.ndarray) -> float:
    n = len(y)
    if n < 4:
        return 0.5
    mid = n // 2
    v1 = np.var(y[:mid]) + 1e-8
    v2 = np.var(y[mid:]) + 1e-8
    return float(np.clip(1.0 - abs(v1 - v2) / (v1 + v2), 0.0, 1.0))


def _volatility_clustering(y: np.ndarray) -> float:
    if len(y) < 4:
        return 0.0
    dy = np.diff(y.astype(float))
    dy2 = dy ** 2
    return _acf_at_lag(dy2, 1)


def _outlier_rate(y: np.ndarray, sigma: float = 3.0) -> float:
    if len(y) < 3:
        return 0.0
    std = np.std(y, ddof=1) + 1e-8
    z = np.abs(y - np.mean(y)) / std
    return float(np.clip(np.mean(z > sigma), 0.0, 1.0))


def _jump_intensity(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    denom = np.abs(y[:-1].astype(float)) + 1e-8
    pct = np.abs(np.diff(y.astype(float))) / denom
    p95 = float(np.percentile(pct, 95))
    return float(np.clip(p95 / 0.5, 0.0, 1.0))


def _is_valid_y(val: float) -> bool:
    if pd.isna(val) or np.isinf(val):
        return False
    return val >= 0


def _scene_normalize(y: np.ndarray, scene_mean: float, scene_std: float) -> np.ndarray:
    """场景内标准化序列 tilde{y}。"""
    std = scene_std if scene_std > 1e-8 else 1.0
    return (y - scene_mean) / std


def compute_analysis_dimensions(
    y: np.ndarray,
    profile: FreqProfile,
) -> AnalysisDimensions:
    """按频率 profile 计算多维统计诊断。"""
    y = np.asarray(y, dtype=float)
    periodicity = _acf_at_lag(y, profile.seasonal_lag)
    sub_periodicity = (
        _acf_at_lag(y, profile.sub_seasonal_lag)
        if profile.sub_seasonal_lag and len(y) > profile.sub_seasonal_lag + 2
        else 0.0
    )
    mean = float(np.mean(y)) if len(y) else 0.0
    std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    cv = std / (abs(mean) + 1e-8)

    return AnalysisDimensions(
        freq_code=profile.code,
        seasonal_lag=profile.seasonal_lag,
        sub_seasonal_lag=profile.sub_seasonal_lag,
        periodicity=round(periodicity, 4),
        sub_periodicity=round(sub_periodicity, 4),
        trend=round(_linear_trend_strength(y), 4),
        stationarity=round(_variance_stationarity(y), 4),
        persistence=round(_acf_at_lag(y, 1), 4),
        volatility_cluster=round(_volatility_clustering(y), 4),
        outlier_rate=round(_outlier_rate(y), 4),
        jump_intensity=round(_jump_intensity(y), 4),
        cv=round(cv, 4),
    )


def _numpy_heterogeneity(
    history_df: pd.DataFrame,
    cfg: DataConfig,
    profile: FreqProfile,
) -> tuple[float, float, float]:
    y = history_df[cfg.target_col].astype(float).values
    periodicity = _acf_at_lag(y, profile.seasonal_lag)
    trend = _linear_trend_strength(y)
    stationarity = _variance_stationarity(y)
    return periodicity, trend, stationarity


def _try_autoapex_heterogeneity(
    history_df: pd.DataFrame,
    cfg: DataConfig,
    profile: FreqProfile,
) -> tuple[float, float, float]:
    import os

    if os.getenv("PRIOR_ENHANCE_SKIP_AUTOAPEX_STATS", "").lower() in ("1", "true", "yes"):
        return _numpy_heterogeneity(history_df, cfg, profile)

    try:
        from prior_enhance._path import setup_autoapex_path

        setup_autoapex_path()
        from autoapex.timeseries import TimeSeriesDataFrame

        flat = history_df[[cfg.id_col, cfg.time_col, cfg.target_col]].copy()
        flat[cfg.time_col] = pd.to_datetime(flat[cfg.time_col])
        tsdf = TimeSeriesDataFrame.from_data_frame(
            flat, id_column=cfg.id_col, timestamp_column=cfg.time_col
        )
        scores = tsdf._compute_heterogeneity_scores(target=cfg.target_col)
        return (
            float(scores.get("periodicity", 0.0)),
            float(scores.get("trend", 0.0)),
            float(scores.get("stationarity", 0.0)),
        )
    except Exception:
        return _numpy_heterogeneity(history_df, cfg, profile)


def compute_series_stats(
    history_df: pd.DataFrame,
    cfg: Optional[DataConfig] = None,
) -> SeriesStats:
    """从 H_m 提取 LLM prompt 与 S_m 所需的统计摘要。"""
    cfg = cfg or DataConfig()
    df = history_df.sort_values(cfg.time_col)
    y = df[cfg.target_col].astype(float)
    ts = pd.to_datetime(df[cfg.time_col])

    freq_code = infer_freq_code(ts, override=cfg.freq)
    profile = get_freq_profile(freq_code)
    n = len(y)
    recent_k = recent_window_size(n, profile)
    recent = y.iloc[-recent_k:].values
    baseline = y.iloc[: max(n - recent_k, 1)].values

    missing_rate = float(y.isna().mean())
    periodicity, trend, stationarity = _try_autoapex_heterogeneity(df, cfg, profile)

    raw_freq = None
    if len(ts) >= 3:
        diffs = ts.diff().dropna()
        if not diffs.empty:
            raw_freq = str(diffs.mode().iloc[0])

    stats = SeriesStats(
        n_points=n,
        start_time=ts.iloc[0].to_pydatetime(),
        end_time=ts.iloc[-1].to_pydatetime(),
        freq=raw_freq,
        freq_code=freq_code,
        mean=float(y.mean()),
        std=float(y.std(ddof=1) if n > 1 else 0.0),
        min_val=float(y.min()),
        max_val=float(y.max()),
        missing_rate=missing_rate,
        periodicity=periodicity,
        trend=trend,
        stationarity=stationarity,
        recent_mean=float(np.mean(recent)),
        recent_std=float(np.std(recent, ddof=1) if len(recent) > 1 else 0.0),
        baseline_mean=float(np.mean(baseline)) if len(baseline) else float(y.mean()),
        baseline_std=float(np.std(baseline, ddof=1) if len(baseline) > 1 else 0.0),
    )
    log.log_output("series_stats", stats.model_dump(), phase=PHASE)
    return stats


def _compute_n_m(
    history_df: pd.DataFrame,
    cfg: DataConfig,
    c_m: str,
    horizon: int,
) -> tuple[float, int, int]:
    """n_m = clip(N_m^eff / N_m^ref, 0, 1)。"""
    ts = pd.to_datetime(history_df[cfg.time_col])
    y = history_df[cfg.target_col].astype(float)
    labels = scenario_labels_for_series(ts)
    valid = y.apply(_is_valid_y)
    n_eff = int(((labels == c_m) & valid).sum())
    _, n_ref = scenario_ref_count(ts, c_m, horizon=horizon)
    n_m = _clip01(n_eff / max(n_ref, 1))
    return n_m, n_eff, n_ref


def _compute_d_m(
    history_df: pd.DataFrame,
    cfg: DataConfig,
    profile: FreqProfile,
    c_m: str,
) -> tuple[float, float, float]:
    """
    d_m = d_m^μ + d_m^σ。

    R_m: 近期观测区间；C_m: 历史同类场景可比样本（场景内标准化后比较）。
    """
    df = history_df.sort_values(cfg.time_col)
    y = df[cfg.target_col].astype(float).values
    ts = pd.to_datetime(df[cfg.time_col])

    recent_k = recent_window_size(len(y), profile)
    recent_idx = np.arange(max(0, len(y) - recent_k), len(y))

    scene_mask = scenario_labels_for_series(ts) == c_m
    scene_y = y[scene_mask.values]
    if len(scene_y) < 2:
        scene_y = y
    scene_mean = float(np.mean(scene_y))
    scene_std = float(np.std(scene_y, ddof=1) if len(scene_y) > 1 else 1.0)

    y_tilde = _scene_normalize(y, scene_mean, scene_std)
    recent_z = y_tilde[recent_idx]

    comp_mask = scene_mask.values.copy()
    comp_mask[recent_idx] = False
    comparable_z = y_tilde[comp_mask]
    if len(comparable_z) < 2:
        comp_mask = np.ones(len(y), dtype=bool)
        comp_mask[recent_idx] = False
        comparable_z = y_tilde[comp_mask]
    if len(comparable_z) < 2:
        comparable_z = y_tilde[: max(len(y_tilde) // 2, 1)]

    d_m_mu = _drift_component(recent_z, comparable_z, delta=1.0)
    d_m_sigma = _vol_drift_component(recent_z, comparable_z, delta=1.0)
    d_m = _clip01(d_m_mu + d_m_sigma)
    return d_m, d_m_mu, d_m_sigma


def compute_context_state(
    history_df: pd.DataFrame,
    stats: SeriesStats,
    event_context: Optional[EventContext] = None,
    episodes: Optional[Sequence[Episode]] = None,
    cfg: Optional[DataConfig] = None,
    horizon: int = 72,
    encoder: Optional[ChineseTextEncoder] = None,
) -> ContextState:
    """
    估计 S_m = (n_m, ν_m, d_m) 及离散 cond 标签。

    - n_m: 场景相关有效样本量 clip(N_m^eff / N_m^ref)
    - ν_m: 1 - s_m^max，由中文文本编码器 g(E_m) 计算
    - d_m: d_m^μ + d_m^σ（场景内标准化，近期 vs 历史可比段）
    """
    cfg = cfg or DataConfig()
    df = history_df.sort_values(cfg.time_col)
    y = df[cfg.target_col].astype(float).values
    ts = pd.to_datetime(df[cfg.time_col])

    freq_code = stats.freq_code or infer_freq_code(ts, override=cfg.freq)
    profile = get_freq_profile(freq_code)
    analysis = compute_analysis_dimensions(y, profile)

    analysis = analysis.model_copy(update={
        "periodicity": round(stats.periodicity, 4),
        "trend": round(stats.trend, 4),
        "stationarity": round(stats.stationarity, 4),
    })

    decision_time = ts.iloc[-1]
    c_m = scenario_label(decision_time)
    n_m, n_eff, n_ref = _compute_n_m(df, cfg, c_m, horizon)
    d_m, d_m_mu, d_m_sigma = _compute_d_m(df, cfg, profile, c_m)

    hist_texts: List[str] = []
    if episodes:
        hist_texts = historical_event_texts_for_novelty(
            episodes, same_scenario_only=True, scenario=c_m
        )

    if event_context is not None:
        enc = encoder or get_text_encoder()
        nu_m, s_max = compute_event_novelty(event_context, hist_texts, enc)
    else:
        nu_m, s_max = 0.5, 0.0

    cond_labels, n_label, nu_label, d_label = build_cond_labels(n_m, nu_m, d_m)

    xi_m = _clip01(
        0.4 * analysis.volatility_cluster
        + 0.3 * analysis.outlier_rate
        + 0.3 * analysis.jump_intensity
    )

    ctx = ContextState(
        n_m=round(n_m, 4),
        nu_m=round(float(nu_m), 4),
        d_m=round(d_m, 4),
        xi_m=round(xi_m, 4),
        d_m_mu=round(d_m_mu, 4),
        d_m_sigma=round(d_m_sigma, 4),
        n_label=n_label,
        nu_label=nu_label,
        d_label=d_label,
        cond_labels=cond_labels,
        scenario=c_m,
        n_eff=n_eff,
        n_ref=n_ref,
        s_max=round(s_max, 4),
        analysis=analysis,
    )
    log.log_output("context_state_Sm", ctx.model_dump(), phase=PHASE)
    return ctx
