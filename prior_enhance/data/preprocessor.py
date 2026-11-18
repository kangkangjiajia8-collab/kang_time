"""数据预处理：优先 AutoApex clean_and_fill，不可用时 pandas 降级。"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from prior_enhance.config import DataConfig
from prior_enhance.telemetry.run_logger import log

PHASE = "PREPROCESS"


def _normalize_columns(
    df: pd.DataFrame,
    cfg: DataConfig,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """将用户 CSV 列名统一为 AutoApex 标准格式。"""
    out = df.copy()
    target = target_col or cfg.target_col

    rename_map = {}
    if "date" in out.columns and cfg.time_col not in out.columns:
        rename_map["date"] = cfg.time_col
    if "time" in out.columns and cfg.time_col not in out.columns:
        rename_map["time"] = cfg.time_col
    if "power" in out.columns and target not in out.columns and target == "value":
        rename_map["power"] = "value"
    if rename_map:
        out = out.rename(columns=rename_map)

    if cfg.id_col not in out.columns:
        out[cfg.id_col] = cfg.default_item_id

    if target != cfg.target_col and target in out.columns:
        out = out.rename(columns={target: cfg.target_col})

    out[cfg.time_col] = pd.to_datetime(out[cfg.time_col], errors="coerce")
    out[cfg.target_col] = pd.to_numeric(out[cfg.target_col], errors="coerce")
    out = out.dropna(subset=[cfg.time_col])
    out = out.sort_values([cfg.id_col, cfg.time_col]).reset_index(drop=True)
    return out


def _fill_values(series: pd.Series, method: str) -> pd.Series:
    if method in ("auto", "ffill", "pad"):
        return series.ffill().bfill()
    if method in ("bfill", "backfill"):
        return series.bfill().ffill()
    if method == "interpolate":
        return series.interpolate(method="linear", limit_direction="both").ffill().bfill()
    return series.fillna(0.0)


def _pandas_clean_and_fill(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    id_col: str,
    fill_method: str,
) -> pd.DataFrame:
    """不依赖 TimeSeriesDataFrame 的轻量清洗（与 AutoApex 语义近似）。"""
    out = df.copy()
    out["_is_origin"] = True
    out["_orig_val_missing"] = out[target_col].isna()

    out = out.sort_values([id_col, time_col])
    out[target_col] = out.groupby(id_col, group_keys=False)[target_col].transform(
        lambda s: _fill_values(s, fill_method)
    )

    out["fill_flag"] = 3
    out.loc[out["_orig_val_missing"], "fill_flag"] = 0
    out = out.drop(columns=["_is_origin", "_orig_val_missing"])
    return out.reset_index(drop=True)


def _try_autoapex_clean_and_fill(
    df: pd.DataFrame,
    cfg: DataConfig,
) -> Optional[pd.DataFrame]:
    try:
        from prior_enhance._path import setup_autoapex_path

        setup_autoapex_path()
        from autoapex.timeseries import TimeSeriesDataFrame

        return TimeSeriesDataFrame.clean_and_fill(
            df,
            time_col=cfg.time_col,
            target_col=cfg.target_col,
            id_col=cfg.id_col,
            fill_method=cfg.fill_method,
        )
    except Exception:
        return None


def preprocess_csv(
    csv_path: str,
    cfg: Optional[DataConfig] = None,
    target_col: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    读取 CSV → 列名标准化 → clean_and_fill → 返回 (flat_df, tsdf_or_None)。

    优先调用 AutoApex TimeSeriesDataFrame.clean_and_fill；
    若环境缺少 rpy2 等依赖则自动降级为 pandas 填补。
    """
    cfg = cfg or DataConfig()
    raw = pd.read_csv(csv_path)
    log.log_input("raw_csv_shape", {"path": csv_path, "rows": len(raw), "columns": list(raw.columns)}, phase=PHASE)

    normalized = _normalize_columns(raw, cfg, target_col=target_col)
    if list(normalized.columns) != list(raw.columns) or normalized.shape != raw.shape:
        log.log_transform(
            "column_normalize",
            before={"columns": list(raw.columns)},
            after={"columns": list(normalized.columns), "rows": len(normalized)},
            phase=PHASE,
        )

    clean_df = _try_autoapex_clean_and_fill(normalized, cfg)
    backend = "autoapex"
    if clean_df is None:
        backend = "pandas_fallback"
        log.warning("AutoApex clean_and_fill 不可用，降级 pandas", phase=PHASE, event="fallback")
        clean_df = _pandas_clean_and_fill(
            normalized,
            cfg.time_col,
            cfg.target_col,
            cfg.id_col,
            cfg.fill_method,
        )

    tsdf = None
    try:
        from prior_enhance._path import setup_autoapex_path

        setup_autoapex_path()
        from autoapex.timeseries import TimeSeriesDataFrame

        tsdf = TimeSeriesDataFrame.from_data_frame(
            clean_df.drop(columns=["fill_flag"], errors="ignore"),
            id_column=cfg.id_col,
            timestamp_column=cfg.time_col,
        )
    except Exception:
        tsdf = None

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        clean_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    fill_info = {}
    if "fill_flag" in clean_df.columns:
        fill_info = clean_df["fill_flag"].value_counts().to_dict()

    log.log_output("clean_df", {
        "backend": backend,
        "rows": len(clean_df),
        "fill_flag_counts": fill_info,
        "output_path": output_path,
    }, phase=PHASE)

    return clean_df, tsdf
