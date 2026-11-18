"""H_m 内部时间有序 train-val 切分。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from prior_enhance.config import DataConfig


@dataclass
class ValSplit:
    """单个 (D_train, D_val) 对。"""

    split_id: int
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    train_end: pd.Timestamp
    val_start: pd.Timestamp


def make_holdout_val_split(
    history_df: pd.DataFrame,
    val_horizon: int,
    cfg: DataConfig | None = None,
) -> ValSplit:
    """
    在 H_m 末尾留 val_horizon 作内部验证（相对 train 为「未来」）。

    max(train) < min(val)，且 train/val ⊂ H_m。
    """
    cfg = cfg or DataConfig()
    df = history_df.sort_values(cfg.time_col).reset_index(drop=True)
    if len(df) <= val_horizon + 48:
        raise ValueError(f"历史过短，无法切分 val_horizon={val_horizon}")

    min_train = 48
    if len(df) < val_horizon + min_train:
        raise ValueError(f"需要至少 {val_horizon + min_train} 点历史")

    train_df = df.iloc[: len(df) - val_horizon].copy()
    val_df = df.iloc[len(df) - val_horizon :].copy()
    ts = pd.to_datetime(df[cfg.time_col])

    return ValSplit(
        split_id=0,
        train_df=train_df,
        val_df=val_df,
        train_end=pd.Timestamp(ts.iloc[len(train_df) - 1]),
        val_start=pd.Timestamp(ts.iloc[len(train_df)]),
    )


def make_rolling_val_splits(
    history_df: pd.DataFrame,
    val_horizon: int,
    n_splits: int = 1,
    cfg: DataConfig | None = None,
) -> List[ValSplit]:
    """滚动内部验证切分（实验用）；默认单切分与 holdout 一致。"""
    if n_splits <= 1:
        return [make_holdout_val_split(history_df, val_horizon, cfg)]

    cfg = cfg or DataConfig()
    df = history_df.sort_values(cfg.time_col).reset_index(drop=True)
    n = len(df)
    splits: List[ValSplit] = []
    stride = max(val_horizon, (n - val_horizon - 48) // max(n_splits, 1))

    for i in range(n_splits):
        end_idx = n - val_horizon - i * stride
        if end_idx < 48 + val_horizon:
            break
        train_df = df.iloc[: end_idx - val_horizon].copy()
        val_df = df.iloc[end_idx - val_horizon : end_idx].copy()
        if len(train_df) < 48 or len(val_df) < val_horizon:
            continue
        ts = pd.to_datetime(df[cfg.time_col])
        splits.append(
            ValSplit(
                split_id=i,
                train_df=train_df,
                val_df=val_df,
                train_end=pd.Timestamp(ts.iloc[end_idx - val_horizon - 1]),
                val_start=pd.Timestamp(ts.iloc[end_idx - val_horizon]),
            )
        )
    return splits or [make_holdout_val_split(history_df, val_horizon, cfg)]
