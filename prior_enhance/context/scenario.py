"""预测窗口场景标签 c_m（节假日 / 工作日等）。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Union

import pandas as pd

try:
    import chinese_calendar as cc
except ImportError:  # pragma: no cover
    cc = None  # type: ignore


TimeLike = Union[date, datetime, pd.Timestamp, str]


def _to_date(ts: TimeLike) -> date:
    if isinstance(ts, date) and not isinstance(ts, datetime):
        return ts
    return pd.Timestamp(ts).date()


def scenario_label(ts: TimeLike) -> str:
    """
    为时刻 τ 分配场景标签 c_τ。

    优先识别中国法定节假日；否则区分周末 / 工作日。
    """
    d = _to_date(ts)
    if cc is not None:
        if cc.is_holiday(d):
            detail = cc.get_holiday_detail(d)
            if detail[1]:
                name = detail[1]
                if "春节" in name:
                    return "spring_festival"
                if "国庆" in name:
                    return "national_day"
                if "劳动" in name:
                    return "labor_day"
                if "清明" in name:
                    return "qingming"
                if "端午" in name:
                    return "dragon_boat"
                if "中秋" in name:
                    return "mid_autumn"
                if "元旦" in name:
                    return "new_year"
                return "holiday"
        if d.weekday() >= 5:
            return "weekend"
        return "weekday"

    # 无 chinese_calendar 时的降级规则
    if d.weekday() >= 5:
        return "weekend"
    return "weekday"


def scenario_labels_for_series(ts: pd.Series) -> pd.Series:
    """
    为时间序列批量打场景标签（按日期去重，避免逐点重复调用日历 API）。
    """
    ts = pd.to_datetime(ts)
    dates = ts.dt.date
    unique_dates = pd.unique(dates)
    mapping = {d: scenario_label(d) for d in unique_dates}
    return dates.map(mapping)


def scenario_ref_count(
    history_ts: pd.Series,
    c_m: str,
    horizon: int = 1,
) -> tuple[int, int]:
    """
    计算 N_m^eff 与 N_m^ref。

    - N_m^eff: 历史中同类场景且 y 有效的样本数
    - N_m^ref: 该场景理论最大可获得样本数（历史覆盖年数 × 每周期点数）
    """
    ts = pd.to_datetime(history_ts)
    labels = scenario_labels_for_series(ts)
    n_eff = int((labels == c_m).sum())

    years = ts.dt.year.nunique()
    if years <= 0:
        years = 1

    if c_m in ("spring_festival", "national_day", "labor_day", "qingming", "dragon_boat", "mid_autumn", "new_year", "holiday"):
        n_ref = max(years * horizon, 1)
    elif c_m == "weekend":
        n_ref = max(years * 104 * horizon, 1)  # 约 52 周 × 2 天
    else:
        n_ref = max(years * 261 * horizon, 1)  # 约 261 工作日

    return n_eff, n_ref
