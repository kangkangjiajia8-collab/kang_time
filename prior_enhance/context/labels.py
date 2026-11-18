"""S_m 连续值 → 离散语义标签（论文 cond 生成）。"""

from __future__ import annotations

from typing import List, Tuple


def label_n_m(n_m: float) -> str:
    if n_m < 0.33:
        return "low_sample"
    if n_m < 0.66:
        return "medium_sample"
    return "high_sample"


def label_nu_m(nu_m: float) -> str:
    return "novel_event" if nu_m >= 0.5 else "familiar_event"


def label_d_m(d_m: float) -> str:
    if d_m < 0.33:
        return "stable"
    if d_m < 0.66:
        return "moderate_drift"
    return "severe_drift"


def build_cond_labels(n_m: float, nu_m: float, d_m: float) -> Tuple[List[str], str, str, str]:
    """返回 (cond 标签列表, n_label, nu_label, d_label)。"""
    n_l = label_n_m(n_m)
    nu_l = label_nu_m(nu_m)
    d_l = label_d_m(d_m)
    return [n_l, nu_l, d_l], n_l, nu_l, d_l
