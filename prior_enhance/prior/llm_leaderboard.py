"""LLM 公开智能指数排行榜（λ_m^{init} 查表）。"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

# 表内极值：排名 1 / 205
SCORE_MAX = 61.0
SCORE_MIN = 5.0

_CSV_PATH = Path(__file__).resolve().parent / "data" / "llm_leaderboard.csv"

# 可选：配置名 → 排行榜条目（须人工确认）
_MODEL_ALIASES: Dict[str, str] = {}


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@lru_cache(maxsize=1)
def _load_score_index() -> Dict[str, float]:
    """归一化模型名 → 智能指数（同名取最高分）。"""
    index: Dict[str, float] = {}
    if not _CSV_PATH.is_file():
        return index
    with _CSV_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("model_name") or "").strip()
            if not name:
                continue
            try:
                score = float(row.get("score") or 0)
            except ValueError:
                continue
            key = _normalize_key(name)
            index[key] = max(index.get(key, 0.0), score)
    return index


def lookup_leaderboard_score(model_name: Optional[str]) -> Tuple[float, bool]:
    """
    查表得 score_m。

    返回 (score, matched)。未命中时 score=SCORE_MIN，matched=False（λ 归一化为 0）。
    """
    if not model_name or not str(model_name).strip():
        return SCORE_MIN, False

    index = _load_score_index()
    if not index:
        return SCORE_MIN, False

    raw = str(model_name).strip()
    keys_to_try = [_normalize_key(raw)]
    if keys_to_try[0] in _MODEL_ALIASES:
        keys_to_try.append(_MODEL_ALIASES[keys_to_try[0]])

    for key in keys_to_try:
        if key in index:
            return index[key], True

    return SCORE_MIN, False


def leaderboard_bounds() -> Tuple[float, float]:
    return SCORE_MAX, SCORE_MIN
