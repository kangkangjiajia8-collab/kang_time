"""历史 episode 库 D 与 Retrieve(T_m, D) 检索。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from prior_enhance.config import DataConfig
from prior_enhance.context.event_novelty import (
    build_calendar_event_context,
    event_to_text,
    events_to_text,
)
from prior_enhance.context.scenario import scenario_label
from prior_enhance.context.text_encoder import ChineseTextEncoder, get_text_encoder
from prior_enhance.schemas import EventContext, EventItem


@dataclass
class Episode:
    """历史 episode e_i：序列片段 + 场景 + 事件语义。"""

    episode_id: str
    scenario: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    decision_time: pd.Timestamp
    event_context: EventContext
    event_summary: str = ""
    stats_mean: float = 0.0
    stats_std: float = 0.0
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


def _is_valid_y(val: float, cfg: DataConfig) -> bool:
    if pd.isna(val) or np.isinf(val):
        return False
    if val < 0:
        return False
    return True


def _event_context_from_news(
    news_df: pd.DataFrame,
    decision_time: pd.Timestamp,
    *,
    lookback_days: int = 21,
    max_items: int = 5,
) -> Optional[EventContext]:
    """从新闻表构造 episode 事件（决策前 lookback 天内最多 max_items 条）。"""
    if news_df is None or news_df.empty:
        return None
    start = decision_time - pd.Timedelta(days=lookback_days)
    mask = (news_df["news_time"] > start) & (news_df["news_time"] <= decision_time)
    subset = news_df.loc[mask].sort_values("news_time", ascending=False).head(max_items)
    if subset.empty:
        return None
    events = []
    for i, row in subset.iterrows():
        title = str(row.get("title", "")).strip()
        summary = str(row.get("summary", "")).strip()[:300]
        content = f"{title} | {summary}" if summary else title
        events.append(
            EventItem(
                event_id=f"news_{i}",
                event_time=row["news_time"].to_pydatetime(),
                source="fnspid",
                event_type="news",
                title=title[:200],
                content=content,
            )
        )
    return EventContext(events=events)


def build_episode_store_from_history(
    history_df: pd.DataFrame,
    cfg: Optional[DataConfig] = None,
    *,
    block_size: int = 168,
    stride: Optional[int] = None,
    encoder: Optional[ChineseTextEncoder] = None,
    news_df: Optional[pd.DataFrame] = None,
    news_lookback_days: int = 21,
) -> List[Episode]:
    """
    从 H_m 滑动窗口构建 episode 库。

    若提供 news_df，则每个 episode 优先使用同期 FNSPID 新闻；否则日历合成事件。
    """
    cfg = cfg or DataConfig()
    df = history_df.sort_values(cfg.time_col).reset_index(drop=True)
    ts = pd.to_datetime(df[cfg.time_col])
    y = df[cfg.target_col].astype(float)
    stride = stride or max(block_size // 2, 1)
    encoder = encoder or get_text_encoder()

    episodes: List[Episode] = []
    n = len(df)
    idx = block_size
    ep_idx = 0

    while idx < n:
        start = max(0, idx - block_size)
        block = df.iloc[start:idx]
        block_ts = ts.iloc[start:idx]
        block_y = y.iloc[start:idx]
        valid_mask = block_y.apply(lambda v: _is_valid_y(v, cfg))
        if valid_mask.sum() < block_size * 0.8:
            idx += stride
            continue

        decision_time = block_ts.iloc[-1]
        scenario = scenario_label(decision_time)
        event_ctx = _event_context_from_news(
            news_df, decision_time, lookback_days=news_lookback_days,
        )
        if event_ctx is None:
            event_ctx = build_calendar_event_context(
                decision_time.to_pydatetime(),
                horizon=min(block_size, 168),
                scenario=scenario,
            )
        summary = events_to_text(event_ctx.events)

        ep = Episode(
            episode_id=f"ep_{ep_idx:05d}",
            scenario=scenario,
            start_time=block_ts.iloc[0],
            end_time=decision_time,
            decision_time=decision_time,
            event_context=event_ctx,
            event_summary=summary,
            stats_mean=float(block_y.mean()),
            stats_std=float(block_y.std(ddof=1) if len(block_y) > 1 else 0.0),
        )
        episodes.append(ep)
        ep_idx += 1
        idx += stride

    if episodes:
        texts = [ep.event_summary for ep in episodes]
        vecs = encoder.encode(texts)
        for ep, vec in zip(episodes, vecs):
            ep.embedding = vec

    return episodes


@dataclass
class RetrievedEpisode:
    """带检索分数的 episode。"""

    episode: Episode
    retrieval_score: float = 0.0


def retrieve_episodes(
    decision_time: datetime,
    scenario: str,
    event_context: EventContext,
    store: Sequence[Episode],
    *,
    top_k: int = 5,
    encoder: Optional[ChineseTextEncoder] = None,
) -> List[RetrievedEpisode]:
    """
    Retrieve(T_m, D)：优先同场景 episode，再按事件语义相似度排序。
    """
    if not store:
        return []

    encoder = encoder or get_text_encoder()
    query_text = events_to_text(event_context.events)
    query_vec = encoder.encode(query_text) if query_text.strip() else None

    same_scenario = [ep for ep in store if ep.scenario == scenario]
    pool = same_scenario if same_scenario else list(store)

    scored: List[RetrievedEpisode] = []
    for ep in pool:
        if query_vec is not None and ep.embedding is not None:
            sim = float(np.dot(query_vec, ep.embedding))
        elif query_text.strip() and ep.event_summary:
            sim = encoder.max_similarity(query_text, [ep.event_summary])
        else:
            sim = 0.0
        dt_penalty = abs((pd.Timestamp(decision_time) - ep.decision_time).days) / 365.0
        score = sim - 0.05 * dt_penalty
        scored.append(RetrievedEpisode(episode=ep, retrieval_score=score))

    scored.sort(key=lambda x: x.retrieval_score, reverse=True)
    return scored[:top_k]


def episodes_for_novelty_pool(
    store: Sequence[Episode],
    scenario: str,
    decision_time: datetime,
    *,
    min_gap_days: int = 7,
) -> List[Episode]:
    """
    构造 E_m^hist：同类场景历史事件，排除与当前决策时刻过近的 episode。

    避免滑动窗口相邻块产生几乎相同的日历文本，导致 s_m^max 虚高。
    """
    dt = pd.Timestamp(decision_time)
    pool: List[Episode] = []
    for ep in store:
        if ep.scenario != scenario:
            continue
        gap_days = abs((dt - ep.decision_time).days)
        if gap_days < min_gap_days:
            continue
        pool.append(ep)
    return pool if pool else [ep for ep in store if ep.scenario == scenario]


def historical_event_texts_for_novelty(
    episodes: Sequence[Episode],
    *,
    same_scenario_only: bool = True,
    scenario: Optional[str] = None,
) -> List[str]:
    """提取 E_m^hist：用于事件新颖性计算的历史事件文本集合。"""
    pool = episodes
    if same_scenario_only and scenario:
        filtered = [ep for ep in episodes if ep.scenario == scenario]
        if filtered:
            pool = filtered
    return [ep.event_summary for ep in pool if ep.event_summary]


def parse_event_context(
    event_text: Optional[str] = None,
    events: Optional[List[EventItem]] = None,
    decision_time=None,
    horizon: int = 72,
    scenario: str = "weekday",
) -> EventContext:
    """从显式事件或日历场景构造 E_m。"""
    if events:
        return EventContext(events=events)
    if event_text and event_text.strip():
        return EventContext(
            events=[
                EventItem(
                    event_id="evt_user",
                    event_time=decision_time,
                    source="user",
                    event_type="text",
                    title="用户事件",
                    content=event_text.strip(),
                )
            ]
        )
    return build_calendar_event_context(decision_time, horizon, scenario)
