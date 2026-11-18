"""事件新颖性 ν_m = 1 - s_m^max。"""

from __future__ import annotations

from typing import Optional, Sequence

from prior_enhance.context.text_encoder import ChineseTextEncoder, get_text_encoder
from prior_enhance.schemas import EventContext, EventItem


def event_to_text(event: EventItem) -> str:
    """将单条事件转为编码用文本。"""
    parts = [event.title.strip()]
    if event.content and event.content.strip():
        parts.append(event.content.strip())
    if event.event_type:
        parts.append(f"类型:{event.event_type}")
    return "。".join(p for p in parts if p)


def events_to_text(events: Sequence[EventItem]) -> str:
    """将事件集合 E_m 合并为单段文本供 g(E_m) 编码。"""
    if not events:
        return ""
    return "\n".join(event_to_text(e) for e in events if event_to_text(e))


def build_calendar_event_context(
    decision_time,
    horizon: int,
    scenario: str,
) -> EventContext:
    """无显式事件时，由日历场景生成合成事件上下文（供编码与 prompt 使用）。"""
    scenario_desc = {
        "spring_festival": "春节期间，居民用电模式与平日差异显著",
        "national_day": "国庆期间，工业负荷偏低、居民负荷可能升高",
        "labor_day": "劳动节假期，负荷结构发生变化",
        "weekend": "周末模式，工业负荷通常低于工作日",
        "weekday": "正常工作日模式，未出现长假",
        "holiday": "法定节假日，负荷模式与平日不同",
    }.get(scenario, f"当前场景标签为 {scenario}")

    return EventContext(
        events=[
            EventItem(
                event_id=f"cal_{scenario}",
                event_time=decision_time,
                source="calendar",
                event_type="calendar",
                title=f"预测期场景: {scenario}",
                content=(
                    f"决策时刻 {decision_time}，未来 {horizon} 步预测。"
                    f"{scenario_desc}。"
                ),
            )
        ]
    )


def compute_event_novelty(
    current_events: EventContext,
    historical_event_texts: Sequence[str],
    encoder: Optional[ChineseTextEncoder] = None,
) -> tuple[float, float]:
    """
    计算 ν_m 与 s_m^max。

    ν_m = 1 - max_{E_i ∈ E_m^hist} cos(g(E_m), g(E_i))
    """
    encoder = encoder or get_text_encoder()
    current_text = events_to_text(current_events.events)
    if not current_text.strip():
        return 0.5, 0.0

    hist = [t for t in historical_event_texts if t and t.strip()]
    if not hist:
        return 0.7, 0.0

    s_max = encoder.max_similarity(current_text, hist)
    nu_m = float(max(0.0, min(1.0, 1.0 - s_max)))
    return nu_m, s_max
