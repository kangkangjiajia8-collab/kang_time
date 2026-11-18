"""窗口切分：holdout 单窗口（默认）或 rolling 多窗口。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from prior_enhance.config import DataConfig, PriorEnhanceConfig, WindowConfig
from prior_enhance.context.state import compute_context_state, compute_series_stats
from prior_enhance.context.text_encoder import get_text_encoder_from_config
from prior_enhance.data.episode_store import (
    build_episode_store_from_history,
    episodes_for_novelty_pool,
    parse_event_context,
    retrieve_episodes,
)
from prior_enhance.schemas import EventContext, ForecastTask, RetrievedEpisodeSummary
from prior_enhance.telemetry.run_logger import log

PHASE = "WINDOW"

HOLDOUT_TAG = "holdout"


@dataclass
class WindowSlice:
    """单个预测窗口的数据切片。"""

    index: int
    decision_time: pd.Timestamp
    history_df: pd.DataFrame
    oos_df: pd.DataFrame


def _extract_series(clean_df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    df = clean_df.sort_values([cfg.id_col, cfg.time_col]).reset_index(drop=True)
    item_ids = df[cfg.id_col].unique()
    if len(item_ids) == 1:
        return df[df[cfg.id_col] == item_ids[0]].reset_index(drop=True)
    return df.reset_index(drop=True)


def build_holdout_window(
    clean_df: pd.DataFrame,
    horizon: int,
    min_history: int = 48,
    cfg: Optional[DataConfig] = None,
) -> WindowSlice:
    """
    单窗口 holdout：末尾 H 点作测试 O_m，前面全部为 H_m。

    例：3600 行、H=72 → H_m=3528 行，O_m=72 行（未来 3 天）。
    """
    cfg = cfg or DataConfig()
    series = _extract_series(clean_df, cfg)
    n = len(series)

    if n < min_history + horizon:
        raise ValueError(
            f"数据不足：共 {n} 行，需要至少 min_history({min_history}) + horizon({horizon}) = {min_history + horizon}"
        )

    hist = series.iloc[: n - horizon].copy()
    oos = series.iloc[n - horizon :].copy()

    window = WindowSlice(
        index=0,
        decision_time=pd.Timestamp(hist[cfg.time_col].iloc[-1]),
        history_df=hist,
        oos_df=oos,
    )
    log.log_output(
        "holdout_window",
        {
            "total_rows": n,
            "H_m_rows": len(hist),
            "O_m_rows": len(oos),
            "horizon": horizon,
            "decision_time": str(window.decision_time),
            "history_start": str(hist[cfg.time_col].iloc[0]),
            "history_end": str(hist[cfg.time_col].iloc[-1]),
            "oos_start": str(oos[cfg.time_col].iloc[0]),
            "oos_end": str(oos[cfg.time_col].iloc[-1]),
        },
        phase=PHASE,
    )
    return window


def build_rolling_windows(
    clean_df: pd.DataFrame,
    horizon: int,
    stride: Optional[int] = None,
    min_history: int = 48,
    cfg: Optional[DataConfig] = None,
) -> List[WindowSlice]:
    """滚动多窗口（实验用）；默认流程请用 build_holdout_window。"""
    cfg = cfg or DataConfig()
    series = _extract_series(clean_df, cfg)
    stride = stride or horizon

    n = len(series)
    windows: List[WindowSlice] = []
    w_idx = 0
    end = min_history
    while end + horizon <= n:
        hist = series.iloc[: end + 1].copy()
        oos = series.iloc[end + 1 : end + 1 + horizon].copy()
        if len(oos) < horizon:
            break
        windows.append(
            WindowSlice(
                index=w_idx,
                decision_time=pd.Timestamp(hist[cfg.time_col].iloc[-1]),
                history_df=hist,
                oos_df=oos,
            )
        )
        w_idx += 1
        end += stride

    return windows


def build_windows(
    clean_df: pd.DataFrame,
    config: Optional[PriorEnhanceConfig] = None,
) -> List[WindowSlice]:
    """按配置构建窗口列表（holdout 仅 1 个，rolling 多个）。"""
    config = config or PriorEnhanceConfig()
    wcfg = config.window
    horizon = wcfg.effective_horizon()

    if wcfg.mode == "rolling":
        return build_rolling_windows(
            clean_df,
            horizon=horizon,
            min_history=wcfg.min_history_points,
            cfg=config.data,
        )
    return [build_holdout_window(
        clean_df,
        horizon=horizon,
        min_history=wcfg.min_history_points,
        cfg=config.data,
    )]


def window_artifact_tag(window: WindowSlice, mode: str = "holdout") -> str:
    if mode == "holdout":
        return HOLDOUT_TAG
    return f"win_{window.index:04d}"


def save_window_artifacts(
    window: WindowSlice,
    output_dir: str,
    cfg: Optional[DataConfig] = None,
    tag: Optional[str] = None,
) -> Tuple[str, str]:
    """保存 H_m 与 O_m 到 CSV，返回 (history_path, oos_path)。"""
    cfg = cfg or DataConfig()
    os.makedirs(output_dir, exist_ok=True)
    tag = tag or f"win_{window.index:04d}"
    hist_path = os.path.join(output_dir, f"{tag}_history.csv")
    oos_path = os.path.join(output_dir, f"{tag}_oos.csv")
    window.history_df.to_csv(hist_path, index=False, encoding="utf-8-sig")
    window.oos_df.to_csv(oos_path, index=False, encoding="utf-8-sig")
    return hist_path, oos_path


def build_forecast_task(
    window: WindowSlice,  # 输入：单个 holdout/rolling 窗口（含 H_m、O_m、decision_time）
    config: Optional[PriorEnhanceConfig] = None,  # 全局配置；None 则用默认 PriorEnhanceConfig
    known_covariates: Optional[List[str]] = None,  # 已知外生协变量列名（如天气）；FNSPID demo 通常为空
    event_text: Optional[str] = None,  # 用户事件文本（如 FNSPID 新闻摘要）；可替代/补充 event_context
    event_context: Optional[EventContext] = None,  # 结构化事件 E_m；若已构造则跳过 parse_event_context
    episode_count: int = 0,  # 覆盖检索到的 episode 数；0 表示用 retrieve 实际返回条数
    artifact_dir: Optional[str] = None,  # H_m/O_m CSV 落盘目录；None 则用 outputs/windows/{tag}/
    news_df: Optional[pd.DataFrame] = None,  # FNSPID 历史新闻（注入 episode 库）
) -> ForecastTask:  # 输出：论文中的预测任务 T_m（供先验生成与 search 使用）
    """将 WindowSlice 映射为 T_m；prior 生成仅使用 H_m。"""
    config = config or PriorEnhanceConfig()  # 兜底默认配置（数据列名、窗口、LLM、先验等）
    dcfg = config.data  # 数据相关：id_col/time_col/target_col/freq 等
    wcfg = config.window  # 窗口相关：forecast_days、horizon、search_budget、loss_name
    horizon = wcfg.effective_horizon()  # 预测步数 H = forecast_days × points_per_day（AAL 日频为 5）
    tag = window_artifact_tag(window, wcfg.mode)  # 产物标签：holdout 模式固定为 "holdout"

    if artifact_dir is None:  # 未指定落盘路径时
        artifact_dir = os.path.join(config.output_dir, "windows", tag)  # 默认 outputs/windows/holdout/
    hist_path, oos_path = save_window_artifacts(window, artifact_dir, dcfg, tag=tag)  # 把 H_m、O_m 写成 CSV 并返回路径

    # 先预热文本编码器，避免在 AutoApex/R 加载后再初始化 torch 导致长时间阻塞
    encoder = get_text_encoder_from_config(config)  # 按配置加载 BGE 等中文句向量模型（默认 CPU）
    _ = encoder.encode("编码器预热")  # 跑一次空编码，触发模型权重加载，避免后续首次调用卡顿
    log.log_output("text_encoder_ready", {"model": encoder.model_name}, phase=PHASE)  # 记录编码器就绪日志

    stats = compute_series_stats(window.history_df, dcfg)  # 仅用 H_m 算序列统计：均值/方差/周期性/趋势等

    decision_time = window.decision_time.to_pydatetime()  # 决策时刻 t_m（H_m 最后一个时间点），转 Python datetime
    from prior_enhance.context.scenario import scenario_label  # 延迟导入：按日历打场景标签

    c_m = scenario_label(decision_time)  # 场景 c_m：weekday/weekend/春节/国庆等（影响 episode 检索与 n_m）
    log.log_output("build_Tm_scenario", {"c_m": c_m}, phase=PHASE)  # 记录当前场景标签
    episode_store = build_episode_store_from_history(
        window.history_df, dcfg, encoder=encoder, news_df=news_df,
    )
    log.log_output("episode_store", {"count": len(episode_store)}, phase=PHASE)  # 记录 episode 总数
    if event_context is None:  # 调用方未传入结构化 E_m 时
        event_context = parse_event_context(
            event_text=event_text,  # 若有 FNSPID 新闻文本，封装为 EventContext（source=user）
            decision_time=decision_time,  # 事件锚定在决策时刻
            horizon=horizon,  # 未来预测步数，写入事件描述
            scenario=c_m,  # 无用户文本时退化为日历合成事件（weekday 等）
        )
    retrieved = retrieve_episodes(
        decision_time,  # 当前决策时刻，用于时间距离加权
        c_m,  # 当前场景，同场景 episode 优先
        event_context,  # 当前 E_m，用于语义相似度检索
        episode_store,  # 历史 episode 库 D
        top_k=config.prior.episode_ref_k,  # 检索 top-k（默认 10）→ 论文 |D_m|
        encoder=encoder,  # 用同一编码器算 query 与 episode 的余弦相似度
    )
    episode_count = len(retrieved) if episode_count == 0 else episode_count  # 有效检索条数，可被参数覆盖
    retrieved_summaries = [  # 将检索结果转为可序列化摘要，写入 task_Tm.json
        RetrievedEpisodeSummary(
            episode_id=r.episode.episode_id,  # episode 唯一 ID
            scenario=r.episode.scenario,  # 该 episode 所属场景
            decision_time=r.episode.decision_time.to_pydatetime(),  # episode 决策时刻
            event_summary=r.episode.event_summary,  # episode 事件摘要（目前多为日历文本）
            retrieval_score=round(r.retrieval_score, 4),  # 检索得分（场景+语义+时间综合）
        )
        for r in retrieved  # 遍历 top-k 检索命中
    ]

    novelty_pool = episodes_for_novelty_pool(
        episode_store, c_m, decision_time  # 为 ν_m 筛选可比历史事件池（同场景、决策时刻之前）
    )
    context = compute_context_state(
        window.history_df,  # H_m 价格序列，用于漂移 d_m 等
        stats,  # 上面算好的序列统计
        event_context=event_context,  # 当前 E_m，用于算 ν_m = 1 - s_max
        episodes=novelty_pool,  # 历史事件文本池，与当前 E_m 比语义相似度
        cfg=dcfg,  # 数据配置（列名、频率）
        horizon=horizon,  # 预测步数，影响近期窗口长度等
        encoder=encoder,  # 文本编码器
    )  # 得到 S_m = (n_m, ν_m, d_m) 及 cond_labels、analysis 等

    cov_cols = known_covariates or []  # 外生协变量列名列表，默认空
    cov_cols = [
        c for c in cov_cols if c in window.history_df.columns and c != dcfg.target_col  # 只保留 H_m 中实际存在且非目标列
    ]

    prompt_event_text = event_context.to_prompt_text() or event_text  # LLM prompt 用的事件文本；优先结构化 E_m 的渲染结果

    return ForecastTask(  # 组装并返回完整 T_m 对象（Pydantic 模型）
        window_index=window.index,  # 窗口序号（holdout 恒为 0）
        decision_time=decision_time,  # t_m
        horizon=horizon,  # H：未来预测步数
        search_budget=wcfg.search_budget,  # recipe search 预算 B（默认 50）
        loss_name=wcfg.loss_name,  # 评估损失名（默认 mape）
        history_path=hist_path,  # H_m CSV 路径（search/评估时读取）
        history_rows=len(window.history_df),  # H_m 行数
        oos_path=oos_path,  # O_m CSV 路径（先验阶段 LLM 不可见；search 后用于 J_obs）
        oos_rows=len(window.oos_df),  # O_m 行数
        known_covariates=cov_cols,  # X_m+：已知未来协变量列（FNSPID demo 为空）
        event_text=prompt_event_text,  # E_m 文本形式，直接进入 LLM prompt
        event_context=event_context,  # E_m 结构化形式，用于 ν_m 与检索
        domain_knowledge=config.domain_knowledge,  # K_m：领域先验知识（FNSPID 为美股日频描述）
        target_col=dcfg.target_col,  # 目标列名（默认 value）
        item_id=str(window.history_df[dcfg.id_col].iloc[0]),  # 序列 ID（AAL）
        episode_count=episode_count,  # |D_m| 检索 episode 数
        retrieved_episodes=retrieved_summaries,  # 检索到的 episode 摘要列表
        stats=stats,  # H_m 序列统计量
        context_state=context,  # S_m：n_m、ν_m、d_m、cond_labels、analysis
    )
