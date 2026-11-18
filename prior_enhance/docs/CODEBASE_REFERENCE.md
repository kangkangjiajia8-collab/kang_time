# prior_enhance 代码库全文件解析

> **项目全称**：Evidence-Calibrated LLM Priors for Nonstationary Forecasting Recipe Search  
> **工程目录**：`prior_enhance/`  
> **文档版本**：2026-06-12  
> **适用读者**：接手开发、写论文复现、做 code review 的工程师

---

## 目录

1. [项目定位与数据流](#1-项目定位与数据流)
2. [目录树总览](#2-目录树总览)
3. [根目录核心文件](#3-根目录核心文件)
4. [data/ — 数据预处理与任务构建](#4-data--数据预处理与任务构建)
5. [context/ — 上下文状态 S_m](#5-context--上下文状态-s_m)
6. [prior/ — LLM 先验 Z_m 与编译器](#6-prior--llm-先验-z_m-与编译器)
7. [recipe/ — 预测方案 r 的定义与构建](#7-recipe--预测方案-r-的定义与构建)
8. [validation/ — 验证切分、目标函数与评估](#8-validation--验证切分目标函数与评估)
9. [search/ — 窗口内 Prior-Guided Search](#9-search--窗口内-prior-guided-search)
10. [observation/ — O_m 反馈与持久记忆](#10-observation--o_m-反馈与持久记忆)
11. [pipeline.py — 端到端编排](#11-pipelinepy--端到端编排)
12. [telemetry/ — 透明日志](#12-telemetry--透明日志)
13. [examples/ — 可执行入口脚本](#13-examples--可执行入口脚本)
14. [辅助与引导文件](#14-辅助与引导文件)
15. [outputs/ — 运行产物约定](#15-outputs--运行产物约定)
16. [模块依赖与调用链](#16-模块依赖与调用链)
17. [环境变量速查](#17-环境变量速查)

---

## 1. 项目定位与数据流

### 1.1 这个项目做什么

`prior_enhance` 是在 **AutoApex 预测平台**之上的一层「先验增强搜索框架」：

- **输入**：一条（或多条）时间序列 CSV + 可选事件文本（新闻/公告）+ 领域知识描述
- **输出**：在当前决策时刻 t_m 下，经 LLM 先验引导搜索得到的最优预测方案 r_m^，以及对盲盒未来区间 \mathcal{O}_m 的外推预测
- **核心思想**：LLM 只生成**可检验的结构化先验**（推荐/规避哪些 recipe 组件），不直接输出预测值；最终方案由验证损失、成本、风险约束客观决定

### 1.2 端到端数据流（简化）

```
原始 CSV
  └─ data/preprocessor.py          → clean.csv（D_full）
       └─ data/window.py           → H_m.csv + O_m.csv + ForecastTask T_m
            ├─ context/state.py    → S_m = (n_m, ν_m, d_m, ξ_m)
            ├─ data/episode_store.py → D_m 检索
            └─ prior/generator.py  → Z_m 结构化先验
                 └─ prior/compiler.py → φ_{j,m}(r)
                      └─ search/loop.py → 采样 / 评估 / τ^loc 更新 → r*
                           └─ validation/evaluator.py → Train(H_m) + 预测 O_m
                                └─ observation/feedback.py → G^obs → 持久 τ, λ
```

### 1.3 与 AutoApex 的边界


| 职责                       | prior_enhance        | AutoApex serve |
| ------------------------ | -------------------- | -------------- |
| 构造 T_m、S_m、Z_m           | ✅                    | —              |
| Prior-guided 采样与 Gain 回归 | ✅                    | —              |
| 真实模型训练 `/api/train`      | 发起请求                 | 执行             |
| tsfresh / 模型库 / 集成       | 通过 recipe payload 配置 | 执行             |
| 代理快速评估（proxy）            | ✅ `evaluator.py`     | —              |


---

## 2. 目录树总览

```
prior_enhance/
├── __init__.py                 # 包入口，懒加载 Pipeline
├── _bootstrap.py               # 运行时 bootstrap（random patch 等）
├── _path.py                    # AutoApex 路径注入
├── _stdlib_patch/random.py     # Python 3.12+ randbits 兼容补丁
├── config.py                   # 全局配置 dataclass
├── schemas.py                  # Pydantic 核心数据结构（T_m, S_m, Z_m…）
├── pipeline.py                 # 主编排类 PriorEnhancePipeline
├── secrets.py / secrets.example.py  # API Key（不入库）
├── requirements.txt            # Python 依赖
│
├── data/                       # 数据层
│   ├── preprocessor.py
│   ├── window.py
│   ├── episode_store.py
│   ├── 代理购电_小时.csv        # 默认 demo 数据
│   └── fnspid/                 # FNSPID 金融数据工具链
│
├── context/                    # S_m 与文本语义
├── prior/                      # Z_m 生成、φ 编译、τ 初始化
├── recipe/                     # ForecastingRecipe r=(W,F,M,θ,V,Q)
├── validation/                 # 切分、J^val/J^obs、Gain、AutoApex 客户端
├── search/                     # Prior-guided search 主循环
├── observation/                # Observation feedback、持久 memory
├── telemetry/                  # RunLogger 透明日志
│
├── examples/                   # 可执行 demo / 验证脚本
├── docs/                       # 设计文档（含本文）
└── outputs/                    # 运行产物（gitignore 建议）
    ├── preprocessed/
    ├── windows/
    ├── results/
    ├── logs/
    └── prior_store.jsonl
```

---

## 3. 根目录核心文件

### `config.py`

**作用**：集中管理所有可配置参数，支持环境变量覆盖。


| 类                    | 论文符号 / 含义     | 关键字段                                                      |
| -------------------- | ------------- | --------------------------------------------------------- |
| `DataConfig`         | 数据列名与频率       | `time_col`, `target_col`, `id_col`, `freq`                |
| `WindowConfig`       | H, B, l, 窗口模式 | `forecast_days`, `points_per_day`, `mode=holdout|rolling` |
| `LLMConfig`          | 大模型调用         | `model`, `api_key`, `min_priors`, `max_priors`            |
| `PriorConfig`        | 置信度校准         | `confidence_lambda`（b 融合 λ）, `episode_ref_k`（K）           |
| `TextEncoderConfig`  | BGE 编码器       | `model_name`, `device`                                    |
| `SearchConfig`       | 搜索与 Gain      | `max_trials`, `explore_ratio`（ρ）, `beta_k`, `chi_min`     |
| `AutoApexConfig`     | serve 连接      | `base_url`, `eval_backend`, `train_timeout`               |
| `LoggingConfig`      | 日志开关          | `enabled`, `log_dir`, `level`                             |
| `PriorEnhanceConfig` | 顶层聚合          | `from_env()` 读取环境变量                                       |


**谁调用它**：几乎所有模块；`pipeline.py` 在初始化时构造 `PriorEnhanceConfig.from_env()`。

---

### `schemas.py`

**作用**：与论文符号一一对应的 **Pydantic 数据模型**，所有 JSON 落盘（`task_Tm.json` 等）均由此序列化。


| 类                            | 论文对象                | 说明                                                   |
| ---------------------------- | ------------------- | ---------------------------------------------------- |
| `EventItem` / `EventContext` | E_m                 | 单条/集合事件文本，供 LLM prompt 与 ν_m 计算                      |
| `PriorTargetClusters`        | \text{target}_{j,m} | 五簇推荐：window / feature / models / hpo / ensemble      |
| `StructuredPrior`            | z_{j,m}             | cond + target + action + b_llm/b_evidence/confidence |
| `ContextState`               | S_m                 | n_m, nu_m, d_m, xi_m + 离散标签 + analysis               |
| `AnalysisDimensions`         | 辅助诊断                | 周期性、趋势、波动聚集等（供 prompt）                               |
| `SeriesStats`                | H_m 统计摘要            | mean/std/periodicity/trend…                          |
| `RetrievedEpisodeSummary`    | \mathcal{D}_m 元素摘要  | episode_id, score, event_summary                     |
| `ForecastTask`               | T_m                 | 七元组 + 路径 + S_m + stats + episodes                    |
| `PriorSet`                   | Z_m                 | priors 列表 + prompt/response 快照                       |
| `SearchResult`               | 搜索输出                | best_recipe_id, tau_loc_final, retain_ids…           |
| `ObservationResult`          | 慢环输出                | gain_summary, memory_tau, lambda_m, oos_forecasts    |


**设计要点**：`StructuredPrior.valid=False` 时带 `reject_reason`；`ForecastTask` 含 `history_path`/`oos_path` 指向 CSV 文件而非内嵌 DataFrame。

---

### `pipeline.py`

**作用**：**主编排入口**，封装完整 Phase 0→6 流程。


| 方法                            | 阶段        | 功能                                                             |
| ----------------------------- | --------- | -------------------------------------------------------------- |
| `preprocess(csv_path)`        | Phase 0   | 调用 `data/preprocessor.py`，输出 clean.csv                         |
| `build_tasks(clean_csv)`      | Phase 1   | `build_windows` + `build_forecast_task` → `List[ForecastTask]` |
| `generate_priors(task)`       | Phase 2   | 调用 LLM → `PriorSet`                                            |
| `run_search(task, prior_set)` | Phase 3+4 | `run_prior_guided_search` + `run_observation_feedback`         |
| `run_single_window(...)`      | 一键        | preprocess → T_m → Z_m → 可选 search → 落盘 JSON                   |
| `run_rolling_windows(...)`    | 多窗        | 滚动 walk-forward，memory 跨窗传递                                    |


**内部辅助**：`_init_run_logger`、`_load_prior_store`、`_initial_memory`（λ_m 冷启动）。

---

### `_path.py`

**作用**：将 AutoApex 仓库根目录加入 `sys.path`，使 `from autoapex.timeseries import TimeSeriesDataFrame` 可用。

**调用时机**：`evaluator.py`、`context/state.py`（heterogeneity 统计）、examples 脚本开头。

---

### `_bootstrap.py` / `_stdlib_patch/random.py`

**作用**：兼容旧 Python / 某些环境下的 `random.randbits` 缺失问题；demo 脚本 import 前执行。

---

### `secrets.py` / `secrets.example.py`

**作用**：本地存放 `DASHSCOPE_API_KEY`；`config.py` 在环境变量未设置时 fallback 读取。**不应提交到 git**。

---

### `__init__.py`

**作用**：包级懒加载，对外暴露 `PriorEnhancePipeline` 等符号。

---

## 4. data/ — 数据预处理与任务构建

### `data/preprocessor.py`

**作用**：将原始 CSV 规范化为 AutoApex / prior_enhance 统一格式。


| 函数                             | 说明                                                       |
| ------------------------------ | -------------------------------------------------------- |
| `_normalize_columns`           | 列名映射：`date/time`→`timestamp`，`power`→`value`，补 `item_id` |
| `_pandas_clean_and_fill`       | pandas ffill/bfill/interpolate，写 `fill_flag`             |
| `_try_autoapex_clean_and_fill` | 优先 AutoApex `TimeSeriesDataFrame.clean_and_fill`         |
| `preprocess_csv`               | **主入口**：读 CSV → 清洗 → 写 `{name}_clean.csv`                |
| `preprocess_via_serve_api`     | 可选：通过 serve 预处理接口                                        |


**论文对应**：构造 \mathcal{D}_{\text{full}}。

---

### `data/window.py`

**作用**：**最核心的数据切分模块**，实现 H_m/O_m 边界与 T_m 组装。


| 类/函数                         | 说明                                                               |
| ---------------------------- | ---------------------------------------------------------------- |
| `WindowSlice`                | 单窗口：`decision_time`, `history_df`, `oos_df`                      |
| `build_holdout_window`       | 默认：末尾 H 行 = O_m，其余 = H_m                                         |
| `build_rolling_windows`      | 滚动多窗，stride 可配                                                   |
| `build_windows`              | 按 `WindowConfig.mode` 分发                                         |
| `save_window_artifacts`      | 落盘 `{tag}_history.csv` / `{tag}_oos.csv`                         |
| `build_forecast_task`        | **组装 T_m**：stats → episode → S_m → EventContext → `ForecastTask` |
| `list_window_tasks_from_csv` | 从 CSV 一步到多个 ForecastTask                                         |


**关键参数**（`build_forecast_task`）：

- `event_text`：FNSPID 新闻摘要等用户事件
- `news_df`：历史新闻 DataFrame，注入 episode 库
- `known_covariates`：X_m^+ 列名列表

**论文对应**：\mathcal{H}_m、\mathcal{O}_m、T_m、信息边界 enforcement 的数据基础。

---

### `data/episode_store.py`

**作用**：历史 **Episode 库** \mathcal{D} 的构建与检索。


| 类/函数                                 | 说明                                            |
| ------------------------------------ | --------------------------------------------- |
| `Episode`                            | 单条 episode：时间切片 + 场景标签 + 事件文本 + 统计摘要          |
| `build_episode_store_from_history`   | 从 H_m 按固定步长切分 episode；可选 `news_df` 注入新闻       |
| `_event_context_from_news`           | 将 FNSPID 新闻对齐到 episode 决策时刻                   |
| `retrieve_episodes`                  | **Retrieve(T_m, D)**：场景匹配 + BGE 语义相似 + 时间衰减加权 |
| `episodes_for_novelty_pool`          | 为 ν_m 筛选同场景、决策时刻前的历史事件                        |
| `historical_event_texts_for_novelty` | 提取可比历史事件纯文本列表                                 |
| `parse_event_context`                | 将 `event_text` 或日历规则封装为 `EventContext`        |
| `RetrievedEpisode`                   | 检索结果 + `retrieval_score`                      |


**论文对应**：\mathcal{D}_m = \Retrieve(T_m, \mathcal{D})。

---

### `data/fnspid/` — FNSPID 金融数据工具链


| 文件                   | 作用                                                              |
| -------------------- | --------------------------------------------------------------- |
| `download_fnspid.py` | 从 HuggingFace/HTTP Range 下载单股 news/stock，避免全量 23GB              |
| `build_dataset.py`   | 一键：sample news + 官方 stock → `processed/AAL_daily.csv` 等         |
| `prepare_demo.py`    | 构建 demo 所需：`AAL_event_text.txt`、`AAL_news_events.json`、manifest |
| `README.md`          | FNSPID 数据说明与 ticker 混用警告                                        |
| `raw/`               | 原始 news/stock CSV、sample 文件                                     |
| `processed/`         | 清洗后日频序列、事件文本、news JSON                                          |


---

### `data/代理购电_小时.csv`

**作用**：默认 demo 用电负荷数据（小时频），`run_demo.py` 常用输入。

---

## 5. context/ — 上下文状态 S_m

### `context/state.py`

**作用**：计算 **S_m = \Gamma(T_m, \mathcal{D}_m)** 及序列统计。


| 函数                            | 论文公式                                                   |
| ----------------------------- | ------------------------------------------------------ |
| `_compute_n_m`                | n_m = \clip(N_m^{\text{eff}}/N_m^{\text{ref}}, 0, 1)   |
| `_compute_d_m`                | d_m = d_m^\mu + d_m^\sigma，场景内标准化后比较近期 vs 历史可比段        |
| `compute_event_novelty`（调用）   | \nu_m = 1 - s_m^{\max}                                 |
| `compute_context_state`       | **主入口**：组装 `ContextState`，含 ξ_m                        |
| `compute_series_stats`        | H_m 均值/方差/周期性/趋势 → `SeriesStats`                       |
| `compute_analysis_dimensions` | 多维诊断：volatility_cluster, outlier_rate, jump_intensity… |


**ξ_m 计算**（扩展项）：

```python
xi_m = 0.4 * volatility_cluster + 0.3 * outlier_rate + 0.3 * jump_intensity
```

---

### `context/event_novelty.py`

**作用**：事件新颖性 ν_m 的语义计算。


| 函数                                 | 说明                                  |
| ---------------------------------- | ----------------------------------- |
| `event_to_text` / `events_to_text` | EventItem → 纯文本                     |
| `build_calendar_event_context`     | 无用户事件时，用日历/场景合成 E_m                 |
| `compute_event_novelty`            | query 与历史 events 的 BGE 余弦 max → ν_m |
| `extract_historical_event_texts`   | 从 episode 列表抽文本                     |


---

### `context/text_encoder.py`

**作用**：中文语义编码器 **g(·)**，默认 BGE-base-zh-v1.5。


| 函数/类                                                | 说明                                                  |
| --------------------------------------------------- | --------------------------------------------------- |
| `resolve_encoder_model_path`                        | 优先本地 ModelScope 缓存，避免联网                             |
| `ChineseTextEncoder`                                | `encode()`、`max_similarity()`、`cosine_similarity()` |
| `get_text_encoder` / `get_text_encoder_from_config` | 全局单例                                                |


**环境变量**：`PRIOR_ENHANCE_TEXT_ENCODER_MODEL`、`PRIOR_ENHANCE_TEXT_ENCODER_LOCAL_ONLY=1`

---

### `context/labels.py`

**作用**：将连续 n_m, \nu_m, d_m 映射为离散 cond 标签。


| 函数                  | 输出标签                                     |
| ------------------- | ---------------------------------------- |
| `label_n_m`         | low_sample / medium_sample / high_sample |
| `label_nu_m`        | familiar_event / novel_event             |
| `label_d_m`         | stable / moderate_drift / severe_drift   |
| `build_cond_labels` | 三元组列表 + 各 label 字符串                      |


**阈值**：n_m 用 0.33/0.66；ν_m 用 0.5；d_m 用 0.33/0.66。

---

### `context/scenario.py`

**作用**：场景标签 c_m（工作日/周末/春节/国庆等），用于 n_m 有效样本与 episode 检索。


| 函数                           | 说明                            |
| ---------------------------- | ----------------------------- |
| `scenario_label(ts)`         | 单时间点 → 场景字符串                  |
| `scenario_labels_for_series` | 整列时间戳 → Series of labels      |
| `scenario_ref_count`         | 该场景理论最大参考样本数 N_m^{\text{ref}} |


---

### `context/frequency.py`

**作用**：频率 profile（小时/日/周），决定 seasonal_lag、近期窗口长度等。


| 类/函数                 | 说明                                             |
| -------------------- | ---------------------------------------------- |
| `FreqProfile`        | seasonal_lag, sub_seasonal_lag, recent_ratio 等 |
| `infer_freq_code`    | 从时间戳推断 5T/15T/h/D/W/M                          |
| `get_freq_profile`   | 查表                                             |
| `recent_window_size` | 计算漂移检测用的近期窗口长度                                 |


---

## 6. prior/ — LLM 先验 Z_m 与编译器

### `prior/generator.py`

**作用**：**调用 LLM 生成 Z_m** 的主入口。


| 函数                         | 说明                                     |
| -------------------------- | -------------------------------------- |
| `_extract_json_object`     | 从 LLM 回复中提取 JSON（含 markdown 围栏清理）      |
| `_parse_and_validate`      | JSON → `validate_prior_batch`          |
| `_build_retry_hint`        | 校验失败时的重试提示                             |
| `generate_priors_for_task` | **主入口**：prompt → LLM → 校验 → `PriorSet` |
| `_save_failed_response`    | 失败时保存原始回复到 logs                        |


**依赖**：`prompt_builder.py`、`llm_client.py`、`validator.py`、`coverage_rules.py`

---

### `prior/prompt_builder.py`

**作用**：构造 LLM system/user prompt，注入 T_m、S_m、stats、搜索空间、领域知识 K_m。


| 函数                              | 说明                                          |
| ------------------------------- | ------------------------------------------- |
| `build_prior_generation_prompt` | 完整 prompt 字符串；含 cond 默认值、AutoApex 搜索空间 JSON |


---

### `prior/llm_client.py`

**作用**：DashScope OpenAI 兼容 API 封装。


| 类/方法                 | 说明                                  |
| -------------------- | ----------------------------------- |
| `DashScopeLLMClient` | `chat(messages)` → 原始文本回复；支持 stream |


---

### `prior/validator.py`

**作用**：**Valid(z)=1** 校验：schema、搜索空间、action 前缀、cond 合法性。


| 函数                                        | 说明                                      |
| ----------------------------------------- | --------------------------------------- |
| `_parse_cond/_parse_target/_parse_action` | 从 LLM JSON 解析各字段                        |
| `_in_search_space`                        | target 各簇值是否在 `AUTOAPEX_SEARCH_SPACE` 内 |
| `validate_prior`                          | 单条 prior 校验，设置 valid/reject_reason      |
| `validate_prior_batch`                    | 批量校验 + 计算 b_evidence、confidence         |


---

### `prior/confidence.py`

**作用**：置信度校准 **b_{j,m}**。


| 函数                   | 论文公式                                                  |
| -------------------- | ----------------------------------------------------- |
| `compute_b_evidence` | 0.4n_m + 0.3(1-\nu_m) + 0.3\min(                      |
| `blend_confidence`   | b = \lambda b^{LLM} + (1-\lambda) b^{\text{evidence}} |


---

### `prior/search_space.py`

**作用**：定义 LLM 可输出的 **AutoApex 搜索空间** 候选列表（window_strategy、models、feature_strategy 等）。

**谁读它**：`validator.py`（合法性）、`prompt_builder.py`（告诉 LLM 可选值）、`anti_prior_builder.py`（补集采样）。

---

### `prior/coverage_rules.py`

**作用**：检查有效 prior 数量是否在 `[min_priors, max_priors]` 区间，不满足则触发 LLM 重试。

---

### `prior/compiler.py`

**作用**：**先验编译器** — 论文核心 z_{j,m} \mapsto \phi_{j,m}(r) 与 τ 初始化。


| 函数                      | 论文含义                                          |
| ----------------------- | --------------------------------------------- |
| `phi_prior_recipe`      | \phi_{j,m}(r)\in[0,1]：target 重叠 × action 不违反  |
| `phi_total`             | 多 prior confidence 加权平均 φ                     |
| `applicability`         | a_{j,m}：cond 与 S_m 标签匹配度                      |
| `init_tau_persistent`   | 从 prior_store 检索相似 prior → \tau_{j,m}         |
| `init_tau_loc`          | \tau_{j,m}^0 → 搜索起点 \tau^{\text{loc}}_{j,m,0} |
| `recipe_feature_vector` | 混杂控制向量 x_m(r)（Gain 回归用）                       |


**别名表**：`WINDOW_ALIAS`、`FEATURE_ALIAS`、`ENSEMBLE_ALIAS` — LLM 命名 → recipe catalog 命名。

**AVOID_HANDLERS**：`avoid_high_dimensional_tsfresh`、`avoid_full_history_window` 等 → 布尔检查函数。

---

### `prior/similarity.py`

**作用**：结构化 prior 之间的 **结构相似度** s(z, z')。


| 函数                                 | 说明                                                |
| ---------------------------------- | ------------------------------------------------- |
| `prior_structural_similarity`      | cond/target/action 集合重叠 + BGE 语义（可调权重）            |
| `prior_to_embed_text`              | prior 自然语言摘要 → 语义向量输入                             |
| `weighted_tau_from_similar_priors` | \tau_{j,m} = \frac{\sum s\tau_i}{\sum s+\epsilon} |


---

### `prior/llm_trust.py`

**作用**：全局 LLM 信任度 **λ_m** 冷启动。


| 函数                               | 说明                                  |
| -------------------------------- | ----------------------------------- |
| `lambda_m_init_from_leaderboard` | score 归一化 → \lambda_m^{\text{init}} |
| `resolve_lambda_m_init`          | 环境变量 / 配置 / 排行榜查表优先级                |


---

### `prior/llm_leaderboard.py`

**作用**：读取 `prior/data/llm_leaderboard.csv`（205 模型），查当前 `llm.model` 的综合得分。

---

### `prior/recipe_sampler.py`

**作用**：从 prior 的 target 簇**偏置采样**单个 recipe（非 softmax 路径的 guide 采样）。


| 函数                    | 说明                                                                  |
| --------------------- | ------------------------------------------------------------------- |
| `prior_guided_recipe` | 按 prior confidence 加权选一条 prior，从其 target 各簇随机 pick → `build_recipe` |


---

### `prior/data/llm_leaderboard.csv`

**作用**：模型名 → 智能指数分数，供 λ_m 冷启动查表。

---

## 7. recipe/ — 预测方案 r 的定义与构建

### `recipe/schema.py`

**作用**：论文 **Forecasting Recipe** r=(W,F,M,\theta,V,Q) 的 Pydantic 定义。


| 类                        | 分量  | 字段要点                                                          |
| ------------------------ | --- | ------------------------------------------------------------- |
| `SampleWindowSpec`       | W   | strategy, recent_ratio, min_history_points                    |
| `FeatureEngineeringSpec` | F   | mode, use_tsfresh, feature_config                             |
| `ModelSpec`              | M   | name, cluster                                                 |
| `HyperparameterSpec`     | θ   | model_name, params                                            |
| `ValidationSpec`         | V   | enabled, num_val_windows（预留）                                  |
| `OutputAndEnsembleSpec`  | Q   | enable_ensemble, ensemble_types                               |
| `ForecastingRecipe`      | r   | 聚合 + recipe_id（hash）+ time_limit + hyperparameter_tune_kwargs |


**recipe_id**：由 W/F/M/θ/Q 配置 hash 生成，用于去重与日志追踪。

---

### `recipe/catalog.py`

**作用**：**模型与特征 catalog** — AutoApex 支持的模型列表、簇分类、HPO schema、tsfresh handler 列表。


| 符号                 | 说明                                                    |
| ------------------ | ----------------------------------------------------- |
| `ModelCluster`     | STATISTICAL / ML_TABULAR / DEEP_LEARNING / FOUNDATION |
| `MODEL_CATALOG`    | 模型名 → cluster、默认 time_limit                           |
| `get_hpo_schema`   | 某模型可搜索超参字段                                            |
| `TRAINABLE_MODELS` | 当前可训练模型白名单                                            |


---

### `recipe/builder.py`

**作用**：从 catalog **构造具体 ForecastingRecipe 实例**。


| 函数                       | 说明                                                 |
| ------------------------ | -------------------------------------------------- |
| `build_recipe`           | 指定 model + window + feature + ensemble → 完整 recipe |
| `sample_random_recipe`   | 全空间随机采样（explore 用）                                 |
| `sample_recipe_batch`    | 批量随机                                               |
| `default_phase1_recipes` | 预设若干 baseline recipe                               |


---

### `recipe/autoapex_adapter.py`

**作用**：**Recipe → AutoApex API payload** 的适配层。


| 函数                                  | 说明                                    |
| ----------------------------------- | ------------------------------------- |
| `apply_sample_window`               | 按 W.recent_ratio 截断 history_df        |
| `recipe_to_hyperparameters_payload` | → `/api/train` 的 hyperparameters JSON |
| `recipe_to_train_payload`           | 完整 train 请求体                          |
| `infer_best_model_name`             | 集成场景下推断主模型名                           |


---

### `recipe/anti_prior_builder.py`

**作用**：探索分布 **p_exp** 的三路对照 recipe 生成。


| 函数                        | 类型                                    |
| ------------------------- | ------------------------------------- |
| `sample_p_exp_batch`      | 批量：random / low-prior / anti-prior 轮换 |
| `build_low_prior_recipe`  | 低 φ 随机 recipe                         |
| `build_anti_prior_recipe` | 显式违反 target/action 的合法 recipe         |


**论文对应**：Anti-prior but Feasible Recipes，用于因果对照。

---

## 8. validation/ — 验证切分、目标函数与评估

### `validation/split.py`

**作用**：在 **H_m 内部**构造时间有序 train-val 切分。


| 类/函数                      | 说明                                     |
| ------------------------- | -------------------------------------- |
| `ValSplit`                | train_df, val_df, train_end, val_start |
| `make_holdout_val_split`  | 末尾 val_horizon 点为 val（相对 train 是「未来」）  |
| `make_rolling_val_splits` | 多切片（实验用，主循环暂未接入）                       |


**约束**：`max(train) < min(val)` 且 train/val ⊂ H_m。

---

### `validation/objective.py`

**作用**：**J^val / J^obs 目标函数**。


| 函数                | 论文公式                                                    |
| ----------------- | ------------------------------------------------------- |
| `mape`            | 基础损失 L                                                  |
| `compute_cost`    | C_m(r)：wall_time 或 catalog 代理常数                         |
| `mu_from_context` | \mu_m = \mu(S_m)，含 d_m, ν_m, ξ_m                        |
| `validation_risk` | R^{\text{val}}：峰值低估率 + 尾部 MAE                           |
| `j_val`           | J^{\text{val}} = L + \lambda_C C + \mu_m R^{\text{val}} |
| `j_obs`           | J^{\text{obs}}（形式同 J^val，键名不同）                          |


---

### `validation/gain.py`

**作用**：**Validation / Observation Gain** 回归与 τ 更新。


| 类/函数                           | 论文含义                                                                                                  |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `GainResult`                   | beta, gain, delta, se, chi, accepted, reason                                                          |
| `_multivariate_ols_beta`       | J \sim \phi + \gamma^\top x + \varepsilon                                                             |
| `evidence_quality`             | \chi = q_{\text{sample}} \cdot q_{\text{balance}} \cdot q_{\text{quality}} \cdot q_{\text{stability}} |
| `compute_prior_gain`           | G = -\beta_\phi，\Delta = \clip(G/(\SE+\epsilon_G),...)                                                |
| `compute_contrastive_obs_gain` | 配对断层扫描备选估计                                                                                            |
| `update_tau_logit`             | l_{k+1} = (1-\omega)l_k + \eta\chi\Delta → sigmoid → τ                                                |


---

### `validation/evaluator.py`

**作用**：**Recipe 评估器** — 在 val 或 O_m 上得到预测与 metrics。


| 函数                   | 场景                                      |
| -------------------- | --------------------------------------- |
| `evaluate_on_split`  | H_m 内 train/val → J^val（搜索循环用）          |
| `evaluate_on_oos`    | 全量 H_m 训练 + O_m 推理 → J^obs（Phase 3/4 用） |
| `_proxy_forecast`    | 轻量 SeasonalNaive / Ridge 代理（快速调试）       |
| `_autoapex_forecast` | 真实 AutoApex train + infer               |


**后端选择**：`PRIOR_ENHANCE_EVAL_BACKEND=autoapex|proxy`

---

### `validation/autoapex_client.py`

**作用**：AutoApex serve **HTTP 客户端**。


| 类/方法                   | API                              |
| ---------------------- | -------------------------------- |
| `AutoApexClient.train` | POST `/api/train`                |
| `AutoApexClient.infer` | POST `/api/inference/v1`         |
| `TrainResult`          | model_id, wall_time_sec, metrics |
| `get_autoapex_client`  | 单例工厂                             |


**超时**：`AUTOAPEX_TRAIN_TIMEOUT`、`AUTOAPEX_INFER_TIMEOUT`

---

## 9. search/ — 窗口内 Prior-Guided Search

### `search/loop.py`

**作用**：**Prior-guided search 主循环**（Fast-Loop 核心）。


| 函数                        | 论文公式/流程                                                       |
| ------------------------- | ------------------------------------------------------------- |
| `_psi_score`              | \psi_{m,k}(r) = \sum_j \tau^{\text{loc}}_j \phi_j(r)          |
| `_sample_batch`           | p = (1-\rho)q + \rho p_{\exp} 混合采样                            |
| `run_prior_guided_search` | while trials < budget: 采样 → evaluate → Gain → τ^loc 更新 → 选 r* |


**输出**：`SearchResult` + 内部 `_evaluated_records`、`_best_recipe`、`_retain_recipes`（observation 用）。

**重要约束**：只读 `history_df`，**绝不读 oos_df**。

---

### `search/proposal.py`

**作用**：Softmax **提议分布 q_{m,k}**。


| 函数                     | 说明                             |
| ---------------------- | ------------------------------ |
| `softmax_weights`      | 数值稳定 softmax，温度 β_k            |
| `build_candidate_pool` | 从 prior 偏置 + 随机扩展生成候选 recipe 池 |
| `sample_batch_from_q`  | 按 ψ 分数 softmax 无放回抽样           |


---

## 10. observation/ — O_m 反馈与持久记忆

### `observation/feedback.py`

**作用**：**Observation Feedback 慢环** — O_m 揭晓后的审计与 memory 更新。


| 函数                         | 流程                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------ |
| `run_observation_feedback` | 对 retain recipes 算 J^obs → Gain 回归 → `update_persistent_memory` → append prior_store |


**输入**：`SearchResult.retain_ids` 对应的 recipe + O_m 真实值。

---

### `observation/memory.py`

**作用**：跨窗口 **持久 Prior Memory**。


| 类/函数                       | 论文公式                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `PriorMemory`              | 存储 `tau: Dict[prior_id, float]` 与 `lambda_m`                                          |
| `omega_forget_rate`        | \omega_m = \omega_{\max}\sigma(...)，受 S_m 驱动                                          |
| `update_persistent_memory` | l_{j,m+1} = (1-\omega_m)l_{j,m} + \eta_{\text{obs}}\chi\Delta^{\text{obs}}；更新 λ_{m+1} |


**与 Fast-Loop 分离**：此处更新的 τ 是持久层；search 内的 τ^loc 不写回此处（除非通过 observation 路径）。

---

### `observation/prior_store.py`

**作用**：**历史 prior JSONL 库**，供下一窗口 `init_tau_persistent` 检索。


| 类                   | 说明                                                   |
| ------------------- | ---------------------------------------------------- |
| `StoredPriorRecord` | window_index, decision_time, prior 快照, tau, scenario |
| `PriorStore`        | load/save/append_window/candidate_pairs              |


**默认路径**：`outputs/prior_store.jsonl`

---

## 11. pipeline.py — 端到端编排

见 [§3 pipeline.py](#pipelinepy)。补充 **典型调用链**：

```python
cfg = PriorEnhanceConfig.from_env()
pipe = PriorEnhancePipeline(cfg)

# 单窗口
summary = pipe.run_single_window("data.csv", with_search=True)

# 或分步
clean = pipe.preprocess("data.csv")
tasks = pipe.build_tasks(clean)
prior_set = pipe.generate_priors(tasks[0])
search_result, obs_result = pipe.run_search(tasks[0], prior_set)

# 滚动多窗
summaries = pipe.run_rolling_windows("data.csv", max_windows=5, with_search=True)
```

---

## 12. telemetry/ — 透明日志

### `telemetry/run_logger.py`

**作用**：结构化运行日志，支持 console + 文件 + JSONL audit。


| 类/函数                                 | 说明                                                              |
| ------------------------------------ | --------------------------------------------------------------- |
| `RunLogger`                          | run_id、phase、step_start/end、log_input/output/transform/artifact |
| `bind_run_logger` / `get_run_logger` | 上下文绑定                                                           |
| `log`（模块级 `_LoggerProxy`）            | 各子模块直接 `log.log_output(...)`                                    |


**产物**：

- `outputs/logs/{run_id}.log` — 人类可读
- `outputs/logs/{run_id}_audit.jsonl` — 机器可解析，逐步审计

---

## 13. examples/ — 可执行入口脚本


| 文件                         | 用途                             | 典型命令                                                      |
| -------------------------- | ------------------------------ | --------------------------------------------------------- |
| `run_demo.py`              | 代理购电 demo，Phase 1 或 1+2+3      | `python prior_enhance/examples/run_demo.py --with-search` |
| `run_fnspid_demo.py`       | FNSPID/AAL 金融 demo             | `--symbol AAL [--with-search]`                            |
| `validate_fnspid_paper.py` | 论文四阶段审计 + paper_audit.json     | `--symbol AAL --max-trials 8`                             |
| `download_text_encoder.py` | ModelScope 预下载 BGE 模型          | 首次环境准备                                                    |
| `simulate_fnspid.py`       | FNSPID 快速模拟（无 LLM）             | 数据管道测试                                                    |
| `show_recipe_space.py`     | 打印 AutoApex 搜索空间与 catalog      | 调试 LLM prompt                                             |
| `run_mutiagent.sh`         | mutiagent conda + serve 环境变量封装 | `bash prior_enhance/examples/run_mutiagent.sh ...`        |


---

## 14. 辅助与引导文件


| 文件                                   | 说明                                                |
| ------------------------------------ | ------------------------------------------------- |
| `requirements.txt`                   | pydantic, pandas, sentence-transformers, openai 等 |
| `docs/END_TO_END_LOGIC.md`           | 端到端逻辑与 Phase 对照                                   |
| `docs/IMPLEMENTATION_PLAN.md`        | 公式审计 + 月度计划                                       |
| `docs/ALGORITHM_PROJECT_ANALYSIS.md` | 算法深度分析（推荐阅读）                                      |
| `docs/FIVE_STEP_PROGRESS.md`         | 五步计划进度                                            |
| `docs/DATA_PIPELINE_WALKTHROUGH.md`  | 数据管道逐步说明                                          |
| `docs/CODE_LOGIC_DEBUG.md`           | Phase 1 速查                                        |
| `data/fnspid/README.md`              | FNSPID 数据说明                                       |


---

## 15. outputs/ — 运行产物约定


| 路径模式                                    | 内容             | 论文对象                      |
| --------------------------------------- | -------------- | ------------------------- |
| `preprocessed/*_clean.csv`              | 清洗全量数据         | \mathcal{D}_{\text{full}} |
| `windows/{tag}/*_history.csv`           | 历史可见段          | \mathcal{H}_m             |
| `windows/{tag}/*_oos.csv`               | 盲盒未来段          | \mathcal{O}_m             |
| `results/{tag}/task_Tm.json`            | 预测任务 + S_m     | T_m                       |
| `results/{tag}/priors_Zm.json`          | LLM 先验         | Z_m                       |
| `results/{tag}/search_result.json`      | 搜索摘要           | \mathcal{E}_m, r*, τ^loc  |
| `results/{tag}/observation_result.json` | 慢环结果           | Gain, τ, λ                |
| `results/{tag}/forecast_r_star.json`    | r* 在 O_m 预测    | \hat{y}_{\mathcal{O}_m}   |
| `results/{tag}/paper_audit.json`        | 四阶段审计          | 验证脚本专用                    |
| `results/{tag}/summary.json`            | run 总览         | pipeline 输出               |
| `prior_store.jsonl`                     | 跨 run prior 记忆 | \mathcal{P}_{j,m} 历史库     |
| `logs/{run_id}`*                        | 透明日志           | 调试/复现                     |


`{tag}` 常见值：`holdout`、`win_0000`、`fnspid_aal_holdout`

---

## 16. 模块依赖与调用链

### 16.1 Phase 1：T_m + S_m

```
preprocessor → window.build_holdout_window
            → window.build_forecast_task
                 ├─ context.state.compute_series_stats
                 ├─ context.state.compute_context_state
                 │    ├─ context.event_novelty.compute_event_novelty
                 │    └─ context.text_encoder
                 ├─ episode_store.build_episode_store_from_history
                 ├─ episode_store.retrieve_episodes
                 └─ episode_store.parse_event_context
```

### 16.2 Phase 2：Z_m

```
prior.generator.generate_priors_for_task
  ├─ prior.prompt_builder
  ├─ prior.llm_client
  ├─ prior.validator
  └─ prior.confidence
```

### 16.3 Phase 3：Search + r*

```
search.loop.run_prior_guided_search
  ├─ prior.compiler.init_tau_loc
  │    ├─ prior.similarity
  │    └─ observation.prior_store
  ├─ validation.split.make_holdout_val_split
  ├─ search.proposal.sample_batch_from_q
  ├─ recipe.anti_prior_builder.sample_p_exp_batch
  ├─ validation.evaluator.evaluate_on_split
  ├─ validation.gain.compute_prior_gain
  └─ validation.gain.update_tau_logit
```

### 16.4 Phase 4：Observation

```
observation.feedback.run_observation_feedback
  ├─ validation.evaluator.evaluate_on_oos
  ├─ validation.gain.compute_prior_gain / compute_contrastive_obs_gain
  ├─ observation.memory.update_persistent_memory
  └─ observation.prior_store.append_window
```

---

## 17. 环境变量速查


| 变量                                      | 作用                       | 默认                          |
| --------------------------------------- | ------------------------ | --------------------------- |
| `DASHSCOPE_API_KEY`                     | LLM API                  | —                           |
| `AUTOAPEX_BASE_URL`                     | serve 地址                 | `http://127.0.0.1:8000/api` |
| `PRIOR_ENHANCE_EVAL_BACKEND`            | autoapex / proxy         | autoapex                    |
| `PRIOR_ENHANCE_EVAL_FALLBACK_PROXY`     | serve 失败回退               | 0                           |
| `PRIOR_ENHANCE_SKIP_AUTOAPEX_STATS`     | 跳过 R heterogeneity       | 0                           |
| `PRIOR_ENHANCE_TEXT_ENCODER_MODEL`      | BGE 路径                   | 本地 ModelScope               |
| `PRIOR_ENHANCE_TEXT_ENCODER_LOCAL_ONLY` | 禁止联网下载                   | 1 推荐                        |
| `PRIOR_ENHANCE_TEXT_ENCODER_DEVICE`     | cpu / cuda               | cpu                         |
| `PRIOR_ENHANCE_LAMBDA_M`                | 覆盖 λ_m                   | —                           |
| `PRIOR_ENHANCE_MAX_TRIALS`              | 搜索预算                     | SearchConfig                |
| `PRIOR_ENHANCE_VAL_HORIZON`             | 内部 val 长度                | SearchConfig                |
| `PRIOR_ENHANCE_OBS_GAIN_METHOD`         | regression / contrastive | regression                  |
| `AUTOAPEX_TRAIN_TIMEOUT`                | 训练超时秒                    | 7200                        |


---

## 附录：按「我想改 X」快速定位文件


| 需求                    | 改哪个文件                                   |
| --------------------- | --------------------------------------- |
| 改 holdout 切分逻辑        | `data/window.py`                        |
| 改 n_m / d_m / ξ_m 公式  | `context/state.py`                      |
| 改事件新颖性 ν_m            | `context/event_novelty.py`              |
| 改 LLM prompt 内容       | `prior/prompt_builder.py`               |
| 改 prior 校验规则          | `prior/validator.py`                    |
| 改 φ 编译 / avoid 规则     | `prior/compiler.py`                     |
| 改搜索采样 / explore 比例    | `search/loop.py`, `config.SearchConfig` |
| 改 anti-prior 生成策略     | `recipe/anti_prior_builder.py`          |
| 改 Gain 回归 / χ 门控      | `validation/gain.py`                    |
| 改 J^val 损失或风险项        | `validation/objective.py`               |
| 改 AutoApex 请求 payload | `recipe/autoapex_adapter.py`            |
| 改持久 τ / λ 更新          | `observation/memory.py`                 |
| 改跨窗 prior 存储          | `observation/prior_store.py`            |
| 加新 demo 数据集           | `examples/` + `data/`                   |
| 改日志格式                 | `telemetry/run_logger.py`               |


---

*文档维护：新增模块或重命名文件后，请同步更新本文「目录树」与对应章节。*