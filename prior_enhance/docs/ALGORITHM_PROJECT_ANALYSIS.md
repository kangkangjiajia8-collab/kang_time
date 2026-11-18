# prior_enhance 算法项目分析文档

> **Evidence-Calibrated LLM Priors for Nonstationary Forecasting Recipe Search**  
> 工程实现：`prior_enhance/` · 数据示例：代理购电小时负荷 · 文档版本：2026-06

---

## 目录

1. [项目定位与核心问题](#1-项目定位与核心问题)
2. [系统总体架构](#2-系统总体架构)
3. [符号体系与论文对照](#3-符号体系与论文对照)
4. [Phase 1：任务建模与先验生成](#4-phase-1任务建模与先验生成)
5. [Phase 2：窗口内 Validation Search](#5-phase-2窗口内-validation-search)
6. [Phase 3：Observation Feedback 与持久记忆](#6-phase-3observation-feedback-与持久记忆)
7. [双反馈回路的设计哲学](#7-双反馈回路的设计哲学)
8. [Recipe 搜索空间与 φ 编译器](#8-recipe-搜索空间与-φ-编译器)
9. [Holdout 实验结果解读](#9-holdout-实验结果解读)
10. [实现状态矩阵](#10-实现状态矩阵)
11. [已知简化与局限](#11-已知简化与局限)
12. [运行与配置指南](#12-运行与配置指南)
13. [代码模块索引](#13-代码模块索引)
14. [后续演进路线](#14-后续演进路线)

---

## 1. 项目定位与核心问题

### 1.1 要解决什么

在非平稳时序预测场景中，**AutoApex 类系统**的 recipe 搜索空间极大：

```
recipe r = (W, F, M, θ, V, Q)
         = 样本窗口 + 特征工程 + 模型族 + 超参 + 验证协议 + 输出/集成
```

穷举或纯随机搜索代价过高。大模型（LLM）具备领域常识，能给出**可检验的结构化先验**（推荐什么、避免什么），但 LLM 也会：

- 过度自信（自报 b^LLM 偏高）
- 与当前数据上下文脱节
- 在局部验证集上过拟合

**prior_enhance** 的核心思路是：

> LLM 只负责**提议分布**（Proposal），不直接决定最终方案；  
> 所有决策由**历史可观测证据**（内部 validation + 事后 observation）校准。

### 1.2 与 AutoApex 的关系

```
┌─────────────────────────────────────────────────────────┐
│  prior_enhance（本仓库）                                  │
│  · 构造 T_m, S_m, Z_m                                    │
│  · prior-guided recipe search                            │
│  · Validation / Observation Gain                         │
│  · τ, λ 记忆更新                                         │
└──────────────────────┬──────────────────────────────────┘
                       │ recipe payload
                       ▼
┌─────────────────────────────────────────────────────────┐
│  AutoApex（外部 serve API）                               │
│  · /api/train  真实训练与预测                             │
│  · tsfresh / 模型库 / 集成                               │
└─────────────────────────────────────────────────────────┘
```

当前 Phase 2 使用**轻量代理评估器**跑通逻辑闭环；完整 AutoApex 训练接入为下一阶段工作。

### 1.3 设计原则（论文立论）


| 原则                                               | 含义                                |
| ------------------------------------------------ | --------------------------------- |
| **LLM generates testable priors, not forecasts** | 不输出预测数值，只输出结构化配置建议                |
| **No future leakage**                            | 决策时刻 t_m 前严禁使用 O_m                |
| **Validation ≠ Observation**                     | 局部 val 反馈只更新 τ^loc；持久 τ 只来自真实 O_m |
| **LLM 不进最终目标**                                   | r_m^ 的选取不含 LLM 评分项                |
| **Evidence gating**                              | χ 不足时拒绝更新，只记日志                    |


---

## 2. 系统总体架构

### 2.1 端到端数据流

```
原始 CSV
   │
   ▼ preprocess
清洗 CSV
   │
   ▼ build_windows (holdout)
H_m (3528)  │  O_m (72, 盲盒)
   │
   ├─ compute_context_state ──► S_m = (n_m, ν_m, d_m)
   ├─ retrieve_episodes ──────► D_m
   ├─ parse_event_context ────► E_m
   │
   ▼ assemble
T_m (ForecastTask)
   │
   ▼ LLM + Valid + b 校准
Z_m (PriorSet, 3~8 条 prior)
   │
   ▼ [--with-search]
   ├─ H_m 内部 train/val 切分
   ├─ prior-guided 采样 recipe
   ├─ J^val 评估 → G^val → τ^loc 更新
   ├─ r* = argmin J^val
   └─ E_m^retain 保留
   │
   ▼ O_m 解锁（holdout 可立即模拟）
   ├─ J^obs 评估
   ├─ G^obs → τ_{j,m+1} 持久更新
   └─ λ_{m+1} 全局 LLM 信任更新
```

### 2.2 目录结构与职责

```
prior_enhance/
├── config.py              # 全局配置（LLM / Window / Search / Encoder）
├── schemas.py             # T_m, S_m, Z_m, SearchResult, ObservationResult
├── pipeline.py            # PriorEnhancePipeline 编排入口
│
├── data/
│   ├── preprocessor.py    # CSV 清洗
│   ├── window.py          # H_m / O_m 切分，T_m 组装
│   └── episode_store.py   # 历史 episode 池 D，Retrieve → D_m
│
├── context/
│   ├── state.py           # S_m 三维估计
│   ├── scenario.py        # 场景标签 c_m（weekday/holiday/...）
│   ├── event_novelty.py   # ν_m = 1 - s_max
│   ├── text_encoder.py    # BGE 中文编码器 g(·)
│   └── labels.py          # cond 离散标签
│
├── prior/
│   ├── generator.py       # Z_m 生成主流程
│   ├── prompt_builder.py  # LLM prompt 构造
│   ├── validator.py       # Valid(z)  schema 校验
│   ├── confidence.py      # b = λ·b^LLM + (1-λ)·b^evidence
│   ├── compiler.py        # φ_{j,m}(r), τ^loc 初始化
│   └── recipe_sampler.py  # 从 Z_m 偏置采样 recipe
│
├── recipe/
│   ├── schema.py          # ForecastingRecipe r=(W,F,M,θ,V,Q)
│   ├── catalog.py         # AutoApex 模型/窗口 catalog
│   ├── builder.py         # 随机 / 指定 recipe 构造
│   └── autoapex_adapter.py# recipe → train API payload
│
├── validation/
│   ├── split.py           # H_m 内部 train-val 切分
│   ├── objective.py       # J^val, J^obs, C_m, R^val
│   ├── evaluator.py       # 代理评估器（当前非完整 AutoApex）
│   └── gain.py            # G^val/G^obs, χ 门控, τ logit 更新
│
├── search/
│   └── loop.py              # prior-guided search 主循环
│
├── observation/
│   ├── feedback.py          # O_m 观测反馈
│   └── memory.py            # 持久 τ, λ, ω_m 遗忘
│
├── telemetry/
│   └── run_logger.py        # 透明日志 + audit JSONL
│
└── examples/
    ├── run_demo.py          # 主入口
    └── run_mutiagent.sh     # mutiagent conda 环境脚本
```

### 2.3 入口与调用链

**主入口**：`examples/run_demo.py` → `PriorEnhancePipeline`

```python
# Phase 1 only
pipe.run_single_window(csv_path, target_col)

# Phase 1 + Phase 2 + Phase 3
pipe.run_single_window(csv_path, target_col, with_search=True)
# 内部调用:
#   run_search(task, prior_set)
#     → run_prior_guided_search()      # validation 回路
#     → run_observation_feedback()     # observation 回路
```

---

## 3. 符号体系与论文对照

### 3.1 窗口与时间


| 符号            | 含义     | 代码 / 数据                               |
| ------------- | ------ | ------------------------------------- |
| m             | 滚动窗口索引 | `ForecastTask.window_index`           |
| t_m           | 决策时刻   | `decision_time` = 2026-05-27 23:00:00 |
| \mathcal{H}_m | 可见历史   | `holdout_history.csv`, 3528 点         |
| \mathcal{O}_m | 盲盒未来   | `holdout_oos.csv`, 72 点               |
| H             | 预测跨度   | `horizon` = 72                        |
| B             | 搜索预算   | `search_budget` = 50（配置）              |


### 3.2 任务与上下文


| 符号    | 含义         | 代码                                 |
| ----- | ---------- | ---------------------------------- |
| T_m   | 预测任务       | `schemas.ForecastTask`             |
| E_m   | 事件上下文      | `EventContext`                     |
| D_m   | 检索 episode | `retrieved_episodes`, |D_m|=10     |
| S_m   | 上下文状态      | `ContextState`                     |
| n_m   | 有效样本量      | `n_m` = 0.1188 → `low_sample`      |
| \nu_m | 事件新颖性      | `nu_m` = 0.0134 → `familiar_event` |
| d_m   | 分布漂移       | `d_m` = 0.9519 → `severe_drift`    |
| \xi_m | 运行风险       | **未建模**                            |


### 3.3 先验与搜索


| 符号                     | 含义        | 代码                                            |
| ---------------------- | --------- | --------------------------------------------- |
| Z_m                    | 先验集合      | `PriorSet`                                    |
| z_{j,m}                | 单条先验      | `StructuredPrior` = (cond, target, action, b) |
| \phi_{j,m}(r)          | 先验满足度     | `prior/compiler.py::phi_prior_recipe`         |
| \psi_{m,k}(r)          | 综合先验分     | `search/loop.py::_psi_score`                  |
| \tau_{j,m,k}^{loc}     | 局部可靠性     | `tau_loc_final`（窗口内临时）                        |
| \tau_{j,m}             | 持久可靠性     | `PriorMemory.tau`                             |
| \lambda_m              | 全局 LLM 信任 | `PriorMemory.lambda_m`                        |
| r_m^                   | 最终方案      | `best_recipe_id`                              |
| \mathcal{E}_m^{retain} | 保留审计集     | `retain_ids`                                  |


### 3.4 目标与增益


| 符号                   | 含义               | 代码                         |
| -------------------- | ---------------- | -------------------------- |
| J_{m,k}^{val}(r)     | 局部验证目标           | `objective.j_val`          |
| J_m^{obs}(r)         | 真实观测目标           | `objective.j_obs`          |
| G_{j,m,k}^{val}      | Validation Gain  | `-β` from OLS in `gain.py` |
| G_{j,m}^{obs}        | Observation Gain | 同上，kind=`obs`              |
| \chi_{j,m,k}^{val}   | 证据质量门控           | `evidence_quality()`       |
| \Delta_{j,m,k}^{val} | 标准化增益            | `clip(G/(SE+ε), ±Δ_max)`   |


---

## 4. Phase 1：任务建模与先验生成

### 4.1 数据预处理

**模块**：`data/preprocessor.py`

- 解析时间戳，推断/指定频率（默认小时 `h`）
- 缺失值填充，生成 `fill_flag`
- 输出：`outputs/preprocessed/*_clean.csv`

**此时系统不知道** H_m/O_m 如何切分，只做清洗。

### 4.2 Holdout 窗口切分

**模块**：`data/window.py`

默认 **holdout 模式**（单窗口实验）：

```
全量 N 点 = H_m (N-H) + O_m (H)
H = forecast_days × points_per_day = 3 × 24 = 72
```

**信息边界断言**：

```python
max(H_m.timestamp) == t_m          # 历史截止于决策时刻
min(O_m.timestamp) > t_m           # O_m 是未来
build_forecast_task() 不读取 O_m 的 target 用于 S_m 或 prompt
```

### 4.3 上下文状态 S_m

**模块**：`context/state.py`

#### n_m：有效样本量

```
c_m = scenario_label(t_m)          # 如 weekday
N_m^eff = count(同类场景且 y 有效的历史点)
N_m^ref = 参考池总量
n_m = clip(N_m^eff / N_m^ref, 0, 1)
```

Holdout 实测：`n_m=0.1188` → 标签 `low_sample`（同类 weekday 样本相对参考池偏少）。

#### \nu_m：事件新颖性

**模块**：`context/event_novelty.py` + `context/text_encoder.py`

```
g(·) = BAAI/bge-base-zh-v1.5 中文句向量
s_max = max cosine(g(E_m), g(E_i))  for E_i ∈ E_m^hist
ν_m = 1 - s_max
```

Holdout 实测：`s_max=0.9866`, `ν_m=0.0134` → `familiar_event`（当前 weekday 预测场景与历史极相似）。

#### d_m：分布漂移

```
d_m^μ = |mean(recent) - mean(baseline)| / δ
d_m^σ = |std(recent) - std(baseline)| / δ
d_m = clip(d_m^μ + d_m^σ, 0, 1)
```

Holdout 实测：`d_m=0.9519`, `d_m^μ=0.925` → `severe_drift`（近期均值相对基线大幅上移：recent_mean≈2915 vs baseline≈2198）。

#### cond 标签

```python
cond_labels = [n_label, nu_label, d_label]
# → ["low_sample", "familiar_event", "severe_drift"]
```

每条 prior 的 `cond` 字段必须从此集合（或其子集）选取，用于 `a_j(S_m)` 适用性判断。

### 4.4 Episode 检索 D_m

**模块**：`data/episode_store.py`

- 从历史 H_m 滑动切分 episode（每个 episode 含 decision_time、scenario、event_summary）
- `Retrieve(T_m, D)`：按场景 + 语义相似度返回 top-k
- Holdout：D_m = 10，最高 score ≈ 0.9845

**与 ν_m 的分工**：


| 机制         | 用途                                  |
| ---------- | ----------------------------------- |
| D_m 检索     | prompt 参考 + b^evidence 中的 episode 项 |
| E_m^hist 池 | ν_m 新颖性计算（排除近邻块，防泄漏）                |


### 4.5 LLM 先验生成 Z_m

**模块**：`prior/generator.py`

**流程**：

```
build_prior_generation_prompt(T_m, S_m, D_m, search_space)
    → DashScopeLLMClient.chat()  [deepseek-r1-distill-qwen-1.5b]
    → JSON 解析 + validate_prior_batch()
    → coverage_rules 检查（条数、字段覆盖）
    → compute_b_evidence + blend_confidence
    → PriorSet
```

**单条 prior 结构**：

```json
{
  "prior_id": "prior_001",
  "cond": ["low_sample", "familiar_event", "severe_drift"],
  "target": {
    "window_strategy": ["recent_90d", "exp_decay"],
    "feature_strategy": ["none", "calendar"],
    "models": ["LightGBM", "RecursiveTabular"],
    "hyperparameter_policy": ["default"],
    "ensemble_strategy": ["weighted_average"]
  },
  "action": ["avoid_full_history_window", "avoid_high_dimensional_tsfresh"],
  "b_llm": 0.72,
  "b_evidence": 0.6435,
  "confidence": 0.6817
}
```

**置信度校准**：

```
b^evidence = 0.4·n_m + 0.3·(1-ν_m) + 0.3·min(|D_m|/K, 1)
b = λ·b^LLM + (1-λ)·b^evidence        # λ=0.5 默认
```

Holdout prior_001 示例：`0.5×0.72 + 0.5×0.6435 = 0.6817`

**Valid(z) 校验**（`prior/validator.py`）：

- cond 标签合法
- target 各字段值在 search_space catalog 内
- action 为 `avoid_`* 格式
- b_llm ∈ [0,1]

---

## 5. Phase 2：窗口内 Validation Search

> **绝对纪律**：此阶段只使用 H_m；O_m 不可见；τ^loc 更新不写持久 τ。

### 5.1 内部 Train-Validation 切分

**模块**：`validation/split.py`

```
H_m (3528) → train (3480) + val (48)
约束: max(train) < min(val) ⊂ H_m
```

- `val_horizon` 默认 48（小时）
- 此 val 区间 ≠ O_m（72 点）；前者是 H_m 内尾部，后者是 H_m 之后

### 5.2 τ^loc 初始化

**模块**：`prior/compiler.py::init_tau_loc`

```
a_j(S_m) = |cond_j ∩ S_m_labels| / |cond_j|
blended = λ·confidence + (1-λ)·0.5
τ^loc_{j,0} = clip(a_j × blended, ε, 1-ε)
```


| prior     | cond 匹配 | τ^loc 初始                                |
| --------- | ------- | --------------------------------------- |
| prior_001 | 3/3 ✅   | ≈ 0.682                                 |
| prior_002 | 3/3 ✅   | ≈ 0.662                                 |
| prior_003 | 0/3 ❌   | ≈ ε（cond 写了 high_sample/moderate_drift） |


### 5.3 搜索主循环

**模块**：`search/loop.py`

**配置**（`SearchConfig`）：


| 参数                   | 默认   | 含义            |
| -------------------- | ---- | ------------- |
| max_trials           | 12   | 最多评估 recipe 数 |
| batch_size           | 4    | 每轮批量          |
| explore_ratio        | 0.25 | 随机探索比例        |
| val_horizon          | 48   | 内部验证长度        |
| min_samples_for_gain | 4    | Gain 回归最少样本   |
| chi_min              | 0.25 | 证据门控阈值        |
| omega_val            | 0.2  | τ^loc 遗忘率     |
| eta_val              | 0.15 | τ^loc 学习率     |


**每轮 k 的流程**：

```
1. 采样 batch:
   · (1-explore_ratio) 条: prior_guided_recipe(Z_m)  按 confidence 加权选 prior
   · explore_ratio 条: sample_random_recipe()         纯随机对照

2. 对每个 recipe r:
   · train_df 上训练代理模型
   · val_df 上预测 → 算 J^val, φ, ψ

3. 若 |E_{m,k}| >= min_samples_for_gain:
   · 对每条 prior j: OLS 回归 J ~ φ_j
   · G^val = -β
   · Δ^val = clip(G/(SE+ε_G), ±Δ_max)
   · χ^val 门控 → 通过则 logit 更新 τ^loc

4. 重复直到 trials 用尽
```

### 5.4 验证目标函数

**模块**：`validation/objective.py`


J_{m,k}^{val}(r) = L_{m,k}^{val}(r) + \lambda_C C_m(r) + \mu_m R_{m,k}^{val}(r)



| 项       | 实现                                   |
| ------- | ------------------------------------ |
| L^{val} | MAPE                                 |
| C_m     | 模型簇开销 + tsfresh + 集成 + time_limit 代理 |
| \mu_m   | `0.2 + 0.4·d_m + 0.2·ν_m`（ξ_m 未建模）   |
| R^{val} | 峰值低估率 + 归一化尾部 MAE                    |


Holdout 下 \mu_m ≈ 0.58（d_m 高 → 风险惩罚强）。

### 5.5 Validation Gain 与 τ^loc 更新

**模块**：`validation/gain.py`

**回归**（当前简化为一元 OLS）：


J^{val}(r) = \alpha + \beta \phi_j(r) + \varepsilon



G^{val} = -\beta, \quad \Delta^{val} = \text{clip}\left(\frac{G^{val}}{SE + \epsilon_G}, -\Delta_{max}, \Delta_{max}\right)


**证据质量**：


\chi^{val} = q_{sample} \cdot q_{balance} \cdot q_{quality} \cdot q_{stability}



| 因子          | 当前实现                         |
| ----------- | ---------------------------- |
| q_sample    | min(1, n/min_samples)        |
| q_balance   | 0.7 占位（无反先验时不归零）             |
| q_quality   | |corr(φ, J)| clip 到 [0.2, 1] |
| q_stability | 0.8 占位                       |


**Logit 更新**（仅 τ^loc）：


l_{j,k+1}^{loc} = (1-\omega_{val}) l_{j,k}^{loc} + \eta_{val} \chi^{val} \Delta^{val}


\tau_{j,k+1}^{loc} = \sigma(l_{j,k+1}^{loc})


### 5.6 最终方案选择


r_m^ = \arg\min_{r \in \mathcal{E}_m} J_m^{val}(r) \quad (\kappa=0)


**不含** -\eta_s \cdot \text{LLM}(r) — 论文明确要求删除。

### 5.7 保留集 \mathcal{E}_m^{retain}

```
E_top     = J_val 最小的 top_k 个（默认 3）
E_explore = φ 最低的 explore_k 个（默认 2）
E_retain  = E_top ∪ E_explore
```

供 Observation Gain 因果对照（高 φ vs 低 φ）。

---

## 6. Phase 3：Observation Feedback 与持久记忆

> **目的**：清算 LLM 先验在真实世界的对错，更新跨窗口持久记忆 — **不是**回溯改选 r*。

### 6.1 触发条件

- Holdout 实验：O_m CSV 立即可读
- 在线部署：等待 H 步真实值返回

### 6.2 观测目标

对 r \in \mathcal{E}_m^{retain}：


J_m^{obs}(r) = L_m^{obs}(r) + \lambda_C C_m(r) + \mu_m R_m^{obs}(r)


- 在**全 H_m** 上重训代理模型
- 预测 O_m，与真实 y_{\mathcal{O}_m} 比较

### 6.3 Observation Gain

同 Validation Gain，在 retain 集上回归，G^{obs} = -\beta^{obs}。

Holdout 实测：


| prior     | G^obs  | χ^obs | accepted | 解读                      |
| --------- | ------ | ----- | -------- | ----------------------- |
| prior_001 | +0.145 | 0.323 | ✅        | 顺应 prior_001 → O_m 损失更低 |
| prior_002 | -0.107 | 0.301 | ✅        | 顺应 prior_002 → O_m 损失更高 |
| prior_003 | +0.079 | 0.218 | ❌        | χ < χ_min，拒绝更新          |


### 6.4 持久 Prior Memory

**模块**：`observation/memory.py`

**动态遗忘率**（非平稳环境）：


\omega_m = \omega_{max} \cdot \sigma(\alpha_0 - n_m + \nu_m + d_m)


d_m 高 → ω_m 大 → 更快遗忘陈旧经验。

**持久 τ 更新**：


l_{j,m+1} = (1-\omega_m) l_{j,m} + \eta_{obs} \chi^{obs} \Delta^{obs}


Holdout 结果：`τ: 0.5 → {0.506, 0.495, 0.500}`（保守小幅调整）。

**全局 λ 更新**：


\lambda_{m+1} = (1-\eta_\lambda)\lambda_m + \eta_\lambda \frac{\sum_j \chi_j \tau_{j,m+1}}{\sum_j \chi_j + \epsilon}


证据不足时：\lambda_{m+1} = \lambda_m。

### 6.5 τ^loc vs τ 持久 对比


| 维度     | τ^loc       | τ 持久         |
| ------ | ----------- | ------------ |
| 生命周期   | 单窗口搜索内      | 跨窗口          |
| 更新源    | G^val       | G^obs        |
| 学习率    | η_val=0.15  | η_obs=0.10   |
| 波动幅度   | 大（0.5→0.59） | 小（0.5→0.506） |
| 能否写回对方 | ❌           | —            |


---

## 7. 双反馈回路的设计哲学

```
                    ┌─────────────────────────────────┐
                    │         决策时刻 t_m             │
                    │   可见: H_m, S_m, E_m, D_m, Z_m  │
                    └───────────────┬─────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
              ▼                                           ▼
   ┌──────────────────────┐                 ┌──────────────────────┐
   │  Validation 回路      │                 │  Observation 回路     │
   │  数据: H_m 内 val     │                 │  数据: 真实 y_Om      │
   │  更新: τ^loc          │                 │  更新: τ 持久, λ      │
   │  用途: 搜索中途 refine │                 │  用途: 跨窗 memory    │
   │  风险: val overfit    │                 │  风险: 样本少/延迟     │
   └──────────────────────┘                 └──────────────────────┘
              │                                           │
              └─────────────────┬─────────────────────────┘
                                ▼
                    r* 由 J^val 客观选出（不含 LLM 分）
```

**为何 Validation 不能写持久 τ？**

局部 val 被搜索循环反复「压榨」——同一份 val 数据被用于评估 dozens of recipes 并驱动 τ^loc 迭代。若将此信号写入长期记忆，系统会对**当前窗口的 val 切分**过拟合，而非对真实未来泛化。

**为何需要 Observation？**

只有 O_m 上的真实误差才能无偏地评估「顺应 prior j 是否真的更好」。这是跨窗口因果校准的唯一可靠来源。

---

## 8. Recipe 搜索空间与 φ 编译器

### 8.1 Recipe 结构

**模块**：`recipe/schema.py`

```
r = (W, F, M, θ, V, Q)
W: 样本窗口策略（full_history / recent_XXpct / exponential_decay）
F: 特征工程（none / tsfresh_comprehensive / tsfresh_custom）
M: 模型族（SeasonalNaive / AutoARIMA / RecursiveTabular / TFT / ...）
θ: 超参数
V: 内部验证协议（预留）
Q: 输出模式 + 集成策略
```

每个 recipe 可编译为 AutoApex `/api/train` payload（`recipe/autoapex_adapter.py`）。

### 8.2 φ_{j,m}(r) 编译逻辑

**模块**：`prior/compiler.py`

```
φ_j(r) = pos_score × avoid_ok

pos_score = mean([
  overlap(window_strategy),
  overlap(feature_strategy),
  overlap(models),
  overlap(ensemble_strategy),
])

avoid_ok = 0  if 任一 action 违反
         = 1  otherwise
```

**action 处理器示例**：


| action                            | 检查                                     |
| --------------------------------- | -------------------------------------- |
| avoid_full_history_window         | W ≠ full_history 或 recent_ratio < 0.99 |
| avoid_high_dimensional_tsfresh    | F.use_tsfresh == False                 |
| avoid_large_ensemble              | Q.enable_ensemble == False             |
| avoid_high_capacity_neural_models | M.name ∉ 神经网络集合                        |


**名称映射**（LLM → catalog）：

```
recent_90d → recent_75pct
LightGBM   → RecursiveTabular
tsfresh_statistical → tsfresh_comprehensive
```

### 8.3 ψ 综合分


\psi_{m,k}(r) = \sum_j \tau_{j,m,k}^{loc} \cdot \phi_{j,m}(r)


用于搜索过程中的偏置参考；*不参与 r 最终选取**。

---

## 9. Holdout 实验结果解读

基于 `outputs/results/holdout/` 的真实运行结果。

### 9.1 上下文摘要


| 指标       | 值                                          |
| -------- | ------------------------------------------ |
| 决策时刻     | 2026-05-27 23:00                           |
| H_m      | 3528 点（2026-01-01 ~ 05-27）                 |
| O_m      | 72 点（3 天）                                  |
| S_m      | low_sample + familiar_event + severe_drift |
| 有效 prior | 3 条                                        |


### 9.2 搜索评估（7 recipes, 3 rounds）


| rank | recipe       | model                      | W            | J_val     | φ     | ψ     |
| ---- | ------------ | -------------------------- | ------------ | --------- | ----- | ----- |
| 1★   | 4c1eb302d452 | MovingAverageInterpolation | exp_decay    | **0.373** | 0.503 | 0.639 |
| 2    | 4df25b5db8ab | AutoARIMA                  | exp_decay    | 0.581     | 0.670 | 0.834 |
| 3    | 4fe8f07a9470 | SeasonalNaive              | recent_25pct | 0.582     | 0.421 | 0.493 |
| ...  | ...          | ...                        | ...          | ...       | ...   | ...   |


**关键发现**：

1. *r 来自 random explore**，非 LLM 直接推荐的 LightGBM — 说明当前代理评估器下先验引导未必命中最优，但 observation 仍可审计 prior 因果效应。
2. 多个统计模型 **L_val 相同（0.205）** — 代理评估器对 AutoARIMA/ETS/SeasonalNaive 降级为相同 seasonal naive 逻辑。
3. **prior_001 的 τ^loc 从 0.68 升至 0.59**（G^val>0）；**prior_002 略降**（G^val<0）。

### 9.3 观测反馈（5 retain recipes）


| recipe         | J_obs     | 备注            |
| -------------- | --------- | ------------- |
| 4c1eb302d452 ★ | **0.377** | val 与 obs 双第一 |
| 4df25b5db8ab   | 0.548     | 高 φ 对照        |
| 4fe8f07a9470   | 0.549     | 低 φ explore   |


**持久记忆更新**：

```
τ: prior_001 0.500 → 0.506  (+)
   prior_002 0.500 → 0.495  (-)
   prior_003 0.500 → 0.500  (χ 不足，不变)
λ: 0.500 → 0.50005
```

---

## 10. 实现状态矩阵


| 论文模块             | 符号                | 状态    | 代码                        |
| ---------------- | ----------------- | ----- | ------------------------- |
| 数据清洗             | —                 | ✅     | `data/preprocessor.py`    |
| 窗口切分             | H_m, O_m          | ✅     | `data/window.py`          |
| 预测任务             | T_m               | ✅     | `schemas.ForecastTask`    |
| 事件上下文            | E_m               | ✅     | `EventContext`            |
| Episode 检索       | D_m               | ✅     | `episode_store.py`        |
| 上下文状态            | S_m=(n_m,ν_m,d_m) | ✅     | `context/state.py`        |
| 中文编码器            | g(·)              | ✅     | BGE `bge-base-zh-v1.5`    |
| 运行风险             | ξ_m               | ❌     | 未建模                       |
| LLM 先验           | Z_m               | ✅     | `prior/generator.py`      |
| 置信度校准            | b                 | ✅     | `prior/confidence.py`     |
| φ 编译             | φ_{j,m}(r)        | ✅     | `prior/compiler.py`       |
| ψ 综合分            | ψ_{m,k}(r)        | ✅     | `search/loop.py`          |
| 内部 val 切分        | train/val ⊂ H_m   | ✅     | `validation/split.py`     |
| J^val / J^obs    | 目标函数              | ✅     | `validation/objective.py` |
| 代理评估             | —                 | ⚠️    | 轻量 proxy，非完整 AutoApex     |
| Validation Gain  | G^val, χ          | ✅     | `validation/gain.py`      |
| τ^loc 更新         | 局部                | ✅     | `gain.update_tau_logit`   |
| r* 选择            | argmin J^val      | ✅     | `search/loop.py`          |
| E_m^retain       | 保留集               | ✅     | top + explore             |
| Observation Gain | G^obs             | ✅     | `observation/feedback.py` |
| 持久 τ 更新          | τ_{j,m+1}         | ✅     | `observation/memory.py`   |
| 全局 λ 更新          | λ_{m+1}           | ✅     | `observation/memory.py`   |
| 反先验对照采样          | anti-prior        | ❌     | 刻意延后                      |
| 配对断层扫描           | P_{j,m}           | ❌     | 备选方案未实现                   |
| AutoApex 真实训练    | Train(r*, H_m)    | ❌     | adapter 已有，未接 serve       |
| 滚动多窗口            | m=1,2,...         | ⚠️    | rolling 模式预留，主测 holdout   |
| LLM 分进 r*        | —                 | ✅ 已删除 | κ=0, 纯 J^val              |


---

## 11. 已知简化与局限

### 11.1 代理评估器

当前 `validation/evaluator.py` 不调用 AutoApex `/api/train`，而是：

- 统计模型 → seasonal naive（lag=168/24）
- 表格模型 → Ridge 日历特征回归
- 其他 → 降级为 naive

**影响**：多个模型族 L_val/L_obs 相同；r* 可能不代表真实 AutoApex 最优。

### 11.2 无反先验对照

`q_balance=0.7` 占位，未强制采样「违反 prior」的方案。因果 Gain 估计偏保守，β 的标准误可能偏大。

### 11.3 ξ_m 未建模

μ_m 和 ω_m 用 d_m、ν_m、n_m 代理，缺少显式运行风险（覆盖率不足、尾部误差等）。

### 11.4 χ 的 q_stability

固定 0.8 占位，未做时间交叉切片稳定性检验。

### 11.5 prior cond 与 S_m 不一致

LLM 可能输出与当前 S_m 不完全匹配的 cond（如 prior_003 写了 high_sample）。系统通过 `a_j(S_m)=0` 降低 τ^loc 初始值，但不会 reject 该 prior。

### 11.6 单窗口 holdout

当前主实验为单一 holdout 窗口；rolling 多窗口链式 τ/λ 传递尚未充分测试。

---

## 12. 运行与配置指南

### 12.1 环境

```bash
conda activate mutiagent
cd /mnt/raid/forecast/Valrant/GZSB/prob/autoapex

# 预下载 BGE 编码器
python prior_enhance/examples/download_text_encoder.py

# 配置 API Key
export DASHSCOPE_API_KEY=sk-xxx
```

### 12.2 运行命令

```bash
# Phase 1：T_m + Z_m
bash prior_enhance/examples/run_mutiagent.sh \
  --clean-csv prior_enhance/outputs/preprocessed/代理购电_小时_clean.csv

# Phase 1 + 2 + 3：含 search + observation
bash prior_enhance/examples/run_mutiagent.sh \
  --clean-csv prior_enhance/outputs/preprocessed/代理购电_小时_clean.csv \
  --with-search --max-trials 8
```

### 12.3 主要配置项


| 配置类               | 关键字段                             | 默认               |
| ----------------- | -------------------------------- | ---------------- |
| WindowConfig      | forecast_days × points_per_day   | 3×24=72          |
| LLMConfig         | model, min_priors, max_priors    | deepseek-r1, 3~8 |
| PriorConfig       | confidence_lambda                | 0.5              |
| SearchConfig      | max_trials, val_horizon, chi_min | 12, 48, 0.25     |
| TextEncoderConfig | model_name                       | bge-base-zh-v1.5 |


### 12.4 环境变量


| 变量                                    | 作用               |
| ------------------------------------- | ---------------- |
| `PRIOR_ENHANCE_SKIP_AUTOAPEX_STATS=1` | 跳过 R 异质性统计（默认开启） |
| `PRIOR_ENHANCE_MAX_TRIALS`            | 搜索 trial 上限      |
| `PRIOR_ENHANCE_VAL_HORIZON`           | 内部 val 长度        |
| `PRIOR_ENHANCE_TEXT_ENCODER_MODEL`    | BGE 模型路径         |
| `DASHSCOPE_API_KEY`                   | LLM API          |


### 12.5 输出产物

```
prior_enhance/outputs/
├── preprocessed/          # 清洗 CSV
├── windows/holdout/       # H_m, O_m CSV
├── results/holdout/
│   ├── task_Tm.json       # T_m + S_m
│   ├── priors_Zm.json     # Z_m
│   ├── search_result.json # Phase 2（--with-search）
│   ├── observation_result.json  # Phase 3
│   └── summary.json
└── logs/
    ├── run_*.log          # 人类可读
    └── run_*_audit.jsonl  # 机器审计
```

---

## 13. 代码模块索引


| 若你要改…       | 看这里                                            |
| ----------- | ---------------------------------------------- |
| 窗口怎么切       | `data/window.py`                               |
| S_m 怎么算     | `context/state.py`                             |
| ν_m 中文语义    | `context/text_encoder.py`, `event_novelty.py`  |
| LLM prompt  | `prior/prompt_builder.py`                      |
| prior 校验规则  | `prior/validator.py`, `coverage_rules.py`      |
| b 校准公式      | `prior/confidence.py`                          |
| φ 怎么编译      | `prior/compiler.py`                            |
| recipe 怎么采样 | `prior/recipe_sampler.py`, `recipe/builder.py` |
| 搜索主循环       | `search/loop.py`                               |
| J^val 公式    | `validation/objective.py`                      |
| Gain 回归     | `validation/gain.py`                           |
| O_m 反馈      | `observation/feedback.py`                      |
| 持久记忆        | `observation/memory.py`                        |
| 编排入口        | `pipeline.py`                                  |
| 日志          | `telemetry/run_logger.py`                      |


---

## 14. 后续演进路线

### 短期（跑通真实训练）

1. **接 AutoApex serve API**：`evaluator.py` 替换 proxy → 真实 `/api/train`
2. *r 全量重训**：`Train(r*, H_m)` → 正式预测 O_m
3. **滚动多窗口**：验证 τ/λ 链式传递

### 中期（因果估计增强）

1. **反先验 / 低 φ 对照采样**：恢复 q_balance 严格门控
2. **多元回归**：J ~ φ_j + x_m(r) 完整控制混杂
3. **配对断层扫描**：P_{j,m} twin matching 备选 Gain

### 长期（论文完整版）

1. **ξ_m 运行风险**：CoverageGap, TailRisk, PeakUnderestimation
2. **q_stability 交叉切片**：多 val split 稳定性
3. **p_{m,k} softmax 提议分布**：ψ 驱动采样概率
4. **LLM τ 注入 prompt**：下一窗生成时参考持久 memory

---

## 附录 A：信息边界检查表


| 数据        | Phase 1 | Search | Observe |
| --------- | ------- | ------ | ------- |
| H_m       | ✅       | ✅      | ✅ 全量重训  |
| O_m 真实值   | ❌       | ❌      | ✅       |
| 内部 val    | —       | ✅      | ✅（已属历史） |
| 写 τ 持久    | ❌       | ❌      | ✅       |
| 写 τ^loc   | —       | ✅      | ❌       |
| LLM 分进 r* | ❌       | ❌      | ❌       |


## 附录 B：相关文档

- [DATA_PIPELINE_WALKTHROUGH.md](./DATA_PIPELINE_WALKTHROUGH.md) — 数据管道逐步说明
- [CODE_LOGIC_DEBUG.md](./CODE_LOGIC_DEBUG.md) — Phase 1 代码速查
- [../README.md](../README.md) — 快速开始

---

*文档由 prior_enhance 代码库与 holdout 实验结果自动生成分析，如有算法变更请同步更新本文档。*