# prior_enhance 详细实现计划与困难分析

> **起点日期**：2026-06-12  
> **对应论文**：Evidence-Calibrated LLM Priors for Nonstationary Forecasting Recipe Search  
> **工程目录**：`prior_enhance/`  
> **文档版本**：v7 · 2026-06-12（排行榜 CSV + softmax 提议 ρ=0）

---

## 目录

1. [执行摘要](#一执行摘要)（含 [§1.4 Prior 初始化](#14-prior-初始化与-llm-信任已接入--2026-06-12)）
2. [论文未实现条文（原文摘录）](#二论文未实现条文原文摘录)
3. [论文公式实现对照表](#三论文公式实现对照表)
4. [论文逐节实现审计（代码核实）](#三附论文逐节实现审计代码核实)
5. [核心困难分析](#四核心困难分析)
6. [一个月实施计划（2026-06-12 ~ 07-11）](#五一个月实施计划2026-06-12--07-11)
7. [风险登记册](#六风险登记册)
8. [验收标准与里程碑](#七验收标准与里程碑)
9. [待决事项](#八待决事项)

---

## 一、执行摘要

### 1.1 当前状态

- **Phase 1→3 算法闭环**：数据 → \(T_m\) → \(S_m\) → \(Z_m\) → search → \(\tau^{\text{loc}}\) → \(r^*\) → observation → 持久 \(\tau, \lambda\) 已编排完成。
- **AutoApex 真实训练/推理**：已接入并通过 mutiagent 验证（2026-06-12，`max-trials=1`，AutoARIMA，`J_val=0.354`，`J_obs=0.351`）。
- **Prior 初始化 + LLM 信任（公式 11–13）**：`prior/similarity.py`、`observation/prior_store.py`、`prior/llm_trust.py` + **`prior/data/llm_leaderboard.csv`（205 模型）**。
- **Softmax 提议分布（公式 16，β=1，ρ=0）**：`search/proposal.py`；\(p_{m,k}=q_{m,k}\)，暂不混合 \(p_{\exp}\)。
- **论文逐节审计**：见 [§三附](#三附论文逐节实现审计代码核实)（2026-06-12 对照 `prior_enhance/` 全量核实）。
- **一个月目标**：补齐 §三 / §三附 中标记为 ❌/⚠️ 的公式与 §四 中参数定标，完成 holdout + rolling 主实验及至少 2 个数据集验证。

### 1.2 进度总览（截至 2026-06-12）

| 阶段 | 内容 | 状态 |
|------|------|------|
| 已完成 | Phase 1→3 闭环、AutoApex 接入、holdout smoke | ✅ |
| 已完成 | Prior 相似检索、prior_store、排行榜 \(\lambda_m^{\text{init}}\) | ✅ |
| 已完成 | Softmax \(q_{m,k}\propto e^{\beta\psi}\)（\(\beta=1,\rho=0\)） | ✅ |
| 第 1 周剩余 | holdout 全预算搜索、serve 参数固化 | 进行中 |
| 第 2 周 | anti-prior、多元 Gain、χ 实装 | 未开始 |
| 第 3 周 | rolling 多窗编排、\(\tau\) 链实验验证 | 未开始 |
| 第 4 周 | 参数定标、ablation、多数据集、论文产物 | 未开始 |

### 1.3 AutoApex 接入说明

| 项 | 配置 |
|----|------|
| serve 目录 | `/mnt/raid/forecast/Valrant/GZSB/prob/autoapex/serve` |
| 默认 base URL | `http://127.0.0.1:8000/api`（`AUTOAPEX_BASE_URL`） |
| 客户端 | `validation/autoapex_client.py` |
| 评估入口 | `validation/evaluator.py` |
| 训练超时 | `AUTOAPEX_TRAIN_TIMEOUT`（默认 7200s） |

运行前须在 mutiagent 环境启动 serve；`--with-search` 对每个 recipe 调用真实 `/api/train`。

### 1.4 Prior 初始化与 LLM 信任（已接入 · 2026-06-12）

论文 §Prior Reliability 初始化中的两条机制均已实现并接入主链路：

| 机制 | 论文公式 | 实现 | 说明 |
|------|----------|------|------|
| 相似 prior 检索 | \(\tau_{j,m}=\frac{\sum s\tau_i}{\sum s+\epsilon}\) | `prior/similarity.py` | cond/target/action 集合重叠 + BGE 语义（默认 15% 权重） |
| 历史 prior 库 | \(\mathcal{P}_{j,m}\) | `observation/prior_store.py` | JSONL 默认 `outputs/prior_store.jsonl`；observation 后 append |
| 初始局部可靠性 | \(\tau_{j,m}^0=\clip(a_j[\lambda_m b+(1-\lambda_m)\tau_{j,m}],...)\) | `prior/compiler.py::init_tau_loc` | 搜索前从 prior_store 检索；`search/loop.py` 调用 |
| LLM 冷启动信任 | \(\lambda_m^{\text{init}}=\frac{score_m-score_{\min}}{score_{\max}-score_{\min}+\epsilon}\) | `prior/llm_trust.py` + `prior/data/llm_leaderboard.csv` |

**环境变量**：

| 变量 | 作用 | 默认 |
|------|------|------|
| `prior/data/llm_leaderboard.csv` | 205 模型智能指数表 | \(score_{\max}=61\), \(score_{\min}=5\) |
| `PRIOR_ENHANCE_LLM_LEADERBOARD_SCORE` | 手动指定 \(score_m\)（覆盖 CSV） | — |
| `PRIOR_ENHANCE_LLM_LEADERBOARD_MAX` / `_MIN` | 归一化上下界 | 61 / 5 |
| `PRIOR_ENHANCE_LAMBDA_M` | 显式覆盖 \(\lambda_m\) | — |
| `PRIOR_ENHANCE_PRIOR_SIM_MIN` | 相似 prior 阈值 \(s_{\min}\) | 0.25 |
| `SearchConfig.prior_semantic_weight` | BGE 在 \(s(z,z')\) 中的权重 | 0.15 |
| `SearchConfig.beta_k` | softmax 温度 \(\beta_k\) | **1.0** |
| `SearchConfig.explore_ratio` | 探索比例 \(\rho_k\) | **0.0**（当前 \(p=q\)） |
| `SearchConfig.proposal_pool_size` | 候选池规模 | 48 |

**排行榜查表规则**：配置 `llm.model` 与 CSV 中 `model_name` 做严格归一化匹配；**未命中**时使用 \(score_m=score_{\min}=5\) → \(\lambda_m^{\text{init}}=0\)（低信任）。例：`deepseek-r1-distill-qwen-1.5b` 不在表中 → \(\lambda=0\)。

**首窗行为**：prior_store 为空时 \(\tau_{j,m}=0.5\)；第 2 窗起可检索历史相似 prior。

### 1.5 Softmax 提议分布（已接入 · 2026-06-12）

论文：\(q_{m,k}(r)\propto \exp\{\beta_k \psi_{m,k}(r)\}\)，\(p_{m,k}=(1-\rho_k)q+\rho_k p_{\exp}\)。

| 项 | 实现 | 说明 |
|----|------|------|
| \(\psi_{m,k}(r)\) | ✅ | `search/loop.py::_psi_score` |
| \(q_{m,k}\) softmax | ✅ | `search/proposal.py`；候选池 + 数值稳定 softmax |
| \(\beta_k\) | ✅ | 默认 **1.0**（`SearchConfig.beta_k`） |
| \(\rho_k\) | ⚠️ | 默认 **0.0** → \(p_{m,k}=q_{m,k}\)，**暂不**混合 \(p_{\exp}\) |
| anti-prior / 三类 \(p_{\exp}\) | ❌ | 待 W2 |

**尚未完成（仍属 W3）**：`pipeline.run_rolling()`；\(s_{\min}\)、语义权重实验定标。

## 二、论文未实现条文（原文摘录）

以下为论文正文中**尚未在 prior_enhance 完整实现**的表述，按原文摘录。

### 2.1 上下文状态与风险

> 在当前实现阶段，风险等级 \(\xi_m\) 及其诱导的风险惩罚项暂不显式建模。后续若任务需要概率预测、峰值风险控制或高风险场景审计，可进一步将 \(\xi_m\) 扩展回上下文状态向量中。

> 若业务涉及概率分布，则可将真实泛化风险显式定义为：\(R_m^{\text{obs}}(r) = \zeta_1 \CoverageGap_m(r) + \zeta_2 \TailRisk_m(r) + \zeta_3 \PeakUnderestimation_m(r)\)。

> 如果任务指定要求输出概率预测，则约束集额外加入置信覆盖率约束：\(\Coverage_m(r) \ge 1 - \alpha - \epsilon\)。

> 概率校准模块在当前阶段暂不启用。

> **说明**：论文「Prior 初始化与 LLM 信任」已实现，见 [§1.4](#14-prior-初始化与-llm-信任已接入--2026-06-12)。Softmax \(q_{m,k}\) 已实现（\(\beta=1,\rho=0\)），见 [§1.5](#15-softmax-提议分布已接入--2026-06-12)。

### 2.2 Prior-Guided 采样分布（部分未实现）

> 基于该评分，定义先验指导推荐分布：\[q_{m,k}(r) \propto \exp \{ \beta_k \psi_{m,k}(r) \}, \quad r \in \mathcal{F}_m.\] **（✅ 已实现，`search/proposal.py`，\(\beta_k=1\)）**

> 为了从根本上消除大模型先验陷入自我强化的正反馈死循环，必须保留统计学意义上的探索，引入探索基准分布 \(p_{\text{exp}}(r)\)。……最终的混合方案采样分布定义为：\[p_{m,k}(r) = (1 - \rho_k) q_{m,k}(r) + \rho_k p_{\text{exp}}(r), \quad r \in \mathcal{F}_m.\]

> 探索分布 \(p_{\text{exp}}\) 必须保证覆盖以下三类独特的方案空间：(1) 随机生成的方案（Random Recipes）；(2) 低先验符合度的方案（Low-prior Recipes）；(3) **反先验但满足可行性约束的方案**（Anti-prior but Feasible Recipes）。

> 所谓 Anti-prior recipe 指的是显式违反了部分 LLM 先验建议、但在计算和业务上依然合法的方案。

### 2.3 Validation Gain 与混杂控制

> 在当前已评估集合 \(\mathcal{E}_{m,k}\) 上拟合一个受到严格控制的因果多元线性回归：\[J_{m,k}^{\text{val}}(r) = \alpha_{j,m,k}^{\text{val}} + \beta_{j,m,k}^{\text{val}} \phi_{j,m}(r) + (\gamma_{j,m,k}^{\text{val}})^\top \bm{x}_m(r) + \varepsilon_{j,m,k}^{\text{val}}(r)\]

> 各个乘性因子的考核标准为：样本充分性（\(q_{\text{sample}}\)）；对照组平衡性（\(q_{\text{balance}}\)）：如果评估的全是单向顺从先验的方案，没有任何反先验方案做对比，则此因子直接归零；拟合质量（\(q_{\text{quality}}\)）；跨切片稳定性（\(q_{\text{stability}}\)）。

> 在窗口 \(m\) 的局部运行期间……\[ \mathcal{H}_m \longrightarrow \{(\mathcal{D}_{m,b}^{\text{train}}, \; \mathcal{D}_{m,b}^{\text{val}})\}_{b=1}^{B_{\text{val}}}. \]

### 2.4 Observation 与持久记忆

> 作为因果回归的备选方案，也可以为每条先验 \(j\) 显式构造紧密配对的因果方案对集合：\[P_{j,m} = \{(r_{j,m}^+, \; r_{j,m}^-)\}.\]……Observation Gain 可以通过对配对差值求期望来稳健获取。

> 非平稳环境下的动态漂移遗忘率 \(\omega_m\) 紧密地受控于当前窗口的上下文状态向量 \(S_m\)：\[\omega_m = \omega_{\max} \cdot \sigma \left( \alpha_0 - \alpha_n n_m + \alpha_\nu \nu_m + \alpha_d d_m + \alpha_\xi \xi_m \right).\]

### 2.5 实验与数据

> （注：在实际实验中，可以先拿电价和金融数据、重过载场景做验证。）

> 这里的 episode 可理解为「历史序列 + 外生变量 + 事件语义 + 上下文」的综合物理组合。

> 工程实现中，若最大语义相似度 \(s_m^{\max} < 0.5\)，则认为当前事件具有较高新颖性。

---

## 三、论文公式实现对照表

| 编号 | 论文公式 / 对象 | 含义 | 实现状态 | 代码位置 |
|------|-----------------|------|----------|----------|
| — | \(\mathcal{D}_{\text{full}}=\{(y_\tau,x_\tau)\}\) | 全量数据 | ✅ | `data/preprocessor.py` |
| — | \(\mathcal{H}_m,\ \mathcal{O}_m\) | 历史 / 盲盒未来 | ✅ | `data/window.py` |
| — | \(T_m=(\mathcal{H}_m,X_m^+,E_m,K_m,H,B,l)\) | 预测任务 | ✅ | `schemas.ForecastTask` |
| — | \(\mathcal{D}_m=\Retrieve(T_m,\mathcal{D})\) | episode 检索 | ✅ | `data/episode_store.py` |
| (1) | \(S_m=(n_m,\nu_m,d_m)\) | 上下文状态 | ✅ | `context/state.py` |
| (2) | \(n_m=\clip(N_m^{\mathrm{eff}}/N_m^{\mathrm{ref}},0,1)\) | 有效样本量 | ✅ | `context/state.py` |
| (3) | \(\nu_m=1-s_m^{\max}\) | 事件新颖性 | ✅ | `context/event_novelty.py` |
| (4) | \(d_m=d_m^\mu+d_m^\sigma\) | 分布漂移 | ✅ | `context/state.py` |
| (5) | \(\xi_m\) | 运行风险 | ❌ | — |
| (6) | \(r=(W,F,M,\theta,V,Q)\) | Forecasting Recipe | ✅ | `recipe/schema.py` |
| (7) | \(\mathcal{F}_m=\{r:\ C_m\le B,\ \Leakage=0,\ \Valid=1\}\) | 可行方案集 | ⚠️ | schema 有；Coverage 硬约束未接 |
| (8) | \(z_{j,m}=(\text{cond},\text{target},\text{action},b)\) | 结构化 prior | ✅ | `prior/generator.py` |
| (9) | \(b_{j,m}=\lambda b^{LLM}+(1-\lambda)b^{\text{evidence}}\) | 置信度校准 | ✅ | `prior/confidence.py` |
| (10) | \(b^{\text{evidence}}=0.4n_m+0.3(1-\nu_m)+0.3\min(|\mathcal{D}_m|/K,1)\) | 证据置信度 | ✅ | `prior/confidence.py` |
| (11) | \(\tau_{j,m}\) 相似 prior 加权初始化 | 持久可靠性初值 | ✅ | `prior/similarity.py` + `observation/prior_store.py` |
| (12) | \(\lambda_m^{\text{init}}\) 排行榜映射 | LLM 初始 trust | ✅ | `llm_leaderboard.csv`；未命中 → \(\lambda=0\) |
| (13) | \(\tau_{j,m}^0=\clip(a_j[\lambda_m b+(1-\lambda_m)\tau_{j,m}],\epsilon_\tau,1-\epsilon_\tau)\) | 初始可靠性 | ✅ | `init_tau_loc` + prior_store |
| (14) | \(\phi_{j,m}(r)\) | prior 满足度 | ✅ | `prior/compiler.py` |
| (15) | \(\psi_{m,k}(r)=\sum_j \tau_{j,m,k}^{\text{loc}}\phi_{j,m}(r)\) | 综合 prior 分 | ✅ 用于采样 | `search/loop.py` + `proposal.py` |
| (16) | \(q_{m,k}(r)\propto\exp\{\beta_k\psi_{m,k}\}\) | softmax 提议 | ✅ | `search/proposal.py`；\(\beta_k=1\) |
| (17) | \(p_{m,k}=(1-\rho_k)q+\rho_k p_{\text{exp}}\) | 混合采样 | ⚠️ | \(\rho_k=0\) 时 \(p=q\)；\(p_{\exp}\)/anti-prior 未做 |
| (18) | Anti-prior feasible recipes | 反先验对照 | ❌ | — |
| (19) | \(J^{\text{val}}=L+\lambda_C C+\mu_m R^{\text{val}}\) | 验证目标 | ✅ | `validation/objective.py` |
| (20) | \(\mu_m=\mu(S_m)\) 含 \(\xi_m\) | 风险惩罚强度 | ⚠️ | \(0.2+0.4d+0.2\nu\)，无 \(\xi_m\) |
| (21) | \(J^{\text{val}}=\alpha+\beta\phi+\gamma^\top x+\varepsilon\) | 多元 Gain | ❌ | 当前一元 OLS |
| (22) | \(G^{\text{val}}=-\beta^{\text{val}}\) | Validation Gain | ✅ | `validation/gain.py` |
| (23) | \(\Delta=\clip(G/(\SE+\epsilon_G),-\Delta_{\max},\Delta_{\max})\) | 标准化增益 | ✅ | `validation/gain.py` |
| (24) | \(\chi=q_{\text{sample}}q_{\text{balance}}q_{\text{quality}}q_{\text{stability}}\) | 证据门控 | ⚠️ | balance/stability 占位 |
| (25) | \(l^{\text{loc}}_{k+1}=(1-\omega_{\text{val}})l^{\text{loc}}_k+\eta_{\text{val}}\chi\Delta\) | \(\tau^{\text{loc}}\) 更新 | ✅ | `validation/gain.py` |
| (26) | \(r_m^*=\arg\min(J^{\text{val}}+\kappa\SE)\) | 最终方案 | ✅ | \(\kappa=0\) |
| (27) | \(f_m^*=\Train(r_m^*,\mathcal{H}_m)\) | 全量重训 | ✅ | `evaluator.py` + serve |
| (28) | \(\hat{y}_{\mathcal{O}_m}=f_m^*(X_m^+)\) | 正式外推 | ⚠️ | `/api/inference/v1` 已接；\(X_m^+\) 外生变量传递不完整 |
| (29) | \(\mathcal{E}_m^{\text{retain}}=\mathcal{E}^{\text{top}}\cup\mathcal{E}^{\text{explore}}\) | 保留审计集 | ✅ | `search/loop.py` |
| (30) | \(G^{\text{obs}}=-\beta^{\text{obs}}\) | Observation Gain | ✅ | `observation/feedback.py` |
| (31) | \(P_{j,m}=\{(r^+,r^-)\}\) 配对断层 | 备选 Gain | ❌ | — |
| (32) | \(\omega_m=\omega_{\max}\sigma(\alpha_0-\alpha_n n+\alpha_\nu\nu+\alpha_d d+\alpha_\xi\xi)\) | 动态遗忘 | ⚠️ | 简化 sigmoid，无 \(\xi\) |
| (33) | \(l_{j,m+1}=(1-\omega_m)l_{j,m}+\eta_{\text{obs}}\chi\Delta^{\text{obs}}\) | 持久 \(\tau\) | ✅ | `observation/memory.py` |
| (34) | \(\lambda_{m+1}=(1-\eta_\lambda)\lambda_m+\eta_\lambda\frac{\sum\chi\tau}{\sum\chi+\epsilon}\) | 全局 \(\lambda\) | ✅ | `observation/memory.py` |

**图例**：✅ 已实现；⚠️ 简化/部分实现；❌ 未实现。

---

## 三附、论文逐节实现审计（代码核实）

> **核实日期**：2026-06-12  
> **核实范围**：`prior_enhance/` 全模块，对照论文正文逐节比对  
> **判定标准**：公式/机制在代码中有对应实现且语义一致 → ✅；有骨架但简化或未接主链路 → ⚠️；论文明确要求但代码缺失 → ❌

### 3附.0 审计总览：核心链条 \(Z_m \to \phi \to \psi \to p \to G \to \tau\)

| 环节 | 论文要求 | 工程状态 |
|------|----------|----------|
| \(Z_m\) 结构化 prior | LLM 生成 cond/target/action/\(b\) | ✅ |
| \(\phi_{j,m}(r)\) | prior 满足度编译 | ✅（有缺口，见 §3附.5） |
| \(\psi_{m,k}(r)\) | \(\sum_j \tau^{\text{loc}}\phi\) | ✅ 驱动 \(q_{m,k}\) 采样 |
| \(p_{m,k}(r)\) | \((1-\rho)q+\rho p_{\exp}\) | ⚠️ | \(\rho=0\) 时 \(p=q\) |
| \(G_{j,m}\) | validation / observation gain | ⚠️ 一元 OLS，非多元 |
| \(\tau_{j,m}\) | 证据校准持久可靠性 | ⚠️ | 初始化 ✅（§1.4）；observation 更新受 \(\chi\) 门控限制 |

**结论**：Prior 初始化（11–13）、softmax \(q\)（16，\(\beta=1,\rho=0\)）已接入。当前最大缺口：**anti-prior / \(p_{\exp}\)**（17–18）、多元 Gain（21）、rolling 编排。

---

### 3附.1 核心立场（LLM generates testable priors, not forecasts）

| 论文原则 | 状态 | 代码依据 |
|----------|------|----------|
| LLM 不直接预测时序 | ✅ | `prior/generator.py` prompt 禁止数值预测 |
| LLM 不直接决定最终 recipe | ✅ | `search/loop.py`：`r^*=\arg\min J^{\text{val}}`，无 LLM 项 |
| 只生成结构化 prior | ✅ | `schemas.StructuredPrior` |
| prior → recipe activation | ✅ | `prior/compiler.py` → \(\phi\) |
| prior reliability 控制搜索分布 | ⚠️ | \(\tau^{\text{loc}}\) 有，采样走 `prior_guided_recipe`+随机，**不是** \(p_{m,k}\) |
| 预测反馈估计 gain | ✅ | `validation/gain.py`, `observation/feedback.py` |
| reliability 动态校准 | ⚠️ | 窗内 \(\tau^{\text{loc}}\) ✅；跨窗 \(\tau\) 公式 ✅，\(\chi\) 门控常拒更新 |
| 最终 recipe 仅由验证/风险/成本决定 | ✅ | `validation/objective.py` + `search/loop.py` |

---

### 3附.2 数据组织与滚动预测设定

| 论文对象 | 状态 | 说明 |
|----------|------|------|
| \(\mathcal{D}_{\text{full}}\) | ✅ | `data/preprocessor.py` |
| \(\mathcal{H}_m\)、\(\mathcal{O}_m\) 严格分离 | ✅ | `data/window.py`；搜索只用 `history_df` |
| 滚动原点 / walk-forward | ⚠️ | `build_rolling_windows()` 存在；`pipeline.py` 只跑 `windows[0]`，无 `run_rolling()` |
| 搜索/训练/LLM 阶段遮蔽 \(\mathcal{O}_m\) | ✅ | `run_prior_guided_search` 不读 O_m |
| O_m 揭晓后才 observation | ✅ | `observation/feedback.py` |

---

### 3附.3 预测任务 \(T_m\) 与上下文 \(S_m\)

#### 3附.3.1 七元组 \(T_m=(\mathcal{H}_m,X_m^+,E_m,K_m,H,B,l)\)

| 分量 | 状态 | 说明 |
|------|------|------|
| \(\mathcal{H}_m\) | ✅ | 落盘 `H_m` CSV |
| \(X_m^+\) 已知未来外生变量 | ⚠️ | `ForecastTask.known_covariates` 有字段，默认常为空，未传入 AutoApex inference |
| \(E_m\) 事件文本 | ⚠️ | 无外部事件时用 `build_calendar_event_context` 合成日历事件 |
| \(K_m\) 领域知识 | ✅ | `config.domain_knowledge` → prompt |
| \(H, B, l\) | ✅ | `WindowConfig` / `SearchConfig` |
| \(\mathcal{D}_m=\Retrieve(T_m,\mathcal{D})\) | ✅ | `data/episode_store.py` |

#### 3附.3.2 \(S_m=(n_m,\nu_m,d_m)\)

| 公式/对象 | 状态 | 代码 |
|-----------|------|------|
| \(n_m=\clip(N^{\text{eff}}/N^{\text{ref}},0,1)\) | ✅ | `context/state.py::_compute_n_m` |
| \(\nu_m=1-s_m^{\max}\) | ✅ | `context/event_novelty.py` + BGE 编码器 |
| \(d_m=d_m^\mu+d_m^\sigma\) | ✅ | `context/state.py::_compute_d_m` |
| \(\xi_m\) 运行风险 | ❌ | 论文写明当前阶段不做；代码亦未做 |
| 离散 cond 标签 | ✅ | `context/labels.py`（0.33/0.66 分档；\(\nu_m\ge0.5\)→novel） |
| \(s_m^{\max}<0.5\) 高新颖性规则 | ⚠️ | 标签逻辑一致，未在 BGE 分布上验证 |

---

### 3附.4 Forecasting Recipe \(r=(W,F,M,\theta,V,Q)\)

| 论文要求 | 状态 | 说明 |
|----------|------|------|
| 六元组 schema | ✅ | `recipe/schema.py` |
| AutoApex 搜索空间映射 | ✅ | `recipe/autoapex_adapter.py`, `prior/search_space.py` |
| \(V\) 验证划分策略 | ⚠️ | 沿用 AutoApex 默认、不作搜索维（与论文一致），未作为 recipe 组件显式建模 |
| \(Q\) 概率校准 | ❌ | 论文写明暂不启用；`Q` 仅 `enable_ensemble` |
| \(\mathcal{F}_m: C_m\le B,\ \Leakage=0,\ \Valid=1\) | ⚠️ | 预算靠 `max_trials`；`prior/validator.py` 查泄露关键词；无 \(\Coverage_m\) 硬约束 |
| tsfresh / 特征工程空间 | ✅ | catalog + adapter |

---

### 3附.5 结构化 LLM Prior \(Z_m\)

| 论文组件 | 状态 | 说明 |
|----------|------|------|
| \(z=(\text{cond},\text{target},\text{action},b)\) | ✅ | `schemas.StructuredPrior` |
| cond 由 \(S_m\) 标签组合 | ⚠️ | prompt 给出默认 cond；LLM 可输出与 \(S_m\) 不一致的 cond，validator 不强制匹配 |
| target 五簇 \(\mathcal{W},\mathcal{F},\mathcal{M},\Theta,\mathcal{E}\) | ⚠️ | schema 全有；\(\phi\) **不算** `hyperparameter_policy` 重叠 |
| action 负向规避 | ✅ | `AVOID_HANDLERS` 映射 avoid_* |
| \(b=\lambda b^{LLM}+(1-\lambda)b^{\text{evidence}}\) | ✅ | `prior/confidence.py` |
| \(b^{\text{evidence}}=0.4n+0.3(1-\nu)+0.3\min(\|\mathcal{D}_m\|/K,1)\) | ✅ | \(K=\) `episode_ref_k`（默认 10） |
| LLM 不自由生成 recipe | ✅ | 只生成搜索空间内簇 |

**\(\phi\) 编译缺口**（影响 Gain 因果解释）：

- `prior/compiler.py` 中 `FEATURE_ALIAS` 将 `calendar`/`lag` 映射为 `none`，LLM 推荐这些特征时 \(\phi\) 系统性偏低。
- `prior/recipe_sampler.py` 采样时亦未使用 `hyperparameter_policy`。

---

### 3附.6 Prior Reliability 初始化

| 论文公式 | 状态 | 说明 |
|----------|------|------|
| \(\tau_{j,m}\) 历史相似 prior 加权 | ✅ | `prior/similarity.py` 离散重叠 + BGE 语义（15% 权重） |
| 无相似 prior 时 \(\tau=0.5\) | ✅ | `s<s_{\min}` 时无信息先验；**首窗** prior_store 为空亦为 0.5 |
| \(\lambda_m^{\text{init}}\) 公开排行榜映射 | ✅ | `prior/llm_trust.py`；`PRIOR_ENHANCE_LLM_LEADERBOARD_SCORE` 或模型表 |
| \(a_{j,m}=a_j(S_m)\) cond 匹配度 | ✅ | `prior/compiler.py::applicability` |
| \(\tau_{j,m}^0=\clip(a_j[\lambda_m b+(1-\lambda_m)\tau_{j,m}],\epsilon_\tau,1-\epsilon_\tau)\) | ✅ | `init_tau_loc` 已接 prior_store 检索 |
| \(\tau_{j,m,0}^{\text{loc}}=\tau_{j,m}^0\) | ✅ | `init_tau_loc` |

---

### 3附.7 Prior-Guided Recipe Search（论文与工程差距最大）

| 论文公式 | 状态 | 说明 |
|----------|------|------|
| \(\psi_{m,k}(r)=\sum_j \tau^{\text{loc}}_j \phi_j(r)\) | ✅ | 驱动 softmax 采样 |
| \(q_{m,k}(r)\propto\exp\{\beta_k\psi\}\) | ✅ | `search/proposal.py`；\(\beta_k=1\) |
| \(p_{m,k}=(1-\rho_k)q+\rho_k p_{\exp}\) | ⚠️ | \(\rho_k=0\) → \(p=q\) |
| \(p_{\exp}\) 三类：random / low-prior / **anti-prior** | ❌ | 待 W2 |
| Anti-prior feasible recipes | ❌ | 待 W2 |
| \(\mathcal{A}_{m,k}\sim p_{m,k}\) | ✅ | softmax 无放回抽样 |
| \(\mathcal{E}_m\) 累积评估集 | ✅ | `search/loop.py` |

当前采样（`search/loop.py::_sample_batch`）：

- **guide**：`prior_guided_recipe` — 按 confidence 加权选 prior，从 target 簇偏置采样；
- **explore**：`sample_random_recipe` — 纯随机。

与论文 Boltzmann 提议 + 三类对照探索**不等价**。

---

### 3附.8 窗口内 Validation 与 Validation Gain

| 论文要求 | 状态 | 说明 |
|----------|------|------|
| \(\mathcal{H}_m\to\{(\mathcal{D}^{\text{train}},\mathcal{D}^{\text{val}})\}_{b=1}^{B_{\text{val}}}\) | ⚠️ | 仅单切分 `make_holdout_val_split`；`make_rolling_val_splits` 未接入主循环 |
| 时间有序 train < val ⊂ \(\mathcal{H}_m\) | ✅ | `validation/split.py` |
| \(J^{\text{val}}=L+\lambda_C C+\mu_m R^{\text{val}}\) | ✅ | `validation/objective.py` |
| \(\mu_m=\mu(S_m)\) 含 \(\xi_m\) | ⚠️ | `0.2+0.4d+0.2ν`，无 \(\xi_m\) |
| \(R^{\text{val}}\) 峰值/尾部风险 | ⚠️ | 简化 peak_under + tail MAE，非 \(\CoverageGap\)/\(\TailRisk\) |
| \(C_m(r)\) 真实算力 | ⚠️ | AutoApex 下可用 `wall_time_sec`；否则 catalog 代理常数 |
| 局部反馈**只**更新 \(\tau^{\text{loc}}\) | ✅ | 不写持久 \(\tau\) |
| 多元回归 \(J=\alpha+\beta\phi+\gamma^\top x+\varepsilon\) | ❌ | 一元 OLS；`recipe_feature_vector` **已定义未使用** |
| \(G^{\text{val}}=-\beta\), \(\Delta=\clip(G/(\SE+\epsilon_G),...)\) | ✅ | `validation/gain.py` |
| \(\chi=q_{\text{sample}}q_{\text{balance}}q_{\text{quality}}q_{\text{stability}}\) | ⚠️ | `q_balance=0.7` 占位；`require_balance=False`；`q_stability=0.8` 占位 |
| \(\chi<\chi_{\min}\) 拒绝更新 | ✅ | config `chi_min=0.25` vs gain 默认 `0.3` **不一致** |

**运行观测**（mutiagent smoke，`max-trials=1`）：`memory_tau` 全为 0.5，因样本不足 + \(\chi\) 门控失败——与论文「证据不足则拒绝更新」一致，但反馈回路 practically 未激活。

---

### 3附.9 局部 \(\tau^{\text{loc}}\) 更新

| 论文公式 | 状态 | 代码 |
|----------|------|------|
| \(l^{\text{loc}}=\logit(\tau^{\text{loc}})\) | ✅ | `update_tau_logit` |
| \(l_{k+1}=(1-\omega_{\text{val}})l_k+\eta_{\text{val}}\chi\Delta\) | ✅ | 默认 \(\omega_{\text{val}}=0.2,\ \eta_{\text{val}}=0.15\) |
| \(\tau^{\text{loc}}_{k+1}=\sigma(l_{k+1})\) | ✅ | |
| 不写回持久 \(\tau_{j,m}\) | ✅ | |

---

### 3附.10 最终 Recipe 选择

| 论文要求 | 状态 | 说明 |
|----------|------|------|
| \(r_m^*=\arg\min_{r\in\mathcal{E}_m\cap\mathcal{F}_m}(J^{\text{val}}+\kappa SE)\) | ✅ | \(\kappa=0\)；从已评估集选 |
| 删除 LLM 评分项 | ✅ | 无 \(-\eta_s\text{LLM}(r)\) |
| \(f_m^*=\Train(r_m^*,\mathcal{H}_m)\) | ✅ | AutoApex `/api/train` on full H_m |
| \(\hat{y}_{\mathcal{O}_m}=f_m^*(X_m^+)\) | ⚠️ | `/api/inference/v1` 已接；\(X_m^+\) 传递不完整 |

---

### 3附.11 Observation Feedback 与 Observation Gain

| 论文要求 | 状态 | 说明 |
|----------|------|------|
| O_m 揭晓后算 observation | ✅ | holdout 末尾 H 点 |
| \(\mathcal{E}_m^{\text{retain}}=\mathcal{E}^{\text{top}}\cup\mathcal{E}^{\text{explore}}\) | ✅ | top by J_val，explore by 低 \(\phi\) |
| \(J^{\text{obs}}=L^{\text{obs}}+\lambda_C C+\mu_m R^{\text{obs}}\) | ✅ | 形式同 J_val |
| \(R^{\text{obs}}=\zeta_1\CoverageGap+\zeta_2\TailRisk+\zeta_3\PeakUndest\) | ❌ | 点预测 MVP，用简化 \(R^{\text{val}}\) |
| 多元 Observation 回归 | ❌ | 同 validation，一元 OLS |
| 配对断层 \(P_{j,m}=\{(r^+,r^-)\}\) 备选 Gain | ❌ | |
| 样本不足/失衡则拒绝强更新 | ⚠️ | \(\chi\) 门控有，balance 未 strict |

---

### 3附.12 持久 Memory 与全局 \(\lambda\)

| 论文公式 | 状态 | 说明 |
|----------|------|------|
| \(\Delta^{\text{obs}}=\clip(G^{\text{obs}}/(\SE+\epsilon_G),...)\) | ✅ | 同 validation |
| \(\omega_m=\omega_{\max}\sigma(\alpha_0-\alpha_n n+\alpha_\nu\nu+\alpha_d d+\alpha_\xi\xi)\) | ⚠️ | `omega_forget_rate` 简化 sigmoid，无 \(\alpha_0,\ldots,\xi\) |
| \(l_{j,m+1}=(1-\omega_m)l_{j,m}+\eta_{\text{obs}}\chi\Delta^{\text{obs}}\) | ✅ | `observation/memory.py` |
| \(\tau_{j,m+1}=\sigma(l_{j,m+1})\) | ✅ | |
| \(\lambda_{m+1}=(1-\eta_\lambda)\lambda_m+\eta_\lambda\frac{\sum\chi\tau}{\sum\chi+\epsilon}\) | ✅ | 证据不足时保持不变 |
| prior_store / 跨窗 JSONL | ✅ | `observation/prior_store.py`；默认 `outputs/prior_store.jsonl`；observation 后 append |

---

### 3附.13 实验与数据（论文脚注）

| 论文要求 | 状态 |
|----------|------|
| 电价 / 金融 / 重过载验证 | ❌ 目前仅代理购电 CSV + holdout smoke |
| 多窗口 rolling 主实验 | ❌ `pipeline` 未编排多窗链 |
| 参数标定（\(\beta_k,\rho_k,\chi_{\min}\) 等） | ❌ 大量工程默认值，见 §4.1 |

---

### 3附.14 与论文差距最大的 8 项（实施优先级）

| 优先级 | 缺口 | 对应公式/章节 | 计划周次 |
|--------|------|---------------|----------|
| P0 | \(p_{\exp}\) + anti-prior 对照（\(\rho_k>0\)） | (17)(18) §3附.7 | W2 |
| P1 | 多元 Gain 回归 + \(x_m(r)\) 混杂控制 | (21) §3附.8 | W2 |
| P1 | \(\chi\) 的 \(q_{\text{balance}}\)/\(q_{\text{stability}}\) 实装 | (24) §3附.8 | W2 |
| P1 | rolling 多窗编排 + \(\tau\) 链实验 | §3附.2 §3附.13 | W3 |
| P2 | \(\phi\) 完整性（hyperparameter_policy、calendar/lag） | §3附.5 | W1–W2 |
| P2 | \(X_m^+\)、真实 \(E_m\)、概率 \(Q\)/Coverage | §3附.3 §3附.4 | W4 可选 |

---

### 3附.15 已复现且与论文一致（可写入论文「工程复现」）

- 信息边界：\(\mathcal{H}_m\) / \(\mathcal{O}_m\) 分离，搜索阶段不看未来。
- \(S_m=(n_m,\nu_m,d_m)\) 三因子及 cond 离散化。
- LLM → 结构化 \(Z_m\) → Valid 过滤 → \(b\) 证据校准。
- \(\phi\)/\(\psi\) 计算、\(\tau^{\text{loc}}\) logit 更新、局部/持久更新职责分离。
- \(r^*\) 纯验证目标选取（无 LLM 偏置）。
- \(\mathcal{E}^{\text{retain}}=\text{top}\cup\text{explore}\)、\(G^{\text{obs}}\)、\(\lambda\) 更新公式。
- \(\tau_{j,m}\) 相似 prior 检索、\(\lambda_m^{\text{init}}\) 排行榜（205 模型 CSV）。
- \(q_{m,k}\propto e^{\beta\psi}\) softmax 采样（\(\beta=1,\rho=0\)）。
- AutoApex 真实 \(\Train(r^*,\mathcal{H}_m)\) + O_m 推理（2026-06-12 mutiagent 验证：`J_val=0.354`，`J_obs=0.351`）。

---

## 四、核心困难分析

### 4.1 困难一：论文参数缺乏设定依据（P0）

论文公式含大量自由参数，正文给出形式但未给出标定方法。当前代码采用工程默认值，与论文符号的对应关系及**设定状态**如下。

#### 4.1.1 上下文状态 \(S_m\)

| 参数 | 论文/公式 | 代码现状 | 定标状态 |
|------|-----------|----------|----------|
| \(N_m^{\mathrm{ref}}\) | 节假日 \(Y\times H\) | 场景参考计数 | 未按数据集验证 |
| \(\delta_\mu,\ \delta_\sigma\) | 漂移归一化阈值，建议 1.0 | `state.py` 默认 1.0 | 无敏感性实验 |
| \(n_m\) 分档 | low/medium/high | 0.33/0.66 硬阈值 | 无分位数标定 |
| \(\nu_m\) 新颖阈值 | \(s_m^{\max}<0.5\) | 0.5 用于 cond 标签 | BGE 分布未验证 |
| \(d_m\) 上界 | clip 至 \([0,1]\) | \(\mu+\sigma\) 可超 1 | 聚合方式未定 |

#### 4.1.2 置信度与记忆

| 参数 | 论文/公式 | 代码现状 | 定标状态 |
|------|-----------|----------|----------|
| \(0.4,\ 0.3,\ 0.3\) | \(b^{\text{evidence}}\) 权重 | `confidence.py` | 无 ablation |
| \(K\) | \(\|\mathcal{D}_m\|\) 归一化 | 10 | 与 retrieve top-k 未统一论证 |
| \(\lambda\) | \(b\) 融合（`PriorConfig.confidence_lambda`） | 0.5 | 与 \(\lambda_m\) 符号不同 |
| \(\epsilon_\tau\) | logit 边界 | \(10^{-3}\) | 未验证 |
| \(s_{\min}\) | 相似 prior 阈值 | 0.25（`prior_similarity_min`） | 无敏感性实验 |
| \(\lambda_m^{\text{init}}\) | 排行榜映射 | `prior/data/llm_leaderboard.csv` | \(score_{\max}=61,min=5\)；未命中→\(\lambda=0\) |
| `prior_semantic_weight` | BGE 在 \(s(z,z')\) 中权重 | 0.15 | 无 ablation |

#### 4.1.3 搜索与采样

| 参数 | 论文/公式 | 代码现状 | 定标状态 |
|------|-----------|----------|----------|
| \(\beta_k\) | softmax 温度 | **1.0** | W4 可扫描 |
| \(\rho_k\) | 探索比例 | **0.0** | 待 anti-prior 后启用 |
| `proposal_pool_size` | 候选池 | 48 | 可调 |
| \(B\) | 搜索预算 | 12 / 50 不一致 | 真实 serve 下未测 |
| \(B_{\text{val}}\) | 多 val 切分 | 单切片 | 未实现 |
| val_horizon | 内部验证长度 | 48 | 与 \(H=72\) 比例未论证 |

#### 4.1.4 Gain 与门控

| 参数 | 论文/公式 | 代码现状 | 定标状态 |
|------|-----------|----------|----------|
| \(\chi_{\min}\) | 证据门槛 | config 0.25 / gain 0.3 | **两处不一致** |
| \(\Delta_{\max}\) | 增益 clip | 2.0 | 无依据 |
| \(\epsilon_G\) | SE 平滑 | 0.1 | 无依据 |
| \(q_{\text{balance}}\) | 反先验对照 | 0.7 占位 | strict 模式未启用 |
| \(q_{\text{stability}}\) | 跨切片稳定 | 0.8 占位 | 多切片未实现 |
| min_samples | Gain 最少样本 | 4 | 真实训练下可能不足 |

#### 4.1.5 更新动力学

| 参数 | 论文/公式 | 代码现状 | 定标状态 |
|------|-----------|----------|----------|
| \(\omega_{\text{val}},\ \eta_{\text{val}}\) | \(\tau^{\text{loc}}\) 更新 | 0.2 / 0.15 | 无敏感性分析 |
| \(\omega_{\max},\ \alpha_0,\ \alpha_n,\ \alpha_\nu,\ \alpha_d,\ \alpha_\xi\) | \(\omega_m\) | 简化版 | 与论文系数不一致 |
| \(\eta_{\text{obs}},\ \eta_\lambda\) | 持久更新 | 0.1 / 0.05 | holdout 更新幅度约 0.005 |
| \(\lambda_C\) | 成本惩罚 | 0.1 | 相对 MAPE 量级未标定 |
| \(\kappa\) | \(r^*\) 方差惩罚 | 0 | 未探索 |
| \(\zeta_1,\ \zeta_2,\ \zeta_3\) | \(R^{\text{obs}}\) 分项 | 未实现 | — |
| \(\alpha\) | Coverage 约束 | 未实现 | — |

**定标方法（W4）**：单因素扫描 → 正交网格 → 留出窗验证 → 固化 `CALIBRATION.md`。

---

### 4.2 困难二：反先验对照与 \(p_{\text{exp}}\)（P0）

论文要求 \(p_{\text{exp}}\) 覆盖 random、low-prior、anti-prior 三类。当前 `search/loop.py` 仅 `explore_ratio` 随机采样；`gain.py` 中 `q_balance=0.7` 占位，strict 模式未默认启用。无 anti-prior 时 \(G=-\beta\) 的因果解释不成立。

**实施方案**：`recipe/anti_prior_builder.py`；三路采样；逐步启用 `strict_balance`。

---

### 4.3 困难三：softmax 提议分布（P1 · 已实现基础版）

\(\psi_{m,k}\) 已通过 `search/proposal.py` 驱动 \(q_{m,k}\)（候选池 + softmax，\(\beta_k=1\)）。当前 \(\rho_k=0\)，尚未混合 \(p_{\exp}\)。

**剩余**：anti-prior 与 \(\rho_k>0\) 的三类探索；\(\beta_k\) 敏感性实验（W4）。

---

### 4.4 困难四：多元 Gain 与 \(x_m(r)\)（P1）

论文要求 \(J \sim \phi_j + x_m(r)\)；当前一元 OLS。样本量 12、控制维 5 导致欠定，须提高 trials 或岭回归。

---

### 4.5 困难五：Prior Memory 相似检索（P1 · 代码已接入，待实验验证）

论文 \(\tau_{j,m}\) 初始化依赖 \(\mathcal{P}_{j,m}\) 加权：**已实现**（`prior/similarity.py`、`prior_store`、`init_tau_loc`，见 §1.4）。剩余工作：

- **rolling 多窗编排**：`pipeline.run_rolling()` 尚未实现，单 holdout 首窗仍无历史可检索。
- **参数定标**：\(s_{\min}\)、语义权重、排行榜分映射须 E5/E6 类实验（W4）。
- **可观测性**：须 ≥2 窗或复用 `prior_store.jsonl` 才能看到 \(\tau_{j,m}\neq 0.5\)。

---

### 4.6 困难六：AutoApex 训练时延（P0，已接入，待优化）

每个 recipe 一次 `/api/train` 阻塞调用。`max_trials=12` 在真实 serve 下耗时显著。须约束搜索子空间、`time_limit` 上限及并行策略。

| 环境变量 | 作用 |
|----------|------|
| `PRIOR_ENHANCE_EVAL_BACKEND=autoapex` | 默认，真实训练 |
| `PRIOR_ENHANCE_EVAL_BACKEND=proxy` | 快速调试 |
| `PRIOR_ENHANCE_EVAL_FALLBACK_PROXY=1` | serve 失败时回退 proxy |

---

---

## 五、一个月实施计划（2026-06-12 ~ 07-11）

> **周期**：4 周（约 30 天）  
> **原则**：每周交付可运行实验 + 文档更新；优先 P0（因果对照、参数定标），P2（概率预测、配对扫描）放入第 4 周可选。

### 5.1 总览甘特图

```
         W1          W2          W3          W4
         06-12~18    06-19~25    06-26~07-02  07-03~11
         ──────────────────────────────────────────────
AutoApex ████████░░  (验收+全链)  ░           ░
反先验   ░           ████████    ░           ░
softmax  ████████    ░           ░           ░
多元Gain ░           ██████░░    ░           ░
rolling  ░           ░           ████████    ░
Memory   ████░░░░    ░           ████░░░░    ░
参数定标 ░           ░░░░        ░░░░        ████████
多数据集 ░           ░           ░           ██████░░
论文产物 ░           ░           ░░░         ████████
```

### 5.2 四周目标与交付物

| 周次 | 日期 | 核心目标 | 对应公式编号 | 交付物 |
|------|------|----------|--------------|--------|
| **W1** | 06-12 ~ 06-18 | AutoApex 稳定运行 + holdout 全链 | (27)(28) | smoke 报告、serve 参数默认值 |
| **W2** | 06-19 ~ 06-25 | anti-prior + 多元 Gain | (17)(18)(21)(24) | \(p_{\exp}\)、\(\rho>0\)、多元 OLS |
| **W3** | 06-26 ~ 07-02 | rolling 编排 + \(\tau\) 链实验 | (32)(33) | `run_rolling()`、5 窗 \(\tau\) 曲线 |
| **W4** | 07-03 ~ 07-11 | 定标 + 实验 + 收尾 | §4.1 全表 | CALIBRATION.md、ablation、多数据集 |

---

### 5.3 第 1 周（06-12 ~ 06-18）：AutoApex 稳定与 holdout 基线

**周目标**：真实 serve 下 holdout 全流程可复现，建立后续对比基线。

| 天 | 任务 | 产出 | 状态 |
|----|------|------|------|
| 06-12 | AutoApex 客户端接入；fix freq/eval_metric | `autoapex_client.py` | ✅ 完成 |
| 06-12 | Softmax 提议 `search/proposal.py`（β=1, ρ=0） | 公式 (16) | ✅ |
| 06-12 | 排行榜 CSV + `llm_leaderboard.py` | 公式 (12) | ✅ |
| 06-13 | mutiagent smoke（`max-trials=1`） | 跑通日志 | ✅ 完成 |
| 06-14 | holdout 全链（`max-trials=8`，`time_limit=300`） | `search_result.json` | 待做 |
| 06-15 | 记录单 recipe 训练耗时；确定 \(B\)、`time_limit` 可行上界 | 算力预算表 | 待做 |
| 06-16 | proxy vs autoapex 排序对比（同 4 recipe） | 对比 CSV | 待做 |
| 06-17 | \(\phi\) 审计：recipe 实际 \(W/F/M\) 与编译器一致性 | 审计脚本 | 待做 |
| 06-18 | 更新 README；W1 复盘 | W1 报告 | 待做 |

**本周验收**：
- [x] `eval_backend=autoapex` holdout smoke 成功
- [ ] `max-trials≥8` 全链无报错
- [ ] 单 recipe 平均训练耗时记录在案

---

### 5.4 第 2 周（06-19 ~ 06-25）：因果搜索与 Gain 升级

**周目标**：实现论文要求的对照采样与多元回归，使 \(G=-\beta\) 具有因果解释基础。

| 天 | 任务 | 产出 | 对应公式 |
|----|------|------|----------|
| 06-19 | 设计 `anti_prior_builder` API | 设计文档 | (18) |
| 06-20 | 实现 target 补集 + action 违反生成 | `recipe/anti_prior_builder.py` | (18) |
| 06-21 | 三路 \(p_{\exp}\) + anti-prior | 改 `search/loop.py` | (17)(18) |
| 06-22 | \(\rho_k>0\) 混合采样 | 改 `search/loop.py` | (17) |
| 06-23 | 多元 OLS：\(J \sim \phi_j + x_m(r)\) | 改 `validation/gain.py` | (21) |
| 06-24 | \(\chi\) 实装：\(q_{\text{balance}}\) strict 模式；统一 \(\chi_{\min}\) | 改 `gain.py`、`config.py` | (24) |
| 06-25 | holdout 实验：strict on/off；W2 复盘 | ablation JSON | — |

**本周验收**：
- [ ] 每轮 search 至少 1 个 recipe 满足 \(\phi<0.3\)
- [ ] `strict_balance=True` 时 \(\tau^{\text{loc}}\) 有更新记录
- [ ] 报告 \(\beta\) 与 \(\beta_{\text{uncontrolled}}\) 差异

**参数初定（W2 末）**：\(\rho=0.30\)，\(\beta_0=1.0\)，\(\chi_{\min}=0.28\)（临时值，W4 定稿）

---

### 5.5 第 3 周（06-26 ~ 07-02）：Rolling 编排与 Memory 实验

**周目标**：实现 `run_rolling()`，用多窗实验验证 prior_store 与 \(\tau_{j,m}\) 链（代码已接入，见 §1.4）。

| 天 | 任务 | 产出 | 对应公式 |
|----|------|------|----------|
| 06-26 | `pipeline.run_rolling()` 编排 | 新 API | — |
| 06-27 | ~~`observation/prior_store.py`~~ | prior 库 | (11) ✅ 已完成 |
| 06-28 | ~~`prior/similarity.py`；改 `init_tau_loc`~~ | \(\tau_{j,m}\) 检索初始化 | (11)(13) ✅ 已完成 |
| 06-29 | 5 窗 rolling（proxy，`max-trials=6`） | \(\tau,\lambda\) 轨迹图 | (33)(34) |
| 06-30 | 3 窗 rolling（autoapex，`max-trials=4`） | 真实对照表 | (27)(28) |
| 07-01 | episode / \(\nu_m\) 池随窗扩展；泄漏审计 | 审计清单 | — |
| 07-02 | W3 复盘；算力与 LLM 成本汇总 | W3 报告 | — |

**本周验收**：
- [ ] 5 窗 \(\tau_{j,m}\) 曲线呈非平凡变化（非全 0.5）
- [ ] 相似 prior 检索命中率 > 0（第 2 窗起）
- [ ] 信息边界审计通过

**rolling 配置建议**：stride = \(H=72\)，min_history = 3528，窗口数约 5–8（算力允许范围内）。

---

### 5.6 第 4 周（07-03 ~ 07-11）：参数定标、实验与论文收尾

**周目标**：§4.1 全部参数定稿，完成 ablation 与多数据集验证，产出可复现脚本。

#### 5.6.1 参数定标实验（07-03 ~ 07-07）

| 实验 | 扫描参数 | 固定条件 | 主指标 |
|------|----------|----------|--------|
| E1 | \(\rho \in \{0.15,0.25,0.35,0.45\}\) | holdout, trials=8 | \(J^{\text{obs}}\)、\(G^{\text{obs}}\) 通过率 |
| E2 | \(\chi_{\min} \in \{0.20,0.28,0.35\}\) | holdout | \(\tau\) 更新率 vs 波动 |
| E3 | min_samples \(\in \{4,6,8,12\}\) | holdout | \(\beta\) SE、符号稳定率 |
| E4 | \(n_m\) 分档：固定 0.33/0.66 vs 分位数 | holdout | cond 一致率 |
| E5 | \(b^{\text{evidence}}\) 权重单纯形网格 | holdout | \(b\) 与 \(G^{\text{obs}}\) 相关 |
| E6 | \(\eta_{\text{obs}},\ \omega_{\max}\) | 3 窗 rolling | \(\tau\) 收敛速度 |

**定标产出**：`docs/CALIBRATION.md`（每个参数：搜索范围、最优值、依据、敏感性）。

#### 5.6.2 Ablation 实验（07-06 ~ 07-07）

| 组 | 配置 | 对比目的 |
|----|------|----------|
| A0 完整 | 默认 | 基线 |
| A1 无 prior | \(\beta=0,\ \rho=1\) | prior 引导价值 |
| A2 无 explore | \(\rho=0\)，无 anti-prior | 探索必要性 |
| A3 无 memory | \(\tau\) 恒 0.5 | memory 价值 |
| A4 无 \(\tau\) 更新 | \(\eta_{\text{val}}=\eta_{\text{obs}}=0\) | 反馈回路价值 |

#### 5.6.3 多数据集与收尾（07-08 ~ 07-11）

| 天 | 任务 | 产出 |
|----|------|------|
| 07-08 | 代理购电 holdout + rolling 完整实验 | 电价报告 |
| 07-09 | 金融数据集接入（1 个） | 跨域 summary |
| 07-10 | 论文图表：\(\tau\) 轨迹、Gain 分布、\(r^*\) 命中率 | `outputs/figures/` |
| 07-11 | `reproduce.sh`；总复盘；IMPLEMENTATION_PLAN 定稿 | 可复现脚本 |

**本周验收**：
- [ ] `CALIBRATION.md` 覆盖 §4.1 全部 L1 参数
- [ ] 4 组 ablation 对比表完成
- [ ] ≥2 数据集 `summary.json`
- [ ] 一键复现 holdout 实验

---

### 5.7 一个月里程碑时间表

| 里程碑 | 日期 | 条件 |
|--------|------|------|
| **M0** | 06-12 | AutoApex smoke 跑通 | ✅ |
| **M1** | 06-18 | holdout `max-trials=8` 全链完成 |
| **M2** | 06-25 | anti-prior + \(\rho>0\) + 多元 Gain 上线 |
| **M3** | 07-02 | 5 窗 \(\tau\) 链实验完成 |
| **M4** | 07-07 | `CALIBRATION.md` 定稿 |
| **M5** | 07-11 | ablation + 多数据集 + `reproduce.sh` |

---

### 5.8 工作量与算力预估

| 项目 | 预估值 | 说明 |
|------|--------|------|
| 单 recipe 训练 | 1–3 min | `time_limit=300`，轻量模型 |
| holdout 全搜索（8 trials） | 15–30 min | W1 基线 |
| 5 窗 rolling（4 trials/窗，autoapex） | 2–4 h | W3 主实验 |
| 参数网格 E1–E6 | 1–2 天 GPU/CPU | W4，可夜间跑 |
| LLM 调用 | 每窗 1 次 | rolling 5 窗约 5 次 |

**算力策略**：
- 开发调试：`PRIOR_ENHANCE_EVAL_BACKEND=proxy`
- 正式实验：`autoapex`，`time_limit≤300`，搜索子空间限统计+ML 模型族
- 网格实验：固定 seed，结果落盘 `outputs/calibration/`

---

### 5.9 延后项（一个月外 / 可选）

以下项不在一个月必交付范围，第 4 周有余力时择项实施：

| 项 | 公式 | 说明 |
|----|------|------|
| \(\xi_m\) 运行风险 | (5)(20)(32) | 点预测 MVP 可暂缓 |
| 概率预测 + Coverage | (7) | Q 校准模块 |
| 配对断层 \(P_{j,m}\) | (31) | Gain 备选估计器 |
| memory 回灌 LLM prompt | — | 闭环演示 |
| 重过载场景 | §2.6 | 依赖数据可用性 |

---

## 六、风险登记册

| ID | 风险 | 缓解 |
|----|------|------|
| R1 | serve 训练超时 | `time_limit` 上限；搜索子空间限制 |
| R2 | strict_balance 导致 \(\tau\) 不更新 | 渐进启用；增大 batch |
| R3 | 参数未定导致结论不可复现 | W4 定标 + CALIBRATION.md |
| R4 | \(\phi\) 与真实 \(W/F/M\) 不一致 | \(\phi\) 审计脚本 |
| R5 | rolling 算力超预算 | stride=\(H\)；proxy 预跑 |

---

## 七、验收标准与里程碑

与 §5.7 里程碑表一致。月末总验收须满足：

1. holdout + rolling 主实验可在 mutiagent + serve 下复现。
2. §三 / §三附 中 P0 公式 (17)(18)(21) 由 ❌ 变为 ✅ 或 ⚠️→✅（(11)(12)(13)(16) 已于 v6–v7 完成）。
3. §4.1 L1 参数写入 `CALIBRATION.md` 并附实验依据。
4. ablation 4 组 + 2 数据集对比表完成。

---

## 八、待决事项

| 编号 | 事项 | 建议值 |
|------|------|--------|
| D1 | serve 并行训练数上限 | 2 |
| D2 | 搜索 `time_limit` 上限 | 300–600 s |
| D3 | \(\chi_{\min}\) 统一取值 | 0.28 |
| D4 | anti-prior 主策略 | target 补集 + action 违反 |
| D5 | 定标主指标 | \(J^{\text{obs}}\) 为主，\(G\) 稳定性为辅 |
| D6 | \(s_{\min}\) / `prior_semantic_weight` | 0.25 / 0.15（W4 E5 定标） |

---

## 附录：相关文档

- [ALGORITHM_PROJECT_ANALYSIS.md](./ALGORITHM_PROJECT_ANALYSIS.md)
- [../README.md](../README.md)

*维护：代码变更后同步更新 §三附；参数定标完成后更新 §4.1 定标状态列。*
