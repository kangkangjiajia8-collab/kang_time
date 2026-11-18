# 计划进度文档

> **项目**：Evidence-Calibrated LLM Priors for Nonstationary Forecasting Recipe Search  
> **工程目录**：`prior_enhance/`  
> **更新日期**：2026-06-12  
> **当前进度**：第一步、第二步、第三步已完成；第四步进入计划阶段

---

## 五步计划总览

| 步骤 | 对应论文阶段 | 核心目标 | 状态 |
|------|-------------|----------|------|
| **第一步** | 数据组织 + 预测任务 + 上下文状态 | 构造 \(T_m\)、\(S_m\)、\(\mathcal{D}_m\)，严格 H_m/O_m 边界 | ✅ 已完成 |
| **第二步** | 结构化 LLM Prior + 可靠性初始化 | 生成 \(Z_m\)，编译 \(\phi\)，初始化 \(\tau^0\)、\(\lambda_m\) | ✅ 已完成 |
| **第三步** | 窗口内搜索 + 最终方案 + 外推 | Fast-Loop：\(p_{m,k}\) 采样 → \(r_m^\*\) → \(\hat{y}_{\mathcal{O}_m}\) | ✅ 已完成 |
| **第四步** | Observation Feedback + 持久记忆 | Slow-Loop：\(G^{\text{obs}}\) → \(\tau_{j,m+1}\)、\(\lambda_{m+1}\)，多窗传递 | 📋 计划中 |
| **第五步** | 实验验证与工程落地 | 参数定标、消融、多数据集、AutoApex 生产化 | 📋 未开始 |

**核心链条**（论文）：

\[
Z_m \to \phi_{j,m}(r) \to \psi_{m,k}(r) \to p_{m,k}(r) \to G_{j,m} \to \tau_{j,m}
\]

---

## 第一步：数据组织与上下文感知任务构建（已完成）

### 目标

在滚动窗口 \(m\) 下，仅使用 \(t_m\) 之前可见数据，构造预测任务七元组 \(T_m\) 与上下文状态 \(S_m\)，并完成历史 episode 检索 \(\mathcal{D}_m\)。

### 已完成内容

#### 1.1 数据预处理与窗口切分

| 能力 | 实现 | 关键文件 |
|------|------|----------|
| 全量数据 \(\mathcal{D}_{\text{full}}\) 清洗 | CSV 列归一化、缺失填充、落盘 clean.csv | `data/preprocessor.py` |
| Holdout 单窗口 | 末尾 \(H\) 点为 \(\mathcal{O}_m\)，其余为 \(\mathcal{H}_m\) | `data/window.py::build_holdout_window` |
| Rolling 多窗口切分（数据结构） | stride 滚动，生成多个 WindowSlice | `data/window.py::build_rolling_windows` |
| 信息边界 | 搜索/LLM 阶段不读 O_m | `search/loop.py` 仅读 history_df |

#### 1.2 预测任务 \(T_m\)

| 分量 | 状态 | 说明 |
|------|------|------|
| \(\mathcal{H}_m\) | ✅ | 落盘 `{tag}_history.csv` |
| \(\mathcal{O}_m\) | ✅ | 落盘 `{tag}_oos.csv`，封存至 observation 阶段 |
| \(E_m\) 事件上下文 | ✅ | `parse_event_context`；FNSPID 支持新闻文本注入 |
| \(K_m\) 领域知识 | ✅ | `config.domain_knowledge` 写入 LLM prompt |
| \(H, B, l\) | ✅ | `WindowConfig` / `SearchConfig` |
| \(X_m^+\) 已知未来外生变量 | ⚠️ | 字段已有，电价/金融 demo 多为空 |

**结构化任务对象**：`schemas.ForecastTask` → 落盘 `task_Tm.json`

#### 1.3 上下文状态 \(S_m = \Gamma(T_m, \mathcal{D}_m)\)

| 因子 | 公式 | 实现 |
|------|------|------|
| \(n_m\) | \(\clip(N_m^{\text{eff}}/N_m^{\text{ref}}, 0, 1)\) | `context/state.py::_compute_n_m` |
| \(\nu_m\) | \(1 - s_m^{\max}\)，BGE 中文编码器 | `context/event_novelty.py` + `context/text_encoder.py` |
| \(d_m\) | \(d_m^\mu + d_m^\sigma\)，场景内标准化 | `context/state.py::_compute_d_m` |
| \(\xi_m\) | 波动聚集 + 异常率 + 跳跃强度 | `context/state.py`（已接入，供后续 \(\mu_m\)、\(\omega_m\) 使用） |
| 离散 cond 标签 | low/medium/high_sample、familiar/novel_event、stable/moderate/severe_drift | `context/labels.py` |

#### 1.4 Episode 检索 \(\mathcal{D}_m\)

| 能力 | 实现 |
|------|------|
| 历史 episode 库 \(\mathcal{D}\) | `data/episode_store.py::build_episode_store_from_history` |
| 语义 + 场景 + 时间加权检索 | `retrieve_episodes` |
| FNSPID 新闻注入 episode | `news_df` 参数，新闻摘要进入历史 episode 文本池 |

#### 1.5 FNSPID 金融数据验证（第一步侧）

- 数据：`data/fnspid/processed/AAL_daily.csv`（4595 行，2005–2023）
- 事件：`AAL_event_text.txt`（决策前 21 天最多 10 条新闻摘要）
- Holdout：\(t_m = 2023-12-20\)，H_m=4590 行，O_m=5 行
- 上下文示例：\(n_m=0.17\)，\(\nu_m=0.25\)，\(d_m=1.0\)，cond=`[low_sample, familiar_event, severe_drift]`
- Episode 检索：\(|D_m|=10\)，含新闻 episode

### 第一步产出物

```
prior_enhance/outputs/windows/holdout/holdout_history.csv   # H_m
prior_enhance/outputs/windows/holdout/holdout_oos.csv     # O_m
prior_enhance/outputs/results/*/task_Tm.json              # T_m + S_m + D_m 摘要
```

---

## 第二步：结构化 LLM Prior 生成与可靠性初始化（已完成）

### 目标

大模型生成可检验结构化先验 \(Z_m\)，经校验与编译映射为 \(\phi_{j,m}(r)\)，并完成搜索前的 \(\tau_{j,m}^0\)、\(\lambda_m\) 初始化。

### 已完成内容

#### 2.1 结构化 Prior \(Z_m\)

| 能力 | 实现 | 关键文件 |
|------|------|----------|
| LLM 生成 \(z_{j,m}=(\text{cond}, \text{target}, \text{action}, b)\) | DashScope API + 结构化 prompt | `prior/generator.py` |
| Schema / 泄露 / 搜索空间校验 | \(\Valid(z)=1\) 过滤 | `prior/validator.py` |
| 置信度校准 \(b_{j,m}\) | \(b=\lambda b^{LLM}+(1-\lambda)b^{\text{evidence}}\) | `prior/confidence.py` |
| 证据置信度 | \(0.4n_m+0.3(1-\nu_m)+0.3\min(|D_m|/K,1)\) | `prior/confidence.py` |
| 搜索空间约束 | 窗口/特征/模型/集成均在 AutoApex 候选空间内 | `prior/search_space.py` |

**产出**：`priors_Zm.json`（FNSPID AAL 缓存 3 条有效 prior）

#### 2.2 Prior 编译 \(\phi_{j,m}(r)\)

| 能力 | 实现 |
|------|------|
| target 正向重叠（窗口、特征、模型、集成） | `prior/compiler.py::phi_prior_recipe` |
| action 负向规避（avoid_* 规则） | `AVOID_HANDLERS` |
| 综合激活度 \(\phi_{\text{total}}\) | confidence 加权平均 |

#### 2.3 Prior Reliability 初始化

| 公式 | 实现 |
|------|------|
| \(\tau_{j,m}\) 历史相似 prior 加权 | `prior/similarity.py` + `observation/prior_store.py` |
| 无历史时 \(\tau_{j,m}=0.5\) | `init_tau_persistent` |
| \(\lambda_m^{\text{init}}\) 排行榜映射 | `prior/llm_trust.py` + `prior/data/llm_leaderboard.csv` |
| \(a_{j,m}\) cond 与 \(S_m\) 匹配度 | `prior/compiler.py::applicability` |
| \(\tau_{j,m}^0 = \clip(a_j[\lambda_m b + (1-\lambda_m)\tau_{j,m}], \epsilon_\tau, 1-\epsilon_\tau)\) | `init_tau_loc` |
| \(\tau_{j,m,0}^{\text{loc}} = \tau_{j,m}^0\) | `search/loop.py` 搜索入口 |

#### 2.4 工程配套

- 文本编码器：BGE-base-zh，优先本地 ModelScope 缓存（`context/text_encoder.py`）
- 透明日志：prompt / LLM 响应 / audit.jsonl（`telemetry/run_logger.py`）
- Demo 入口：`examples/run_demo.py`、`examples/run_fnspid_demo.py`

### 第二步产出物

```
prior_enhance/outputs/results/*/priors_Zm.json
prior_enhance/outputs/prior_store.jsonl          # 跨 run 历史 prior 库（observation 后追加）
prior_enhance/outputs/logs/*_prompt_*.txt        # LLM prompt 快照
```

---

## 第三步：保留探索的 Prior-Guided Search 与泛化外推（已完成）

### 目标

在窗口 \(m\) 内仅用 \(\mathcal{H}_m\) 完成 recipe 搜索闭环：混合采样 → 验证反馈 → \(\tau^{\text{loc}}\) 快环更新 → 选 \(r_m^\*\) → 全量 H_m 训练并预测 O_m。

### 已完成内容

#### 3.1 Prior-Guided 采样分布

| 公式 | 实现 | 关键文件 |
|------|------|----------|
| \(\psi_{m,k}(r) = \sum_j \tau_{j,m,k}^{\text{loc}} \phi_{j,m}(r)\) | ✅ | `search/loop.py::_psi_score` |
| \(q_{m,k}(r) \propto \exp\{\beta_k \psi_{m,k}(r)\}\) | ✅ softmax 候选池采样 | `search/proposal.py` |
| \(p_{m,k} = (1-\rho_k) q + \rho_k p_{\text{exp}}\) | ✅ 默认 \(\rho=0.15\) | `search/loop.py::_sample_batch` |
| \(p_{\text{exp}}\) 三类对照 | ✅ random / low-prior / anti-prior | `recipe/anti_prior_builder.py` |

#### 3.2 窗口内 Validation（Fast-Loop）

| 能力 | 实现 |
|------|------|
| H_m 内时间有序 train-val 切分 | `validation/split.py::make_holdout_val_split` |
| \(J^{\text{val}} = L + \lambda_C C + \mu_m R^{\text{val}}\) | `validation/objective.py` |
| \(\mu_m = \mu(S_m)\)（含 \(\xi_m\)） | `validation/objective.py::mu_from_context` |
| Validation Gain 多元回归 | \(J \sim \phi + \gamma^\top x_m(r) + \varepsilon\) | `validation/gain.py` |
| \(\chi\) 四因子门控 | sample / balance / quality / stability | `validation/gain.py::evidence_quality` |
| \(\tau^{\text{loc}}\) logit 更新（不写持久 \(\tau\)） | `validation/gain.py::update_tau_logit` |

#### 3.3 最终方案选择与外推

| 能力 | 实现 |
|------|------|
| \(r_m^\* = \arg\min J^{\text{val}}\)（无 LLM 评分项） | `search/loop.py` |
| \(f_m^\* = \Train(r_m^\*, \mathcal{H}_m)\) | `validation/evaluator.py` + AutoApex `/api/train` |
| \(\hat{y}_{\mathcal{O}_m}^{r_m^\*} = f_m^\*(X_m^+)\) | `validation/evaluator.py::evaluate_on_oos` |
| \(\mathcal{E}_m^{\text{retain}} = \mathcal{E}^{\text{top}} \cup \mathcal{E}^{\text{explore}}\) | `search/loop.py` retain_ids |

#### 3.4 评估后端

| 模式 | 用途 | 环境变量 |
|------|------|----------|
| AutoApex serve | 真实训练/推理 | `PRIOR_ENHANCE_EVAL_BACKEND=autoapex` |
| Proxy | 快速调试、论文流程验证 | `PRIOR_ENHANCE_EVAL_BACKEND=proxy` |

#### 3.5 FNSPID/AAL 验证结果（第三步）

| 指标 | 结果 |
|------|------|
| 搜索预算 | 8 trials，`explore_ratio=0.15` |
| 最优方案 \(r_m^\*\) | recipe `65f43659d8f0` |
| 窗口内验证 \(J^{\text{val}}\) | 0.374 |
| O_m 泛化 \(L^{\text{obs}}\)（MAPE） | ≈ 7.5% |
| O_m 综合 \(J^{\text{obs}}\) | 0.391 |
| retain 方案数 | 4（top + explore） |

验证脚本：`examples/validate_fnspid_paper.py`

### 第三步产出物

```
prior_enhance/outputs/results/*/search_result.json    # 搜索摘要、tau_loc、retain_ids
prior_enhance/outputs/results/*/forecast_r_star.json   # r* 在 O_m 上的预测与真实值
prior_enhance/outputs/results/*/paper_audit.json       # 四阶段审计摘要
```

### 第三步边界说明（与第四步的衔接）

- 窗口内 \(\tau^{\text{loc}}\) 更新**已完成**，且**不写入**持久 \(\tau_{j,m}\)（符合论文 Fast/Slow 分离）。
- holdout 单次验证中，O_m 揭晓后的 Observation 慢环已在 `observation/feedback.py` **实现并 smoke 通过**，但尚未作为独立「第四步」做系统化多窗实验与定标——归入第四步计划。

---

## 第四步：跨窗口 Observation Feedback 与持久记忆校准（计划）

### 目标

当 \(\mathcal{O}_m\) 真实值揭晓后，对 retain 方案集合计算 \(J^{\text{obs}}\)、\(G^{\text{obs}}\)，慢速更新持久 \(\tau_{j,m+1}\) 与 \(\lambda_{m+1}\)，并在多个滚动窗口间传递 prior memory，形成可审计的因果反馈闭环。

### 4.1 核心算法任务（代码骨架已有，待系统化验收）

| 任务 | 论文公式 | 当前代码 | 计划工作 |
|------|----------|----------|----------|
| O_m 观测目标 | \(J^{\text{obs}} = L^{\text{obs}} + \lambda_C C + \mu_m R^{\text{obs}}\) | `objective.py::j_obs` | 在 AutoApex 后端下复验；补充 \(R^{\text{obs}}\) 分项日志 |
| Observation Gain 回归 | \(G^{\text{obs}} = -\beta^{\text{obs}}\) | `validation/gain.py` | holdout + rolling 对比回归 vs contrastive |
| 配对断层扫描（备选） | \(G^{\text{obs}} = E[J(r^-)-J(r^+)]\) | `compute_contrastive_obs_gain` | 小样本时作为主估计器评估 |
| \(\chi^{\text{obs}}\) 门控 | 四因子乘性，balance strict | `evidence_quality` | \(\rho>0\) 下强制 balance，统一 `chi_min` |
| 动态遗忘率 | \(\omega_m = \omega_{\max}\sigma(\alpha_0 - \alpha_n n + \alpha_\nu \nu + \alpha_d d + \alpha_\xi \xi)\) | `memory.py::omega_forget_rate` | 多窗实验验证遗忘曲线 |
| 持久 \(\tau\) 更新 | \(l_{j,m+1} = (1-\omega_m) l_{j,m} + \eta_{\text{obs}} \chi \Delta^{\text{obs}}\) | `observation/memory.py` | 验证 \(\tau\) 非平凡变化（非全 0.5） |
| 全局 \(\lambda\) 更新 | \(\lambda_{m+1}\) 证据加权 | `observation/memory.py` | 与排行榜冷启动 \(\lambda^{\text{init}}\) 对比 |
| prior_store 跨窗追加 | JSONL 持久库 | `observation/prior_store.py` | 第 2 窗起相似 prior 检索命中率 > 0 |

### 4.2 滚动多窗口编排（计划重点）

| 任务 | 说明 | 计划产出 |
|------|------|----------|
| `pipeline.run_rolling_windows()` | 已实现 API，待主实验调用 | 5–8 窗 walk-forward 一键脚本 |
| 窗口间 memory 传递 | 每窗 observation 后更新 `PriorMemory`，下一窗 `init_tau_loc` 读取 prior_store | \(\tau_{j,m}\)、\(\lambda_m\) 轨迹 JSON |
| 信息边界审计 | 每窗搜索阶段断言不读未来 O_m | `docs/LEAKAGE_AUDIT.md` 或审计日志 |
| FNSPID rolling | AAL 日频，stride=H=5，多决策点 | `outputs/results/win_*/` 系列产物 |
| 代理购电 rolling | 小时数据，验证非平稳场景 | 与金融场景对比 summary |

**建议配置**：

```python
cfg.window.mode = "rolling"
cfg.window.forecast_days = 5          # 日频金融
cfg.search.max_trials = 6              # 算力可控
cfg.search.explore_ratio = 0.15
```

### 4.3 实验与验收标准（第四步交付）

| 编号 | 实验 | 验收条件 |
|------|------|----------|
| E4-1 | Holdout + AutoApex 全链 | `max-trials≥8`，`observation_result.json` 含非空 gain_summary |
| E4-2 | 5 窗 rolling（proxy 预跑） | 每窗 `search_result` + `observation_result` 齐全 |
| E4-3 | 3 窗 rolling（AutoApex） | 真实 \(J^{\text{obs}}\) 可对比 |
| E4-4 | \(\tau\) 链可视化 | 至少 1 条 prior 的 \(\tau_{j,m}\) 跨窗变化 ≠ 常数 0.5 |
| E4-5 | Gain 因果性检查 | strict balance 下 \(G^{\text{obs}}\) accepted 率 > 0；报告 \(\beta\) 符号稳定性 |

**第四步完成标志**：

1. 多窗 rolling 可在 mutiagent 下一键复现；
2. `prior_store.jsonl` 随窗增长，第 2 窗起 \(\tau_{j,m}\) 初始化受历史相似 prior 影响；
3. Observation 慢环与 Validation 快环职责分离经日志可审计；
4. FNSPID + 代理购电至少各 1 套 rolling 实验 summary。

### 4.4 第四步时间计划（建议 1–2 周）

| 阶段 | 时间 | 内容 |
|------|------|------|
| 4.1 | 2 天 | Holdout AutoApex 全链复验；统一 `chi_min`；gain 日志增强 |
| 4.2 | 2 天 | FNSPID 5 窗 rolling（proxy）+ \(\tau/\lambda\) 轨迹导出 |
| 4.3 | 2 天 | 3 窗 AutoApex rolling；proxy vs autoapex \(J^{\text{obs}}\) 对比 |
| 4.4 | 1 天 | 代理购电 holdout/rolling 接入；信息边界审计 |
| 4.5 | 1 天 | 第四步复盘文档 + `rolling_summary.json` 归档 |

### 4.5 第四步风险与依赖

| 风险 | 缓解 |
|------|------|
| AutoApex serve 训练耗时 | `time_limit≤300`；rolling 用较小 `max_trials` |
| \(\chi\) 门控过严导致 \(\tau\) 不更新 | 调 `chi_min`；确保 `explore_ratio>0` 且 strict balance |
| FNSPID 新闻与股价 ticker 混用 | 文档标注；后续清洗 news 标的 |
| 单窗 holdout 无法验证 memory 链 | 必须以 rolling 为主实验 |

---

## 第五步：实验验证与工程落地（远期计划，概要）

> 第四步完成后进入第五步，此处仅列方向，详细计划待第四步验收后更新。

| 方向 | 内容 |
|------|------|
| 参数定标 | \(\rho_k\)、\(\beta_k\)、\(\chi_{\min}\)、\(b^{\text{evidence}}\) 权重等 → `docs/CALIBRATION.md` |
| 消融实验 | 无 prior / 无 explore / 无 memory / 无 \(\tau\) 更新 |
| 多数据集 | 代理购电（小时）、FNSPID（日频）、重过载场景（待数据） |
| 概率预测与 Coverage | \(Q\) 校准、\(\Coverage_m\) 硬约束（论文扩展项） |
| 一键复现 | `reproduce.sh`：holdout + rolling + 审计报告 |
| 论文图表 | \(\tau\) 轨迹、Gain 分布、\(r^\*\) 命中率 |

---

## 附录：关键命令

```bash
conda activate mutiagent
cd /mnt/raid/forecast/Valrant/GZSB/prob/autoapex

# 第一步+第二步：FNSPID 任务与先验（需 DASHSCOPE_API_KEY）
python prior_enhance/examples/run_fnspid_demo.py --symbol AAL

# 第三步：含搜索与外推（proxy 快速验证）
PRIOR_ENHANCE_EVAL_BACKEND=proxy \
PRIOR_ENHANCE_SKIP_AUTOAPEX_STATS=1 \
PRIOR_ENHANCE_TEXT_ENCODER_LOCAL_ONLY=1 \
python prior_enhance/examples/validate_fnspid_paper.py --symbol AAL --max-trials 8

# 第三步：AutoApex 真实训练（需 serve 已启动）
PRIOR_ENHANCE_EVAL_BACKEND=autoapex \
python prior_enhance/examples/run_fnspid_demo.py --symbol AAL --with-search --max-trials 8

# 第四步（计划）：rolling 多窗 — 待封装示例脚本
# cfg.window.mode = "rolling"
# pipeline.run_rolling_windows(csv_path, max_windows=5, with_search=True)
```

---

## 附录：文档索引

| 文档 | 说明 |
|------|------|
| [END_TO_END_LOGIC.md](./END_TO_END_LOGIC.md) | 端到端逻辑与模块对照 |
| [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) | 公式审计与月度实施计划 |
| [ALGORITHM_PROJECT_ANALYSIS.md](./ALGORITHM_PROJECT_ANALYSIS.md) | 算法与项目结构分析 |

*维护说明：完成第四步里程碑后，将本文「第四步计划」改为「已完成」，并展开第五步详细计划。*
