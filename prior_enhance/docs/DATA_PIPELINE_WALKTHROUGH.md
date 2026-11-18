# prior_enhance 数据流水线（Phase 1）

## 总览

```
CSV → preprocess → holdout/rolling 窗口
     → episode 库 D + Retrieve → D_m
     → S_m = (n_m, ν_m, d_m)   [BGE 中文编码器]
     → LLM → Z_m → Valid → b 校准
```

**不包含**：φ 编译、ψ/p 搜索、anti-prior 采样、τ 更新。

## 1. 预处理

- 输入：原始 CSV（默认 `data/代理购电_小时.csv`）
- 输出：`outputs/preprocessed/*_clean.csv`
- 列映射：`date→timestamp`, `power→value`

## 2. 窗口 T_m

- **holdout（默认）**：末尾 H=72 为 O_m，其余为 H_m
- 落盘：`holdout_history.csv`, `holdout_oos.csv`
- `ForecastTask` 含：`event_context`, `retrieved_episodes`, `episode_count`, `context_state`

## 3. S_m 三维

| 变量 | 计算 | 模块 |
|------|------|------|
| n_m | clip(N_m^eff / N_m^ref) | `scenario.py` + `state.py` |
| ν_m | 1 - s_m^max，g(·)=BGE-base-zh | `text_encoder.py`, `event_novelty.py` |
| d_m | d_m^μ + d_m^σ，场景内标准化 | `state.py` |

ν_m 历史池：同类场景 episode，排除与 t_m 相距 <7 天的滑动块。

## 4. D_m 检索

- 从 H_m 滑动构建 episode 库（块长 168h）
- `Retrieve(T_m,D)`：同场景优先，按事件语义相似度 + 时间距离排序
- top_k 默认 10（`PriorConfig.episode_ref_k`）

## 5. Z_m 生成

- 格式：`{cond, target, action, b_llm}`
- `target` 必须在 `prior/search_space.py` 词汇表内
- `action` 必须以 `avoid_` 开头（结构化负向先验，非搜索对照组）
- `confidence = λ·b_llm + (1-λ)·b_evidence`

## 6. 输出文件

| 文件 | 内容 |
|------|------|
| `task_Tm.json` | T_m + S_m + D_m 摘要 |
| `priors_Zm.json` | 校验后的 Z_m |
| `summary.json` | 运行摘要 |
