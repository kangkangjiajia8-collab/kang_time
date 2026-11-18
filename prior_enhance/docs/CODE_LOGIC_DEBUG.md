# prior_enhance 代码逻辑速查（Phase 1）

## 入口

`examples/run_demo.py` → `PriorEnhancePipeline.run_single_window()`

## 调用链

```
preprocess_csv()
  └─ build_windows() → build_forecast_task()
       ├─ build_episode_store_from_history()  # D
       ├─ parse_event_context()               # E_m
       ├─ retrieve_episodes()                 # D_m
       ├─ episodes_for_novelty_pool()         # E_m^hist
       └─ compute_context_state()             # S_m
  └─ generate_priors_for_task()              # Z_m
       ├─ build_prior_generation_prompt()
       ├─ DashScopeLLMClient.chat()
       └─ validate_prior_batch() + coverage_rules
```

## 关键配置（config.py）

| 字段 | 默认 |
|------|------|
| `llm.min_priors` | 3 |
| `llm.max_priors` | 8 |
| `prior.confidence_lambda` | 0.5 |
| `prior.episode_ref_k` | 10 |
| `text_encoder.model_name` | BAAI/bge-base-zh-v1.5 |

## S_m 公式落地

```python
# n_m
n_eff = count(c_τ == c_m and y valid)
n_m = clip(n_eff / n_ref, 0, 1)

# ν_m
s_max = max cosine(g(E_m), g(E_i)) for E_i in E_m^hist
nu_m = 1 - s_max

# d_m
d_m = clip(d_m_mu + d_m_sigma, 0, 1)
```

## b 校准

```python
b_evidence = 0.4*n_m + 0.3*(1-nu_m) + 0.3*min(|D_m|/K, 1)
confidence = λ*b_llm + (1-λ)*b_evidence
```

## 未实现模块（勿在 Phase 1 中查找）

- `slot_batch_mode`（已删除）
- `xi_m`（论文当前阶段不做）
- `τ_{j,m}`, `φ_{j,m}(r)`, `ψ_{m,k}(r)`, `p_{m,k}(r)`
- anti-prior 搜索对照采样
- `recipe/builder.py` 与主 pipeline 的接线

## 日志阶段

`[PREPROCESS]` → `[WINDOW]` → `[CONTEXT]` → `[PRIOR]` → `[PIPELINE]`
