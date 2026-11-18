# prior_enhance

面向论文 **Evidence-Calibrated LLM Priors for Nonstationary Forecasting Recipe Search** 的工程实现。

> **Phase 1**：数据 → T_m → S_m → Z_m  
> **Phase 2+**：Validation Search → τ^loc → r* → Observation Feedback → 持久 τ, λ

📖 **详细算法分析**：[docs/ALGORITHM_PROJECT_ANALYSIS.md](docs/ALGORITHM_PROJECT_ANALYSIS.md)

## 目录结构

```
prior_enhance/
├── config.py / schemas.py / pipeline.py
├── data/           # 预处理、窗口、episode 检索
├── context/        # S_m = (n_m, ν_m, d_m)，BGE 中文编码器
├── prior/          # Z_m 生成、φ 编译、recipe 采样
├── recipe/         # ForecastingRecipe r=(W,F,M,θ,V,Q)
├── validation/     # train-val 切分、J^val、Gain、χ 门控
├── search/         # prior-guided search 主循环
├── observation/    # O_m 反馈、持久 Prior Memory
└── telemetry/      # 透明日志
```

## 快速开始

```bash
cd /mnt/raid/forecast/Valrant/GZSB/prob/autoapex

# 1. 预下载中文文本编码器
python prior_enhance/examples/download_text_encoder.py

# 2. 配置 DashScope API Key
export DASHSCOPE_API_KEY=sk-xxx

# 3. Phase 1：T_m + Z_m
bash prior_enhance/examples/run_mutiagent.sh \
  --clean-csv prior_enhance/outputs/preprocessed/代理购电_小时_clean.csv

# 4. Phase 1+2+3：含 search + observation
bash prior_enhance/examples/run_mutiagent.sh \
  --clean-csv prior_enhance/outputs/preprocessed/代理购电_小时_clean.csv \
  --with-search --max-trials 8
```

输出：`prior_enhance/outputs/results/holdout/{task_Tm.json, priors_Zm.json, search_result.json, observation_result.json}`

## 与论文符号对应（摘要）

| 论文 | 代码 | 状态 |
|------|------|------|
| H_m, O_m, T_m | `data/window.py`, `ForecastTask` | ✅ |
| S_m, D_m, E_m | `context/`, `episode_store.py` | ✅ |
| Z_m, b 校准 | `prior/generator.py`, `confidence.py` | ✅ |
| φ, ψ, τ^loc, G^val | `prior/compiler.py`, `validation/gain.py` | ✅ |
| r*, E_m^retain | `search/loop.py`, `search/proposal.py` | ✅ |
| G^obs, τ, λ 持久 | `observation/` | ✅ |
| ξ_m 风险、anti-prior、p_exp softmax | `context/state.py`, `recipe/anti_prior_builder.py`, `search/proposal.py` | ✅ |
| AutoApex 真实训练/推理 | `validation/autoapex_client.py` | ✅ 默认 autoapex |

## 文档

| 文档 | 内容 |
|------|------|
| [ALGORITHM_PROJECT_ANALYSIS.md](docs/ALGORITHM_PROJECT_ANALYSIS.md) | **完整算法项目分析（推荐阅读）** |
| [DATA_PIPELINE_WALKTHROUGH.md](docs/DATA_PIPELINE_WALKTHROUGH.md) | 数据管道逐步说明 |
| [CODE_LOGIC_DEBUG.md](docs/CODE_LOGIC_DEBUG.md) | Phase 1 代码速查 |

## 环境变量

- `AUTOAPEX_BASE_URL` — serve 地址，默认 `http://127.0.0.1:8000/api`
- `PRIOR_ENHANCE_EVAL_BACKEND` — `autoapex`（默认）或 `proxy`
- `PRIOR_ENHANCE_EVAL_FALLBACK_PROXY=1` — serve 失败时回退 proxy
- `AUTOAPEX_TRAIN_TIMEOUT` / `AUTOAPEX_INFER_TIMEOUT`
- `PRIOR_ENHANCE_SKIP_AUTOAPEX_STATS=1` — 跳过 R 统计（run_mutiagent.sh 默认）
- `PRIOR_ENHANCE_MAX_TRIALS` / `PRIOR_ENHANCE_VAL_HORIZON`
- `PRIOR_ENHANCE_TEXT_ENCODER_MODEL` / `DASHSCOPE_API_KEY`
