# FNSPID 数据（prior_enhance 接入）

来源: [Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID)（CC BY-NC 4.0，非商业）

## 目录

| 路径 | 说明 |
|------|------|
| `raw/news_sample_nasdaq.csv` | 流式采样的 Nasdaq 新闻（约 2 万行，~21MB） |
| `raw/full_history.zip` | 全市场日频股价（~589MB，需完整下载） |
| `processed/{SYMBOL}_daily.csv` | prior_enhance 格式：`timestamp,value,item_id` |
| `processed/{SYMBOL}_event_text.txt` | 决策时刻前新闻摘要 → **E_m** |
| `processed/{SYMBOL}_analysis_summary.json` | 简要统计 |

完整新闻 `nasdaq_exteral_data.csv` 约 **23GB**，未全量下载。

## 下载

```bash
# 国内建议镜像
export HF_ENDPOINT=https://hf-mirror.com

# 默认：新闻样本 (~80MB) + 完整股价 zip (~589MB)
python prior_enhance/data/fnspid/download_fnspid.py

# 按股票单独下载（推荐：不必下 23GB 全量新闻）
python prior_enhance/data/fnspid/download_fnspid.py --symbol AAL --news-only   # → raw/news_AAL.csv
python prior_enhance/data/fnspid/download_fnspid.py --symbol AAL --stock-only  # → raw/stock_AAL.csv (~112KB Range)
python prior_enhance/data/fnspid/download_fnspid.py --symbol AAL               # 新闻 + 股价
```

或后台 wget（已在下载时）:

```bash
wget -c https://hf-mirror.com/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip \
  -O prior_enhance/data/fnspid/raw/full_history.zip
```

## 快速模拟（不等 zip）

新闻用已有 `news_sample_nasdaq.csv`，股价 `--quick` 自动补齐（yfinance/Stooq/合成序列）：

```bash
# 1. 准备数据（约几秒）
python prior_enhance/data/fnspid/prepare_demo.py --symbol AAL --quick

# 2. 无 LLM 窗口 + S_m 检查（可选，归档脚本）
python prior_enhance/examples/archive/simulate_fnspid.py --symbol AAL

# 3. 完整 LLM 先验（需 DASHSCOPE_API_KEY）
python prior_enhance/examples/run_fnspid_demo.py --symbol AAL --quick

# 4. 论文四阶段审计（可用 proxy，无需 serve）
PRIOR_ENHANCE_EVAL_BACKEND=proxy python prior_enhance/examples/validate_fnspid_paper.py --symbol AAL
```

zip 下完后去掉 `--quick`，改用 FNSPID 官方股价。

## 整理 + 分析（完整 zip）

```bash
python prior_enhance/data/fnspid/prepare_demo.py --symbol AAL
```

## 接入先验增强

```bash
python prior_enhance/examples/run_fnspid_demo.py --symbol AAL
python prior_enhance/examples/run_fnspid_demo.py --symbol AAL --with-search --max-trials 8
```

新闻通过 `build_forecast_task(event_text=...)` 写入 `E_m`，影响：

- LLM prompt 中的事件描述
- `ν_m = 1 - s_max`（与历史 episode 文本的语义距离）
- `b^{evidence}` 校准

## 样本分析（news_sample_nasdaq.csv）

- 列: `Date, Article_title, Stock_symbol, Url, Publisher, Article, *summary*`
- 样本约 3300 行（流式截取），日期约 2009–2023
- 符号分布（样本内）: AA, AAL, A 等（按 CSV 字节序，非全库分布）
