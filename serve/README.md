# AutoApex Time Series Analysis Service

基于 **FastAPI** 与 **AutoApex** 的时间序列分析与预测后端服务。  
采用 **Router–Service 分层**，提供 **数据探索（EDA）→ 特征工程 → 模型训练 → 推理预测** 全流程接口，适用于时间序列建模与批量预测场景。

---

## 项目特性

### 数据分析（Analysis）
- 月度分布箱线图
- 缺失值自动填补（auto / mean / median / forward_fill 等）
- 时间序列特征评分（趋势性、周期性、平稳性等）
- 多变量相关性矩阵（Pearson / Spearman）

### 特征工程（Feature Engineering）
- 异步全量特征生成，支持自定义 tsfresh 参数（JSON 配置）
- 基于 FDR 的特征筛选与结果下载

### 模型训练（Training）
- 基于 **TimeSeriesPredictor**，支持 HPO 与多种集成策略（Weighted / Per-Item / Simple / Median 等）
- 训练数据自动备份至 `outputs/train_data/`
- 训练结束后自动生成 Leaderboard 与验证集预测结果
- 支持单独打包下载榜单（`/generate_leaderboard`）

### 推理预测（Inference）
- 上传训练好的模型 ZIP 与新数据 CSV
- 输出预测均值及分位数区间（CSV）

### 存储规范
- 所有输出按 `task_id` 隔离存放于 `outputs/` 下各子目录

---

## 目录结构

```text
serve/
├── api/
│   ├── routers/                    # 路由层：HTTP 接口与参数校验
│   │   ├── analysis.py             # 分析 & 特征工程
│   │   ├── training.py             # 训练、任务状态、日志、榜单下载
│   │   └── inference.py            # 推理预测
│   └── services/                   # 业务层：核心逻辑
│       ├── analysis_service.py     # 箱线图、缺失值、特征生成与筛选、相关性
│       ├── data_service.py         # 上传保存、临时目录、任务状态、TSDF 准备、日志
│       ├── predict_service.py      # 推理
│       └── train_service.py        # 训练与 Leaderboard 打包
├── configs/
│   ├── config.py                   # 路径、日志、环境
│   ├── hpo_template.py             # 超参搜索空间解析
│   └── fe_template.py              # 特征工程参数解析
├── example/
│   └── post.py                     # 示例客户端（请求示例与轮询）
├── outputs/                        # [运行时生成]
│   ├── analysis/
│   ├── feature/
│   ├── train_models/
│   ├── train_data/
│   ├── inference/
│   ├── metrics_model/
│   └── temp/
├── logs/                           # [运行时生成] 任务日志
├── main.py                         # FastAPI 入口
└── start_server.sh                 # 启动脚本
```

---

## 安装与依赖

- **Python 3.10+**
- 本项目依赖仓库内 **common、core、features、timeseries、tabular** 等模块，需在 **项目根目录** 下运行或正确设置 `PYTHONPATH`。

```bash
# 若已从项目根安装 AutoApex
pip install fastapi uvicorn pandas numpy python-multipart

# 或按仓库要求安装完整环境（含 AutoApex / autoapex.timeseries）
```

**Protobuf 兼容性**：特征工程依赖 tsfresh → matrixprofile，其自带的 `_pb2` 与 protobuf 4.x+ 不兼容。`main.py` 已通过设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 规避；若你希望用 C++ 实现，可改为安装 `protobuf>=3.19,<4`（如 `pip install 'protobuf>=3.19,<4'`）。

---

## 快速启动

在 `serve/` 下：

```bash
cd serve
export PYTHONPATH="${PYTHONPATH}:$(pwd)"   # 或由 start_server.sh 设置
python main.py
# 或
chmod +x start_server.sh
./start_server.sh
```

服务默认地址：**http://0.0.0.0:8000**  
API 文档：**http://0.0.0.0:8000/docs**

---

## 接入 Nacos（服务注册 + 心跳）

本服务支持按 **Nacos 官方 OpenAPI** 自动注册实例（并为临时实例发送心跳）。开启后流程是：

FastAPI 启动 → 注册到 Nacos →（ephemeral=true 时）后台定时 beat → FastAPI 停止 → 从 Nacos 注销实例

### 环境变量

在 `serve/` 下启动前设置：

```bash
export NACOS_ENABLED=true
export NACOS_SERVER_ADDRESSES="http://172.19.2.191:8848"
export NACOS_SERVICE_NAME="你的系统服务名称"

# 可选
export NACOS_INSTANCE_IP="172.19.x.x"          # 不填会自动探测本机出网 IP
export NACOS_INSTANCE_PORT="8000"              # 不填默认取 PORT 或 8000
export NACOS_NAMESPACE_ID="public"
export NACOS_GROUP_NAME="DEFAULT_GROUP"
export NACOS_CLUSTER_NAME="DEFAULT"
export NACOS_EPHEMERAL=true                    # 临时实例（默认 true）
export NACOS_BEAT_INTERVAL_S=5                 # 心跳间隔秒（默认 5）
export NACOS_METADATA_JSON='{"version":"1.0"}' # 元数据（JSON object）

export PORT=8000
python main.py
```

### 对应的 Nacos OpenAPI

本项目目前走 HTTP OpenAPI（无需额外 SDK 依赖），核心接口是：

- 注册实例：`POST /nacos/v1/ns/instance`
- 注销实例：`DELETE /nacos/v1/ns/instance`
- 临时实例心跳：`PUT /nacos/v1/ns/instance/beat`

官方文档可参考 Nacos OpenAPI Guide（Service Discovery / Register instance）。

---

## 接口概览

所有接口前缀：**`/api`**。

### 数据分析 & 特征工程

#### POST `/api/analyze_csv`
生成月度分布箱线图统计。

| 参数 | 类型 | 说明 |
|------|------|------|
| file | File | 必填，时间序列 CSV |
| target_column | Form | 默认 `value` |

**响应**：CSV 文件 `distribution_boxplot.csv`（含 min/q1/median/q3/max 等）。

---

#### POST `/api/fill_missing_values`
时间序列缺失值填补。

| 参数 | 类型 | 说明 |
|------|------|------|
| file | File | 必填 |
| method | Form | 默认 `auto`，可选 mean、median、forward_fill 等 |

**响应**：CSV 文件，响应头 `X-Filled-Status` 表示填补状态（如 true/false）。

---

#### POST `/api/get_characteristics`
时间序列特征评分（趋势性、周期性等）。

| 参数 | 类型 | 说明 |
|------|------|------|
| file | File | 必填 |
| target_column | Form | 默认 `value` |

**响应**：`{"status": "success", "data": { ...scores... }}`

---

#### POST `/api/calculate_correlation`
多变量相关性矩阵。

| 参数 | 类型 | 说明 |
|------|------|------|
| file | File | 必填 |
| method | Form | 默认 `pearson`，可选 spearman 等 |

**响应**：CSV 文件 `correlation_matrix.csv`。

---

#### POST `/api/generate_all_features`（异步）

提交特征生成任务；完成后可通过 `/api/task_status` 轮询，再通过 `/api/select_features_and_download` 下载筛选结果。

**请求体（JSON）**

| 字段 | 类型 | 说明 |
|------|------|------|
| task_id | str | 必填，任务 ID |
| file_name | str | 必填，文件名 |
| file_content_b64 | str | 必填，CSV 内容 Base64 编码（可带 `data:...;base64,` 前缀） |
| max_timeshift | int | 默认 29 |
| min_timeshift | int | 默认 29 |
| custom_params | array | 可选，自定义特征配置，每项 `{"name": "特征名", "params": {...}}` |

**响应**：`{"code": 200, "message": "Feature generation started", "data": {"task_id": "...", "status": "PENDING", "info": "..."}}`

---

#### POST `/api/select_features_and_download`
按 FDR 筛选特征并下载 CSV。

| 参数 | 类型 | 说明 |
|------|------|------|
| task_id | Form | 必填 |
| fdr | Form | 默认 0.05，越小筛选越严 |

**响应**：CSV 文件 `selected_features_{task_id}_fdr{fdr}.csv`。

---

### 模型训练

#### POST `/api/train`

| 参数 | 类型 | 说明 |
|------|------|------|
| task_id | Form | 必填 |
| file | File | 必填，训练 CSV |
| target | Form | 必填，目标列名（如 value） |
| prediction_length | Form | 必填，预测步数 |
| freq | Form | 可选，如 D、H，空则自动推断 |
| eval_metric | Form | 可选 |
| use_quantile | Form | 可选，如 `interval` 启用分位数 |
| use_tsfresh | Form | 必填，是否使用 tsfresh 特征 |
| time_limit | Form | 必填，训练时长（秒） |
| enable_ensemble | Form | 必填，是否启用集成 |
| hyperparameters | Form | 可选，JSON 字符串，模型与超参 |
| ensemble_types | Form | 可选，JSON 字符串或逗号分隔，如 weighted,simple |
| hyperparameter_tune_kwargs | Form | 可选，HPO 配置 JSON |

**响应**：ZIP 流（模型目录打包，含 predictor 与 evaluation_results 等）。

---

#### GET `/api/task_status`
查询异步任务状态（特征生成、训练等）。

| 参数 | 类型 | 说明 |
|------|------|------|
| task_id | Query | 必填 |

**响应**：`{"code": 200, "data": {"status": "PENDING|RUNNING|SUCCESS|FAILED", "message": "...", ...}}`

---

#### GET `/api/task_logs`
获取任务日志行列表。

| 参数 | 类型 | 说明 |
|------|------|------|
| task_id | Query | 必填 |

**响应**：`{"code": 200, "data": ["line1", "line2", ...]}`

---

#### POST `/api/generate_leaderboard`
对已训练任务打包 Leaderboard 与预测结果为 ZIP 供下载。

| 参数 | 类型 | 说明 |
|------|------|------|
| task_id | Form | 必填 |

**响应**：ZIP 文件流（含 leaderboard.csv 及各模型预测等）。

---

### 推理预测

#### POST `/api/inference`

| 参数 | 类型 | 说明 |
|------|------|------|
| task_id | Form | 必填 |
| best_model | Form | 必填，模型名（如 WeightedEnsemble、DeepAR） |
| model_zip | File | 必填，训练接口返回的模型 ZIP |
| data_csv | File | 必填，历史数据 CSV |

**响应**：CSV 文件 `{task_id}_predictions.csv`（预测均值及分位数等）。

---

## 输出文件说明

| 类型 | 路径 |
|------|------|
| 训练模型 | `outputs/train_models/{task_id}/` |
| 训练数据备份 | `outputs/train_data/{task_id}.csv` |
| 特征文件 | `outputs/feature/{task_id}/full_features.csv` |
| 推理结果 | `outputs/inference/{task_id}/predictions.csv` |
| 临时上传 | `outputs/temp/{service_name}/{task_id}/` |
| 任务日志 | `logs/{task_id}.log` |

---

## 示例调用

可参考 `example/post.py` 中的请求示例与轮询逻辑（分析、特征工程、训练、推理等）。
