# AutoApex 项目完整文档

## 1. 文档范围与目标

本文档面向当前仓库（`autoapex`）的使用者、维护者与二次开发者，提供一份完整的项目说明，覆盖：

- 项目定位与能力边界
- 仓库结构与关键模块
- 时序任务（`timeseries`）训练/预测主流程
- API 服务层（`serve`）入口说明
- 模型支持与协变量能力口径
- 依赖环境、部署运行、产物结构

> 说明：本文档基于当前仓库代码与现有文档整理，作为“本项目本地实现与工程落地说明”。

---

## 2. 项目定位

AutoApex 是一个统一的 AutoML 框架，覆盖多任务（表格、时序、多模态）。在本仓库中，重点落在：

- `timeseries`：时序预测能力（训练、评估、推理、模型扩展）
- `serve`：FastAPI 服务封装（分析、训练、推理 API）
- 结果产物沉淀（`results/`、`serve/outputs/` 等）

简化理解：

- **算法与模型层**：由 `autoapex.timeseries` 模块实现；
- **应用与接口层**：由 `serve` 路由封装成 API；
- **数据与任务执行层**：由脚本或 API 参数驱动训练与预测。

---

## 3. 环境与依赖

### 3.1 Python 与核心依赖

从仓库文档和依赖清单可见：

- Python：README 标注支持 `3.10`
- 关键依赖（当前仓库）：
  - `autoapex.*` 子模块（同版本）
  - `gluonts==0.16.2`
  - `lightning / pytorch-lightning`
  - `mlforecast`, `statsforecast`
  - `fastapi`, `uvicorn`
  - `xgboost`, `lightgbm`, `catboost`, `ray` 等

参考文件：

- `requirements.txt`（研发环境）
- `conda GPU 环境：mutiagent`（研发环境）
- `conda CPU 环境：mutiagent_cpuv1`（Docker镜像环境）
### 3.2 安装建议

- 研发环境建议使用独立虚拟环境（venv/conda）
- 对 GPU/CPU 混合环境，需单独确认 `torch`、`transformers`、`chronos` 相关包兼容性

---

## 4. 仓库结构总览（关键目录）

以下是与你当前项目最相关的目录职责：

- `timeseries/`
  - `src/autoapex/timeseries/`：时序核心源码
- `serve/`
  - `main.py`：FastAPI 入口
  - `api/routers/`：分析、训练、推理路由
  - `outputs/`：训练/评估输出（按任务目录组织）
- `release_instructions/`
  - 发布流程与版本操作说明
- `results/`
  - 本地实验结果、可视化图、预测文件

---

## 5. 时序模块架构（`timeseries/src/autoapex/timeseries`）

### 5.1 核心分层

- `predictor.py`
  - 对外的 `TimeSeriesPredictor` 接口，承接 `fit/predict` 等主流程
- `trainer/learner` 相关组件
  - 模型选择、训练调度、模型管理、集成（ensemble）等
- `models/`
  - 各类模型实现（GluonTS、本地统计、tabular、foundation model）
- `transforms/`
  - 目标/协变量缩放（如 `covariate_scaler.py`）
- `utils/features.py`
  - 协变量元数据管理、列分组、特征工程相关逻辑
- `configs/`
  - 预测器预设、超参数预设

### 5.2 模型家族

按实现来源可分为：

- GluonTS Torch：`DeepAR`, `TFT`, `TiDE`, `PatchTST`, `WaveNet`, `DLinear`, `SimpleFeedForward`等
- Local/StatsForecast：`ARIMA`, `ETS`, `Naive` 等
- Tabular-based：`DirectTabular`, `RecursiveTabular`, `PerStepTabular`等
- Foundation Models：`Chronos`, `Chronos2`, `Toto`等

全量详细表（32个模型）：

| 序号  | 模型接口名                             | 模型家族                          | 历史协变量支持 | known协变量支持（类开关） | 证据/说明                                                                                       |
| --- | --------------------------------- | ----------------------------- | ------- | --------------- | ------------------------------------------------------------------------------------------- |
| 1   | `ADIDAModel`                      | Local / StatsForecast         | 否       | 否               | `local/statsforecast.py` 未设置 `_supports_past_covariates`，继承默认 False                         |
| 2   | `ARIMAModel`                      | Local / StatsForecast         | 否       | 否               | 同上，未设置 past 支持开关                                                                            |
| 3   | `AutoARIMAModel`                  | Local / StatsForecast         | 否       | 否               | 同上                                                                                          |
| 4   | `AutoCESModel`                    | Local / StatsForecast         | 否       | 否               | 同上                                                                                          |
| 5   | `AutoETSModel`                    | Local / StatsForecast         | 否       | 否               | 同上                                                                                          |
| 6   | `AverageModel`                    | Local / Naive family          | 否       | 否               | `local/naive.py` 未设置 past 支持开关                                                              |
| 7   | `CrostonModel`                    | Local / StatsForecast         | 否       | 否               | `local/statsforecast.py` 未设置 past 支持开关                                                      |
| 8   | `DLinearModel`                    | GluonTS Torch                 | 否       | 否               | `gluonts/models.py` 未设置 `_supports_past_covariates`                                         |
| 9   | `DeepARModel`                     | GluonTS Torch                 | 否       | 是               | `gluonts/models.py` 仅有 `_supports_known_covariates=True`，未设置 past                           |
| 10  | `DirectTabularModel`              | AutoApex Tabular (MLForecast) | 否       | 是               | 继承 `AbstractMLForecastModel`（`_supports_known_covariates=True`），无 past 开关                   |
| 11  | `DynamicOptimizedThetaModel`      | Local / StatsForecast         | 否       | 否               | `local/statsforecast.py` 未设置 past 支持开关                                                      |
| 12  | `ETSModel`                        | Local / StatsForecast         | 否       | 否               | 继承链无 past 开关                                                                                |
| 13  | `IMAPAModel`                      | Local / StatsForecast         | 否       | 否               | `local/statsforecast.py` 未设置 past 支持开关                                                      |
| 14  | `ChronosModel`                    | Foundation model              | 否       | 否               | `chronos/model.py` 未设置 `_supports_past_covariates`                                          |
| 15  | `Chronos2Model`                   | Foundation model              | **是**   | 是               | `chronos/chronos2.py` 显式 `_supports_known_covariates=True`、`_supports_past_covariates=True` |
| 16  | `NPTSModel`                       | Local                         | 否       | 否               | `local/npts.py` 未设置 past 支持开关                                                               |
| 17  | `NaiveModel`                      | Local / Naive family          | 否       | 否               | `local/naive.py` 未设置 past 支持开关                                                              |
| 18  | `PatchTSTModel`                   | GluonTS Torch                 | 否       | 是               | `gluonts/models.py` 仅 `_supports_known_covariates=True`                                     |
| 19  | `PerStepTabularModel`             | AutoApex Tabular              | 否       | 否（类开关）          | `autoapex_tabular/per_step.py` 未设置 `_supports_past_covariates`                              |
| 20  | `RecursiveTabularModel`           | AutoApex Tabular (MLForecast) | 否       | 是               | 继承 `AbstractMLForecastModel`，无 past 开关                                                      |
| 21  | `SeasonalAverageModel`            | Local / Naive family          | 否       | 否               | `local/naive.py` 未设置 past 支持开关                                                              |
| 22  | `SeasonalNaiveModel`              | Local / Naive family          | 否       | 否               | `local/naive.py` 未设置 past 支持开关                                                              |
| 23  | `SimpleFeedForwardModel`          | GluonTS Torch                 | 否       | 否               | `gluonts/models.py` 未设置 known/past 开关                                                       |
| 24  | `TemporalFusionTransformerModel`  | GluonTS Torch                 | **是**   | 是               | `gluonts/models.py` 显式 `_supports_past_covariates=True`（且支持 known/static/cat）               |
| 25  | `ThetaModel`                      | Local / StatsForecast         | 否       | 否               | `local/statsforecast.py` 未设置 past 支持开关                                                      |
| 26  | `TiDEModel`                       | GluonTS Torch                 | 否       | 是               | `gluonts/models.py` 仅 `_supports_known_covariates=True`                                     |
| 27  | `TotoModel`                       | Foundation model              | 否       | 否               | `toto/model.py` 未设置 `_supports_past_covariates`                                             |
| 28  | `WaveNetModel`                    | GluonTS Torch                 | 否       | 是               | `gluonts/models.py` 仅 `_supports_known_covariates=True`                                     |
| 29  | `ZeroModel`                       | Local / StatsForecast         | 否       | 否               | `local/statsforecast.py` 未设置 past 支持开关                                                      |
| 30  | `MovingAverageInterpolationModel` | Local                         | 否       | 否               | `local/moving_average.py` 未设置 past 支持开关                                                     |
| 31  | `DETMModel`                       | Local / TimeXer related       | 否       | 否               | `local/ag_TimeXer.py` 未设置 past 支持开关                                                         |
| 32  | `EWMAModel`                       | Local                         | 否       | 否               | `local/ewma.py` 未设置 past 支持开关                                                               |

---

## 6. 时序训练与预测主流程（端到端）

以下描述是你当前项目最关键的“从数据到结果”的执行链路。

### 6.1 数据输入与字段规范

标准字段至少包含：

- `item_id`（序列 ID）
- `timestamp`（时间戳）
- `target`（目标列）

并可选携带：

- `known_covariates`：预测期可提供的外生变量
- `past_covariates`：仅历史可用变量
- `static_features`：序列静态属性

### 6.2 协变量元数据识别

在 `features.py` 等流程中，系统会把列分类为：

- known / past / static
- real / categorical

这一层决定“理论上模型可看到什么”。

### 6.3 模型能力过滤

每个模型类通过 `_supports_*` 开关声明能力，例如：

- `_supports_known_covariates`
- `_supports_past_covariates`
- `_supports_static_features`
- `_supports_cat_covariates`（GluonTS 部分模型）

如果模型不支持某类协变量，该类输入会在构造训练 batch 时被忽略。

### 6.4 数据集与切窗

`TimeSeriesDataFrame` 会转换为模型可消费的内部 dataset（GluonTS 路径会映射为标准字段）：

- `FEAT_DYNAMIC_*`（known）
- `PAST_FEAT_DYNAMIC_*`（past-only，前提是模型支持）

随后通过 transformation/splitter 切成训练窗口，产出 batch 输入。

### 6.5 训练、评估、集成

- 单模型训练：按超参和资源约束执行
- 验证评估：基于指定指标（如 `WQL`）
- 可选集成：`enable_ensemble=True` 构建加权集成模型

### 6.6 推理输出

推理产出包含：

- 分位数预测（如 `0.1, 0.5, 0.9`）
- 预测 CSV
- leaderboard / 验证预测
- 可选分析图与统计文件

---

## 7. API 服务层（`serve`）说明

### 7.1 入口与路由
`注意：更详细的文档见serve/目录下README.md`
`serve/main.py` 关键点：

- FastAPI 应用启动
- CORS 全开放（开发态友好，生产建议收敛）
- 挂载路由：
  - `analysis`
  - `training`
  - `inference`

### 7.2 关键兼容配置

`serve/main.py` 顶部设置：

- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`

作用：规避某些依赖链（如 `tsfresh -> matrixprofile`）与新 protobuf 实现的兼容问题。

### 7.3 运行方式

启动：


```bash
cd serve/
bash start_server.sh
```

`main.log记录服务日志` 


---
