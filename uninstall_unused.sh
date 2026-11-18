#!/bin/bash
# ============================================================
# mutiagent 环境清理脚本 - 删除 autoapex 不需要的包
# 经过全项目彻底 import 扫描确认后生成
# 执行前请确认已激活 mutiagent 环境:
#   conda activate mutiagent
# ============================================================

PIP="/home/forecast/anaconda3/envs/mutiagent/bin/pip"

echo ">>> 开始卸载 autoapex 不需要的包..."
echo ""

# ============================================================
# 保留包说明（扫描确认后不在卸载列表中）：
# - shap / slicer         : eda/analysis/explain.py, eda/visualization/explain.py
# - pyod / suod / combo   : eda/analysis/anomaly.py
# - imodels               : tabular/models/imodels/imodels_models.py
# - rpy2 系列             : timeseries/dataset/time_C/Characteristics_Extractor.py
# - mamba-ssm             : timeseries/models/local/timexer_lib/models/S_mamba.py
# - causal-conv1d         : mamba-ssm 底层依赖
# - onnxruntime(CPU)      : tabular/models/rf/compilers/onnx.py
# - onnx / skl2onnx       : tabular/models/tabular_nn/compilers/onnx.py
# - sentencepiece         : transformers[sentencepiece] autoapex 声明依赖
# - missingno             : eda/visualization/missing.py
# - phik                  : eda/analysis/interaction.py
# - seaborn               : timeseries/utils/features.py
# - fastapi/starlette/uvicorn : serve/main.py
# - tabpfn/tabpfn-common-utils: tabular/models/tabpfnv2/tabpfnv2_model.py 运行时加载
# - tabicl                : tabular/models/tabicl/tabicl_model.py 运行时加载
# - mlxtend               : eda/visualization/model.py (待确认，暂保留)
# - protobuf              : serve/main.py 注释中提及兼容性依赖
# ============================================================

# ------------------------------------------------------------
# 1. LangChain / Agent 框架
# ------------------------------------------------------------
echo "[1/8] 卸载 LangChain / Agent 框架..."
pip uninstall -y \
    langchain \
    langchain-core \
    langchain-openai \
    langchain-text-splitters \
    langgraph \
    langgraph-checkpoint \
    langgraph-prebuilt \
    langgraph-sdk \
    langsmith \
    litellm \
    openai \
    tiktoken \
    fastchat \
    prometheus-eval

# ------------------------------------------------------------
# 2. NLP / CV / 多模态（multimodal 子模块用，timeseries/tabular 不需要）
# ------------------------------------------------------------
echo "[2/8] 卸载 NLP / CV / 多模态包..."
$PIP uninstall -y \
    bert-score \
    sentence-transformers \
    seqeval \
    spacy \
    spacy-legacy \
    spacy-loggers \
    thinc \
    srsly \
    preshed \
    murmurhash \
    cymem \
    blis \
    wasabi \
    weasel \
    catalogue \
    confection \
    nlpaug \
    nltk \
    gensim \
    timm \
    torchvision \
    torchaudio \
    scikit-image \
    pdf2image \
    pytesseract \
    PyWavelets \
    tifffile \
    imageio

# sentencepiece 保留：transformers[sentencepiece] autoapex 依赖

# ------------------------------------------------------------
# 3. CUDA / GPU 专用库（CPU Docker 不需要）
# 保留 torch / onnxruntime(CPU) / causal-conv1d / mamba-ssm
# ------------------------------------------------------------
echo "[3/8] 卸载 CUDA / GPU 专用库..."
pip uninstall -y \
    nvidia-cuda-runtime-cu11 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu11 \
    nvidia-cublas-cu12 \
    nvidia-cudnn-cu11 \
    nvidia-cudnn-cu12 \
    nvidia-cufft-cu11 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu11 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu11 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu11 \
    nvidia-cusparse-cu12 \
    nvidia-cusparselt-cu12 \
    nvidia-cufile-cu12 \
    nvidia-nvshmem-cu12 \
    nvidia-nvjitlink-cu12 \
    nvidia-nvtx-cu11 \
    nvidia-nvtx-cu12 \
    nvidia-nccl-cu11 \
    nvidia-nccl-cu12 \
    nvidia-cuda-cupti-cu11 \
    nvidia-cuda-cupti-cu12 \
    nvidia-cuda-nvrtc-cu11 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-ml-py \
    nvidia-ml-py3 \
    tensorrt \
    tensorrt_cu12 \
    tensorrt_cu12_libs \
    tensorrt_cu12_bindings \
    triton \
    cuda-bindings \
    cuda-pathfinder \
    onnxruntime-gpu

# ------------------------------------------------------------
# 4. 遥测 / 监控 / 分布式追踪
# ------------------------------------------------------------
echo "[4/8] 卸载遥测 / 监控包..."
$PIP uninstall -y \
    opencensus \
    opencensus-context \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-proto \
    opentelemetry-exporter-prometheus \
    opentelemetry-semantic-conventions \
    prometheus_client \
    grpcio \
    googleapis-common-protos \
    google-api-core \
    google-auth \
    proto-plus \
    wandb \
    sentry-sdk \
    posthog

# protobuf 保留：serve/main.py 注释中有 protobuf 兼容性设置

# ------------------------------------------------------------
# 5. 深度学习特殊模块（非 autoapex 依赖）
# ------------------------------------------------------------
echo "[5/8] 卸载非 autoapex 深度学习模块..."
$PIP uninstall -y \
    deepspeed \
    megatron-util \
    axial_positional_embedding \
    CoLT5-attention \
    local-attention \
    product_key_memory \
    hyper-connections \
    einx \
    pytorch-metric-learning

# mamba-ssm / causal-conv1d 保留：S_mamba.py 直接使用
# tabpfn / tabicl 保留：tabular 模型运行时加载

# ------------------------------------------------------------
# 6. 时序分析（非 autoapex 核心依赖，且全项目扫描无 import）
# ------------------------------------------------------------
echo "[6/8] 卸载非核心时序包..."
$PIP uninstall -y \
    tsfresh \
    stumpy \
    matrixprofile \
    tbats \
    prophet \
    pmdarima \
    pystan \
    httpstan \
    stanio \
    cmdstanpy

# rpy2 系列保留：Characteristics_Extractor.py 直接使用
# statsforecast / mlforecast / gluonts 保留：autoapex 核心依赖

# ------------------------------------------------------------
# 7. 数据科学辅助工具（全项目扫描确认无 import）
# ------------------------------------------------------------
echo "[7/8] 卸载无引用的辅助工具..."
$PIP uninstall -y \
    featuretools \
    woodwork \
    mlxtend \
    hyperopt \
    optuna \
    numba \
    llvmlite \
    scikit-fuzzy \
    narwhals \
    model-index \
    akshare \
    gdown \
    openmim \
    openxlab \
    opendatalab \
    py4j \
    antlr4-python3-runtime

# 保留：shap/slicer/pyod/suod/combo/imodels/missingno/phik/seaborn/mlxtend
# yellowbrick 保留：eda/visualization/model.py 直接 from yellowbrick 使用

# ------------------------------------------------------------
# 8. 开发工具 / 杂项（全项目无业务引用）
# ------------------------------------------------------------
echo "[8/8] 卸载开发工具和杂项..."
$PIP uninstall -y \
    mypy \
    mypy_extensions \
    types-requests \
    types-setuptools \
    pytest-mypy \
    pylint \
    astroid \
    flake8 \
    pyflakes \
    pycodestyle \
    mccabe \
    isort \
    backcall \
    pickleshare \
    uv \
    virtualenv \
    adagio \
    addict \
    akracer \
    anykeystore \
    appdirs \
    oauthlib \
    requests-oauthlib \
    PySocks \
    py-mini-racer \
    modelscope \
    fastai \
    fastcore \
    fastdownload \
    fastprogress \
    fasttransform

# ruff 保留：pyproject.toml 配置了 ruff 作为 lint 工具，开发时可能用到

echo ""
echo ">>> 卸载完成！"
echo ""
echo ">>> 验证 autoapex 核心功能（请在 mutiagent 环境下执行）:"
echo "    cd /mnt/raid/forecast/Valrant/GZSB/prob/autoapex"
echo "    python -c \"import sys; [sys.path.insert(0,p) for p in ['timeseries/src','common/src','core/src','features/src','tabular/src']]; from autoapex.timeseries import TimeSeriesPredictor; from autoapex.timeseries.models.blocks import EE; from mamba_ssm import Mamba; import shap; import missingno; import seaborn; import fastapi; print('ALL OK')\""
