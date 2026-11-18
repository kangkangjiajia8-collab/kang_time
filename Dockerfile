
FROM python:3.10-slim

# ================================================================
# 1) 解压已打包好的 conda 环境
# ================================================================
COPY mutiagent_cpuv1.tar.gz /tmp/
RUN mkdir -p /opt && \
    tar -xzf /tmp/mutiagent_cpuv1.tar.gz -C /opt && \
    rm -f /tmp/mutiagent_cpuv1.tar.gz

# ================================================================
# 2) 使用 conda 环境里的 python / pip
# ================================================================
ENV PATH=/opt/mutiagent_cpuv1/bin:$PATH
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV CUDA_VISIBLE_DEVICES=""
ENV R_HOME=/opt/mutiagent_cpuv1/lib/R
ENV PIP_NO_CACHE_DIR=1

# ================================================================
# 3) 清理旧 editable / namespace 安装残留
#    否则 Python 启动时会先处理这些 .pth，导致报错
# ================================================================
RUN find /opt/mutiagent_cpuv1/lib/python3.10/site-packages \( \
      -name "__editable__.autoapex*.pth" -o \
      -name "__editable___autoapex*_finder.py" -o \
      -name "autoapex*-nspkg.pth" \
    \) -delete || true

# ================================================================
# 4) 拷贝源码到构建目录
#    注意：README.md 必须带进去，否则你的 setup 逻辑会报
#    /build/README.md 不存在
# ================================================================
WORKDIR /build

COPY VERSION        ./VERSION
COPY README.md ./README.md

# 如果你项目根目录还有这些文件，而且确实存在，建议取消注释
COPY pyproject.toml ./pyproject.toml
# COPY setup.py       ./setup.py
COPY setup.cfg      ./setup.cfg
COPY LICENSE        ./LICENSE

COPY common/        ./common/
COPY core/          ./core/
COPY features/      ./features/
COPY tabular/       ./tabular/
COPY multimodal/    ./multimodal/
COPY timeseries/    ./timeseries/
COPY eda/           ./eda/
COPY autoapex/      ./autoapex/
COPY dist/serve/   ./serve/

# ================================================================
# 5) 在容器内做普通安装，不再使用 editable
#    --no-deps：不重新装第三方依赖
#    --no-build-isolation：尽量使用当前环境，减少额外构建环境干扰
# ================================================================
RUN /opt/mutiagent_cpuv1/bin/python -m pip install --no-deps --no-build-isolation ./common && \
    /opt/mutiagent_cpuv1/bin/python -m pip install --no-deps --no-build-isolation ./core && \
    /opt/mutiagent_cpuv1/bin/python -m pip install --no-deps --no-build-isolation ./features && \
    /opt/mutiagent_cpuv1/bin/python -m pip install --no-deps --no-build-isolation ./tabular && \
    /opt/mutiagent_cpuv1/bin/python -m pip install --no-deps --no-build-isolation ./multimodal && \
    /opt/mutiagent_cpuv1/bin/python -m pip install --no-deps --no-build-isolation ./timeseries && \
    /opt/mutiagent_cpuv1/bin/python -m pip install --no-deps --no-build-isolation ./eda && \
    /opt/mutiagent_cpuv1/bin/python -m pip install --no-deps --no-build-isolation ./autoapex

# ================================================================
# 6) 安装后做一次快速验证
# ================================================================
RUN /opt/mutiagent_cpuv1/bin/python -c "import autoapex; print('autoapex ok')" && \
    /opt/mutiagent_cpuv1/bin/python -c "import autoapex.timeseries; print('autoapex.timeseries ok')"

# ================================================================
# 7) 删除仅用于安装的源码目录，减少镜像体积
#    serve 保留，因为运行时要用
# ================================================================
RUN rm -rf /build/common \
           /build/core \
           /build/features \
           /build/tabular \
           /build/multimodal \
           /build/timeseries \
           /build/eda \
           /build/autoapex

# ================================================================
# 8) 启动服务
# ================================================================
EXPOSE 8003
WORKDIR /build/serve
CMD ["/opt/mutiagent_cpuv1/bin/python", "-u", "main.py"]