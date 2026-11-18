#!/usr/bin/env bash
# 在 conda mutiagent 环境下运行 prior_enhance demo
set -euo pipefail

MUTIAGENT_PYTHON="${MUTIAGENT_PYTHON:-/home/forecast/anaconda3/envs/mutiagent/bin/python}"
AUTOAPEX_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STDLIB_PATCH="${AUTOAPEX_ROOT}/prior_enhance/_stdlib_patch"

cd "$AUTOAPEX_ROOT"

if [[ ! -x "$MUTIAGENT_PYTHON" ]]; then
  echo "错误: 未找到 mutiagent Python: $MUTIAGENT_PYTHON"
  echo "请先执行: conda activate mutiagent"
  exit 1
fi

# mutiagent Python 3.10.1 缺少 random.randbits，numpy 需要此补丁目录优先于 sys.path
export PYTHONPATH="${STDLIB_PATCH}${PYTHONPATH:+:$PYTHONPATH}"

PIP="${MUTIAGENT_PYTHON%python}pip"
for pkg in openai sentence-transformers chinese-calendar modelscope; do
  "$MUTIAGENT_PYTHON" -c "import ${pkg//-/_}" 2>/dev/null || {
    echo ">>> 正在为 mutiagent 安装 ${pkg} ..."
    "$PIP" install "$pkg" -q
  }
done

# 跳过 AutoApex/R 异质性评分，避免与 torch 编码器初始化争用导致长时间阻塞
export PRIOR_ENHANCE_SKIP_AUTOAPEX_STATS="${PRIOR_ENHANCE_SKIP_AUTOAPEX_STATS:-1}"

echo ">>> 使用: $MUTIAGENT_PYTHON"
exec "$MUTIAGENT_PYTHON" prior_enhance/examples/run_demo.py "$@"
