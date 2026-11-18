"""预下载中文文本编码器模型（推荐 ModelScope，国内更快）。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prior_enhance.context.text_encoder import (
    ChineseTextEncoder,
    HF_MODEL,
    MODELSCOPE_MODEL_ID,
    resolve_encoder_model_path,
)


def download_via_modelscope() -> str:
    from modelscope import snapshot_download

    cache_dir = os.path.expanduser("~/.cache/modelscope")
    return snapshot_download(MODELSCOPE_MODEL_ID, cache_dir=cache_dir)


def main():
    parser = argparse.ArgumentParser(description="下载 prior_enhance 中文文本编码器")
    parser.add_argument(
        "--model",
        default=None,
        help=f"模型名或本地路径（默认优先 ModelScope 本地缓存，否则 {HF_MODEL}）",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "modelscope", "huggingface"],
        default="auto",
        help="下载源：auto 优先 ModelScope",
    )
    args = parser.parse_args()

    if args.source in ("auto", "modelscope") and args.model is None:
        local = resolve_encoder_model_path(None)
        if Path(local).exists() and (Path(local) / "config.json").exists():
            model_path = local
            print(f"已存在本地模型: {model_path}")
        else:
            print(f"正在从 ModelScope 下载: {MODELSCOPE_MODEL_ID}")
            model_path = download_via_modelscope()
            print(f"下载完成: {model_path}")
    else:
        model_path = args.model or HF_MODEL
        print(f"正在加载: {model_path}")

    enc = ChineseTextEncoder(model_path)
    dim = enc.encode("测试中文语义编码").shape[0]
    sim = enc.max_similarity(
        "区域气温较历史同期偏高，居民侧用电需求可能增加",
        [
            "未来一周气温明显升高，空调负荷上升",
            "预测期主要由正常工作日组成",
        ],
    )
    print(f"模型就绪: {enc.model_name}")
    print(f"向量维度={dim}，示例相似度={sim:.4f}")


if __name__ == "__main__":
    main()
