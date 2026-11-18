#!/usr/bin/env python3
"""
测试推理 v1 接口：通过 model_id 加载训练阶段保存的模型包，无需上传 zip。

前置条件：
  1. 服务已启动（默认 http://localhost:8000）
  2. outputs/model_packs/{model_id}.zip 已存在（训练完成后由服务自动写入）

用法示例：
  # 使用已有 model_id
  python test_inference_v1.py --model-id 550e8400-e29b-41d4-a716-446655440000

  # 将本地 zip 注册到 model_packs 后测试（便于无训练环境联调）
  python test_inference_v1.py --register-zip test_results_parallel/train_job1_1775614489634765628_model_pack.zip

  # 训练完成后从 task_status 取 model_id
  python test_inference_v1.py --from-task-status <train_task_id>
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import uuid
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
SERVE_ROOT = SCRIPT_DIR.parent
MODEL_PACKS_DIR = SERVE_ROOT / "outputs" / "model_packs"
DEFAULT_BASE_URL = "http://localhost:8000/api"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "inference_v1_outputs"
DEFAULT_DATA_CSV = SCRIPT_DIR / "data_test.csv"
DEFAULT_BEST_MODEL = "WeightedEnsemble"
# 示例模型包（与 post.py 推理测试同源）
DEFAULT_SAMPLE_ZIP = (
    SCRIPT_DIR
    / "test_results_parallel"
    / "train_job1_1775614489634765628_model_pack.zip"
)


def check_health(base_url: str) -> bool:
    root = base_url.replace("/api", "")
    try:
        resp = requests.get(f"{root}/docs", timeout=10)
        resp.raise_for_status()
        logger.info("服务可用: %s", root)
        return True
    except Exception as e:
        logger.error("无法连接服务 %s: %s", base_url, e)
        return False


def register_zip_to_model_packs(zip_path: Path) -> str:
    """将本地 zip 复制到 outputs/model_packs/{uuid}.zip，返回 model_id。"""
    if not zip_path.is_file():
        raise FileNotFoundError(f"模型包不存在: {zip_path}")
    MODEL_PACKS_DIR.mkdir(parents=True, exist_ok=True)
    model_id = "5_01_gpu_model_pack"
    dest = MODEL_PACKS_DIR / f"{model_id}.zip"
    shutil.copy2(zip_path, dest)
    logger.info("已注册模型包: %s -> %s", zip_path.name, dest)
    return model_id


def fetch_model_id_from_task_status(base_url: str, train_task_id: str) -> str:
    resp = requests.get(
        f"{base_url}/task_status",
        params={"task_id": train_task_id},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    data = (payload.get("data") or {}).get("data") or {}
    model_id = data.get("model_id")
    if not model_id:
        raise ValueError(
            f"任务 {train_task_id} 状态中无 model_id（可能为旧版训练或未成功）。"
            f"响应: {json.dumps(payload, ensure_ascii=False)[:500]}"
        )
    return model_id


def run_inference_v1(
    base_url: str,
    task_id: str,
    model_id: str,
    csv_path: Path,
    best_model: str,
    output_dir: Path,
    timeout: int,
) -> int:
    logger.info("=" * 60)
    logger.info("推理 v1 试验")
    logger.info("  task_id    : %s", task_id)
    logger.info("  model_id   : %s", model_id)
    logger.info("  best_model : %s", best_model)
    logger.info("  data_csv   : %s", csv_path)
    pack = MODEL_PACKS_DIR / f"{model_id}.zip"
    if pack.is_file():
        logger.info("  model_pack : %s (%.1f MB)", pack, pack.stat().st_size / 1e6)
    else:
        logger.warning("  本地未找到 %s，将由服务端解析", pack)
    logger.info("=" * 60)

    with csv_path.open("rb") as cf:
        resp = requests.post(
            f"{base_url}/inference/v1",
            files={
                "data_csv": (csv_path.name, cf, "text/csv"),
            },
            data={
                "task_id": task_id,
                "best_model": best_model,
                "model_id": model_id,
            },
            timeout=timeout,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if resp.status_code != 200:
        logger.error("推理 v1 失败 HTTP %s: %s", resp.status_code, resp.text[:800])
        return 1

    pred_path = output_dir / f"{task_id}_predictions.csv"
    pred_path.write_bytes(resp.content)
    logger.info("预测结果已保存: %s (%d bytes)", pred_path, len(resp.content))

    lines = pred_path.read_text(encoding="utf-8").splitlines()
    logger.info("预测结果预览（前 5 行）:")
    for line in lines[:5]:
        logger.info("  %s", line)

    meta_resp = requests.get(f"{base_url}/inference/{task_id}/metadata", timeout=30)
    if meta_resp.status_code == 200:
        meta = meta_resp.json()
        logger.info("推理元数据 status=%s model_id=%s", meta.get("status"), meta.get("model_id"))

    logger.info("推理 v1 试验完成。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 /api/inference/v1")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model-id", help="训练完成后返回的 model_id（UUID）")
    parser.add_argument(
        "--register-zip",
        type=Path,
        help="将本地 zip 复制到 outputs/model_packs 并生成 model_id 后测试",
    )
    parser.add_argument(
        "--from-task-status",
        metavar="TRAIN_TASK_ID",
        help="从 /api/task_status 读取 model_id",
    )
    parser.add_argument("--data-csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--best-model", default=DEFAULT_BEST_MODEL)
    parser.add_argument("--task-id", default="infer_v1_test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if not check_health(args.base_url):
        return 1

    model_id = args.model_id

    if args.from_task_status:
        try:
            model_id = fetch_model_id_from_task_status(args.base_url, args.from_task_status)
            logger.info("从训练任务 %s 获取 model_id=%s", args.from_task_status, model_id)
        except Exception as e:
            logger.error("%s", e)
            return 1

    if args.register_zip:
        zip_path = args.register_zip
        if not zip_path.is_file() and zip_path == Path(args.register_zip) and not args.register_zip.is_absolute():
            zip_path = DEFAULT_SAMPLE_ZIP if DEFAULT_SAMPLE_ZIP.is_file() else zip_path
        try:
            model_id = register_zip_to_model_packs(Path(zip_path))
        except Exception as e:
            logger.error("%s", e)
            return 1

    if not model_id:
        if DEFAULT_SAMPLE_ZIP.is_file():
            logger.info("未指定 model_id，使用示例 zip 自动注册: %s", DEFAULT_SAMPLE_ZIP.name)
            model_id = register_zip_to_model_packs(DEFAULT_SAMPLE_ZIP)
        else:
            logger.error(
                "请指定 --model-id、--register-zip 或 --from-task-status，"
                "或放置示例 zip 于 test_results_parallel/"
            )
            return 1

    if not args.data_csv.is_file():
        logger.error("数据 CSV 不存在: %s", args.data_csv)
        return 1

    return run_inference_v1(
        base_url=args.base_url,
        task_id=args.task_id,
        model_id=model_id,
        csv_path=args.data_csv,
        best_model=args.best_model,
        output_dir=args.output_dir,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
