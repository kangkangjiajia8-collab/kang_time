import os
import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


CONFIG = {
    "BASE_URL": "http://localhost:8111/api",
    "INPUT_CSV": "just.csv",
    "OUTPUT_DIR": "test_results_train",
    "TIMEOUT": 120000,   # 单个请求超时（秒），训练期间阻塞等待
    "NUM_JOBS": 1,    # 并发任务数
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TrainTester:
    def __init__(self):
        self.base_url = CONFIG["BASE_URL"]
        self.input_csv = CONFIG["INPUT_CSV"]
        self.output_dir = CONFIG["OUTPUT_DIR"]
        os.makedirs(self.output_dir, exist_ok=True)

    def check_health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url.replace('/api', '')}/docs", timeout=5)
            resp.raise_for_status()
            logger.info("✅ 服务健康检查通过")
            return True
        except Exception as e:
            logger.error("❌ 无法连接到服务，请确认 FastAPI 已在 8000 端口启动: %s", e)
            return False

    # -------------------- 单次训练（阻塞直到服务返回 zip）--------------------
    def _run_single_train(self, index: int) -> dict:
        """
        提交一个训练请求，阻塞等待服务返回 zip，
        保存 zip 和日志后返回结果摘要。
        """
        task_id = f"train_concurrent_{int(time.time())}_{index}"

        hyperparameters_payload = [
              {
                    "name": "MovingAverageInterpolation",
                    "params": {
                        "window_size": {"start": 24, "end": 60},
                        "max_growth": {"start": 0.15, "end": 0.35},
                        "min_growth": {"start": -0.25, "end": -0.05},
                        "diff_clip_q": [[0.05, 0.95], [0.1, 0.9]],
                        "base_periods": [[1, 2], [1, 2, 3]]
                    }
                },
                # {"name": "DETM", "params": {}},
                {"name": "EWMA", "params": {"forecast_style": "decay_to_level"}},
                {"name": "Naive", "params": {}},
                {"name": "SeasonalNaive", "params": {}},
                {"name": "Average", "params": {}},
                {"name": "SeasonalAverage", "params": {}},
                {"name": "Zero", "params": {}},
                {
                    "name": "ARIMA",
                    "params": {
                        "order": [(1, 0, 0), (1, 1, 1), (2, 1, 0), (0, 1, 1), (2, 1, 2)],
                        "seasonal_order": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
                        "method": ["CSS-ML", "CSS", "ML"]
                    }
                }
                # {
                #     "name": "AutoARIMA",
                #     "params": {
                #         "max_p": {"start": 3, "end": 8},
                #         "max_q": {"start": 3, "end": 8},
                #         "max_P": {"start": 1, "end": 4},
                #         "max_Q": {"start": 1, "end": 4},
                #         "max_d": {"start": 0, "end": 2},
                #         "max_D": {"start": 0, "end": 2},
                #         "start_p": {"start": 1, "end": 3},
                #         "seasonal": ["true", "false"]
                #     }
                # },
                # {
                #     "name": "ETS",
                #     "params": {
                #         "model": ["AAA", "ANA", "MAM", "MNM"],
                #         "damped": ["true", "false"]
                #     }
                # },
                # {"name": "AutoETS", "params": {}},
                # {
                #     "name": "AutoCES",
                #     "params": {
                #         "model": ["Z", "N", "S", "P", "F"]
                #     }
                # },
                # {
                #     "name": "Theta",
                #     "params": {
                #         "decomposition_type": ["multiplicative", "additive"]
                #     }
                # },
                # {"name": "DynamicOptimizedTheta", "params": {}},
                # {"name": "ADIDA", "params": {}},
                # {
                #     "name": "Croston",
                #     "params": {
                #         "variant": ["SBA", "classic", "optimized"]
                #     }
                # },
                # {"name": "CrostonSBA", "params": {}},
                # {"name": "IMAPA", "params": {}},
                # {"name": "DeepAR", "params": {}},
                # {
                #     "name": "TemporalFusionTransformer",
                #     "params": {
                #         "hidden_dim": [32, 64, 128],
                #         "num_heads": [1, 2, 4],
                #         "dropout_rate": {"start": 0.1, "end": 0.3},
                #         "batch_size": [32, 64, 128],
                #         "lr": {"start": 0.0001, "end": 0.01, "log": "true"}
                #     }
                # },
                # {
                #     "name": "TemporalFusionTransformer",
                #     "params": {
                #         "hidden_dim": [32],
                #         "num_heads": [1, 2, 4],
                #         "dropout_rate": {"start": 0.1, "end": 0.3},
                #         "batch_size": [32],
                #         "lr": {"start": 0.0001, "end": 0.01, "log": "true"}
                #     }
                # },
                # {
                #     "name": "PatchTST",
                #     "params": {
                #         "patch_len": [8, 16, 24],
                #         "stride": [4, 8, 16],
                #         "d_model": [32, 64, 128, 512],
                #         "nhead": [2, 4, 8],
                #         "num_encoder_layers": {"start": 2, "end": 4},
                #         "scaling": ["mean", "std", "None"],
                #         "batch_size": [32, 64, 128],
                #         "lr": {"start": 0.0001, "end": 0.01, "log": "true"}
                #     }
                # },
                # {
                #     "name": "TiDE",
                #     "params": {
                #         "encoder_hidden_dim": [64, 128, 256],
                #         "decoder_hidden_dim": [64, 128, 256],
                #         "temporal_hidden_dim": [64, 128],
                #         "distr_hidden_dim": [64, 128],
                #         "num_layers_encoder": {"start": 1, "end": 4},
                #         "num_layers_decoder": {"start": 1, "end": 4},
                #         "dropout_rate": {"start": 0.1, "end": 0.3},
                #         "layer_norm": ["true", "false"],
                #         "batch_size": [128, 256, 512],
                #         "lr": {"start": 0.0001, "end": 0.01}
                #     }
                # },
                # {
                #     "name": "DLinear",
                #     "params": {}
                # },
                # {
                #     "name": "WaveNet",
                #     "params": {
                #         "num_bins": {"start": 512, "end": 1024},
                #         "use_log_scale_feature": ["true", "false"],
                #         "negative_data": ["true", "false"],
                #         "batch_size": [32, 64, 128],
                #         "lr": {"start": 0.0001, "end": 0.01, "log": "true"},
                #         "weight_decay": {"start": 1e-08, "end": 0.0001, "log": "true"}
                #     }
                # },
                # {
                #     "name": "SimpleFeedForward",
                #     "params": {
                #         "batch_normalization": ["true", "false"],
                #         "mean_scaling": ["true", "false"],
                #         "batch_size": [32, 64, 128],
                #         "lr": {"start": 0.0001, "end": 0.01, "log": "true"}
                #     }
                # },
                # {"name": "Chronos", "params": {}},
                # {"name": "DirectTabular", "params": {}},
                # {"name": "RecursiveTabular", "params": {}},
                # {"name": "PerStepTabular", "params": {}}
        ]

        tune_kwargs_payload = {
            "num_trials": 10,       # 尝试的组合总数
            "searcher": "random",   # 随机搜索
            "scheduler": "local"
        }


        params = {
            "task_id": task_id,
            "target": "value",
            "prediction_length": 5,
            "freq": "D",
            "time_limit": 60,        # 轻量配置，方便并发压测
            "use_tsfresh": False,
            "enable_ensemble": True,
            "hyperparameters": json.dumps(hyperparameters_payload),
            "hyperparameter_tune_kwargs": json.dumps(tune_kwargs_payload)
        }

        print(f"\n[{index}] 🚀 提交训练任务: {task_id}")
        start_time = time.time()

        try:
            with open(self.input_csv, "rb") as f:
                resp = requests.post(
                    f"{self.base_url}/train",
                    files={"file": f},
                    data=params,
                    timeout=CONFIG["TIMEOUT"],
                )
        except Exception as e:
            print(f"[{index}] ❌ 请求异常: {e}")
            return {"task_id": task_id, "index": index, "success": False, "error": str(e)}

        elapsed = time.time() - start_time

        if resp.status_code != 200 or "application/zip" not in resp.headers.get("content-type", ""):
            print(f"[{index}] ❌ 训练失败，状态码: {resp.status_code}，耗时: {elapsed:.1f}s")
            print(f"[{index}]    响应: {resp.text[:300]}")
            return {"task_id": task_id, "index": index, "success": False, "error": resp.text}

        # 保存 zip
        zip_filename = f"{task_id}_model_pack.zip"
        zip_path = os.path.join(self.output_dir, zip_filename)
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        print(f"[{index}] ✅ 训练完成，耗时: {elapsed:.1f}s，zip 已保存: {zip_path}")

        # 拉取并保存日志
        log_lines = []
        try:
            log_resp = requests.get(
                f"{self.base_url}/task_logs",
                params={"task_id": task_id},
                timeout=30,
            )
            log_lines = log_resp.json().get("data", [])
            log_path = os.path.join(self.output_dir, f"{task_id}.log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(log_lines))
            print(f"[{index}] 📝 日志已保存: {log_path} ({len(log_lines)} 行)")
        except Exception as e:
            print(f"[{index}] ⚠️ 拉取日志失败: {e}")

        return {
            "task_id": task_id,
            "index": index,
            "success": True,
            "elapsed": elapsed,
            "zip_path": zip_path,
            "log_lines": log_lines,
        }

    # -------------------- 单任务入口 --------------------
    def run_train(self):
        """单任务训练测试"""
        if not os.path.exists(self.input_csv):
            logger.error("❌ 找不到输入 CSV 文件: %s", self.input_csv)
            return
        result = self._run_single_train(0)
        self._print_summary([result])

    # -------------------- 并发入口 --------------------
    def run_concurrent_train(self, num_jobs: int = None):
        """并发训练测试，完成后打印日志隔离验证报告"""
        if not os.path.exists(self.input_csv):
            logger.error("❌ 找不到输入 CSV 文件: %s", self.input_csv)
            return

        num_jobs = num_jobs or CONFIG["NUM_JOBS"]
        print(f"\n{'='*60}")
        print(f" 并发训练测试：{num_jobs} 个任务同时提交")
        print(f"{'='*60}\n")

        results = []
        with ThreadPoolExecutor(max_workers=num_jobs) as executor:
            futures = {executor.submit(self._run_single_train, i): i for i in range(num_jobs)}
            for fut in as_completed(futures):
                results.append(fut.result())

        # 按 index 排序便于对比
        results.sort(key=lambda x: x["index"])
        self._print_summary(results)
        self._verify_log_isolation(results)

    # -------------------- 报告 --------------------
    def _print_summary(self, results: list):
        print(f"\n{'='*60}")
        print(" 训练结果摘要")
        print(f"{'='*60}")
        for r in results:
            status = "✅ SUCCESS" if r["success"] else "❌ FAILED"
            elapsed = f"{r.get('elapsed', 0):.1f}s" if r["success"] else "-"
            print(f"  [{r['index']}] {status} | task_id={r['task_id']} | 耗时={elapsed}")
            if not r["success"]:
                print(f"       error: {r.get('error', '')[:100]}")
        print()

    def _verify_log_isolation(self, results: list):
        """检查各任务日志是否互相串扰"""
        print(f"{'='*60}")
        print(" 日志隔离验证")
        print(f"{'='*60}")

        successful = [r for r in results if r["success"]]
        if len(successful) < 2:
            print("  ⚠️  成功任务数不足 2，无法做交叉验证\n")
            return

        all_clean = True
        for r in successful:
            own_id = r["task_id"]
            own_lines = r.get("log_lines", [])
            # 检查自己的日志中是否混入了其他任务的 task_id
            contaminated_by = []
            for other in successful:
                if other["task_id"] == own_id:
                    continue
                other_id = other["task_id"]
                leaked = [line for line in own_lines if other_id in line]
                if leaked:
                    contaminated_by.append((other_id, leaked[:2]))  # 只展示前 2 行

            if contaminated_by:
                all_clean = False
                print(f"  [{r['index']}] ❌ 日志串扰！task={own_id} 的日志中发现其他任务内容:")
                for other_id, lines in contaminated_by:
                    print(f"       来自 {other_id}:")
                    for line in lines:
                        print(f"         > {line}")
            else:
                print(f"  [{r['index']}] ✅ 日志干净，共 {len(own_lines)} 行，无串扰")

        print()
        if all_clean:
            print("  🎉 所有任务日志完全隔离，无串扰！")
        else:
            print("  ⚠️  存在日志串扰，请检查 logger handler 清理逻辑")
        print()


if __name__ == "__main__":
    tester = TrainTester()
    if not tester.check_health():
        logger.error("服务不可用，终止测试")
    else:
        # 单任务测试
        # tester.run_train()

        # 并发测试（默认 3 个任务）
        tester.run_concurrent_train(num_jobs=CONFIG["NUM_JOBS"])
