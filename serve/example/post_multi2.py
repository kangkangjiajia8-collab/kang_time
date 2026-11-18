import requests
import os
import time
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 =================
CONFIG = {
    "BASE_URL": "http://localhost:8000/api",
    "INPUT_CSV": "2_selected_featuresv3.csv",
    "OUTPUT_DIR": "test_results_parallel",
    "MAX_WORKERS": 1,
    "REQUEST_TIMEOUT": None,   # /train 可能很久，保持 None
    "POLL_INTERVAL": 3,        # 实时轮询间隔（秒）
}

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AutoApexParallelTester:
    def __init__(self):
        self.base_url = CONFIG["BASE_URL"]
        self.input_csv = CONFIG["INPUT_CSV"]
        self.output_dir = CONFIG["OUTPUT_DIR"]
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_file(self, response, filename):
        path = os.path.join(self.output_dir, filename)
        with open(path, "wb") as f:
            f.write(response.content)
        return path

    def _save_json(self, payload, filename):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def check_health(self):
        try:
            resp = requests.get(f"{self.base_url.replace('/api', '')}/docs", timeout=10)
            if resp.status_code == 200:
                logger.info("✅ 服务连接正常")
                return True
            logger.error(f"❌ docs 返回异常状态码: {resp.status_code}")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("❌ 无法连接到服务，请确保 server 正在运行")
            return False
        except Exception as e:
            logger.error(f"❌ 健康检查异常: {e}")
            return False

    def fetch_status(self, task_id):
        try:
            resp = requests.get(
                f"{self.base_url}/task_status",
                params={"task_id": task_id},
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json().get("data", {})
        except Exception as e:
            logger.warning(f"[{task_id}] fetch_status failed: {e}")
        return {}

    def fetch_logs(self, task_id):
        try:
            resp = requests.get(
                f"{self.base_url}/task_logs",
                params={"task_id": task_id},
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception as e:
            logger.warning(f"[{task_id}] fetch_logs failed: {e}")
        return []

    def _monitor_task(self, task_id, stop_event):
        """
        实时监控单个任务：
        - 打印状态变化
        - 打印新增日志
        - 直到 stop_event 被置位后退出
        """
        last_status = None
        last_message = None
        printed_log_count = 0

        while not stop_event.is_set():
            try:
                # 1. 状态轮询
                state = self.fetch_status(task_id)
                status = state.get("status")
                message = state.get("message")

                if status != last_status or message != last_message:
                    logger.info(f"[{task_id}] 状态: {status} | {message}")
                    last_status = status
                    last_message = message

                # 2. 日志轮询，只打印新增部分
                logs = self.fetch_logs(task_id)
                if logs and len(logs) > printed_log_count:
                    new_logs = logs[printed_log_count:]
                    for line in new_logs:
                        logger.info(f"[{task_id}] {line}")
                    printed_log_count = len(logs)

                # 3. 如果任务已经结束，也可以退出监控
                if status in {"SUCCESS", "FAILED"}:
                    # 最后再补抓一次日志，避免末尾遗漏
                    logs = self.fetch_logs(task_id)
                    if logs and len(logs) > printed_log_count:
                        new_logs = logs[printed_log_count:]
                        for line in new_logs:
                            logger.info(f"[{task_id}] {line}")
                        printed_log_count = len(logs)
                    break

            except Exception as e:
                logger.warning(f"[{task_id}] monitor exception: {e}")

            stop_event.wait(CONFIG["POLL_INTERVAL"])

    def submit_train_job(self, task_suffix, hyperparameters_payload, enable_shap=False):
        # task_id 尽量避免碰撞
        task_id = f"train_{task_suffix}_{time.time_ns()}"

        params = {
            "task_id": task_id,
            "target": "value",
            "prediction_length": 192,
            "freq": "15T",
            "time_limit": 6000,
            "use_tsfresh": False,
            "enable_ensemble": False,
            "enable_shap": str(enable_shap).lower(),
            "hyperparameters": json.dumps(hyperparameters_payload, ensure_ascii=False),
            "ensemble_types": "weighted"
        }

        logger.info(f"🚀 提交训练任务: {task_id}")

        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=self._monitor_task,
            args=(task_id, stop_event),
            daemon=True
        )
        monitor_thread.start()

        try:
            with open(self.input_csv, "rb") as f:
                resp = requests.post(
                    f"{self.base_url}/train",
                    files={"file": (os.path.basename(self.input_csv), f, "text/csv")},
                    data=params,
                    timeout=CONFIG["REQUEST_TIMEOUT"]
                )

            if resp.status_code == 200:
                save_name = f"{task_id}_model_pack.zip"
                save_path = self._save_file(resp, save_name)
                logger.info(f"✅ {task_id} 训练完成，模型包已保存: {save_path}")

                # 保存最终状态
                final_state = self.fetch_status(task_id)
                self._save_json(final_state, f"{task_id}_final_status.json")

                # 保存完整日志
                final_logs = self.fetch_logs(task_id)
                if final_logs:
                    self._save_json(final_logs, f"{task_id}_full_logs.json")

                return {
                    "task_id": task_id,
                    "success": True,
                    "zip_path": save_path
                }
            else:
                error_text = resp.text
                logger.error(f"❌ {task_id} 训练失败: {error_text}")

                final_state = self.fetch_status(task_id)
                self._save_json(final_state, f"{task_id}_final_status.json")

                final_logs = self.fetch_logs(task_id)
                if final_logs:
                    self._save_json(final_logs, f"{task_id}_full_logs.json")

                return {
                    "task_id": task_id,
                    "success": False,
                    "error": error_text
                }

        except Exception as e:
            logger.error(f"❌ {task_id} 请求异常: {e}")

            final_state = self.fetch_status(task_id)
            self._save_json(final_state, f"{task_id}_final_status.json")

            final_logs = self.fetch_logs(task_id)
            if final_logs:
                self._save_json(final_logs, f"{task_id}_full_logs.json")

            return {
                "task_id": task_id,
                "success": False,
                "error": str(e)
            }

        finally:
            stop_event.set()
            monitor_thread.join(timeout=2)

    def run_parallel_training(self):
        job_configs = [
            {
                "task_suffix": "job1",
                "hyperparameters_payload": [
                    {"name": "SeasonalNaive", "params": {}},
                    {"name": "Average", "params": {}},
                    {"name": "SeasonalAverage", "params": {}},
                    {"name": "Zero", "params": {}},
                    {"name": "ARIMA", "params": {}},
                    {"name": "AutoARIMA", "params": {}},
                    {"name": "ETS", "params": {}},
                    {"name": "EWMA", "params": {}},
                    {"name": "AutoETS", "params": {}},
                    {"name": "AutoCES", "params": {}},
                    {"name": "Theta", "params": {}},
                    {"name": "DynamicOptimizedTheta", "params": {}},
                    {"name": "ADIDA", "params": {}},
                    {"name": "Croston", "params": {}},
                    {"name": "CrostonSBA", "params": {}},
                    {"name": "IMAPA", "params": {}},
                    {"name": "DeepAR", "params": {}},
                    {"name": "TemporalFusionTransformer", "params": {}},
                    {"name": "PatchTST", "params": {}},
                    {"name": "TiDE", "params": {}},
                    {"name": "DLinear", "params": {}},
                    {"name": "WaveNet", "params": {}},
                    {"name": "SimpleFeedForward", "params": {}},
                    {"name": "Chronos", "params": {}},
                    {"name": "DirectTabular", "params": {}},
                    {"name": "RecursiveTabular", "params": {}},
                    {"name": "PerStepTabular", "params": {}},
                    {"name": "MovingAverageInterpolation", "params": {}}
                ],
                "enable_shap": False
            }
        ]

        results = []
        with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
            future_to_job = {
                executor.submit(
                    self.submit_train_job,
                    job["task_suffix"],
                    job["hyperparameters_payload"],
                    job.get("enable_shap", False)
                ): job
                for job in job_configs
            }

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"🎯 已完成任务: {result['task_id']}")
                except Exception as e:
                    logger.error(f"❌ 任务执行异常: {job['task_suffix']} -> {e}")

        summary_path = self._save_json(results, "parallel_train_summary.json")
        logger.info(f"📦 汇总结果已保存: {summary_path}")
        return results


if __name__ == "__main__":
    tester = AutoApexParallelTester()

    if tester.check_health():
        results = tester.run_parallel_training()

        logger.info("=== 并发训练结果汇总 ===")
        for item in results:
            logger.info(item)

    logger.info(f"✨ 测试结束，请检查目录: {tester.output_dir}")