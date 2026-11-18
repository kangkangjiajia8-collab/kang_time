import os
import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


CONFIG = {
    "BASE_URL": "http://localhost:8000/api",
    "INPUT_CSV": "just.csv",
    "OUTPUT_DIR": "test_results_multi_features",
    "TIMEOUT": 900,
    "POLL_INTERVAL": 5,
    "NUM_JOBS": 3,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultiFeatureTester:
    def __init__(self):
        self.base_url = CONFIG["BASE_URL"]
        self.input_csv = CONFIG["INPUT_CSV"]
        self.output_dir = CONFIG["OUTPUT_DIR"]
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_binary(self, content: bytes, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def _save_text(self, lines, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(lines, (list, tuple)):
                for line in lines:
                    f.write(str(line) + "\n")
            else:
                f.write(str(lines))
        return path

    def check_health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url.replace('/api', '')}/docs", timeout=5)
            resp.raise_for_status()
            logger.info("✅ 服务健康检查通过")
            return True
        except Exception as e:
            logger.error("❌ 无法连接到服务，请确认 FastAPI 已在 8000 端口启动: %s", e)
            return False

    def _build_custom_params(self) -> str:
        payload = [
            {"name": "time_reversal_asymmetry_statistic", "params": {"lag": {"start": 1, "end": 5}}},
            {"name": "c3", "params": {"lag": {"start": 1, "end": 5}}},
            {"name": "cid_ce", "params": {"normalize": ["false"]}},
        ]
        return json.dumps(payload, ensure_ascii=False)

    def _start_one_feature_job(self, index: int) -> str | None:
        task_id = f"feat_multi_{int(time.time())}_{index}"
        logger.info("🚀 [%s] 提交特征生成任务: %s", index, task_id)
        files = None
        try:
            files = {
                "file": (os.path.basename(self.input_csv), open(self.input_csv, "rb"), "text/csv"),
            }
            data = {
                "task_id": task_id,
                "max_timeshift": 29,
                "min_timeshift": 29,
                "custom_params": self._build_custom_params(),
            }
            resp = requests.post(
                f"{self.base_url}/generate_all_features",
                files=files,
                data=data,
                timeout=60,
            )
        except Exception as e:
            logger.error("❌ [%s] 提交失败: %s", task_id, e)
            return None
        finally:
            if files:
                try:
                    files["file"][1].close()
                except Exception:
                    pass

        if resp.status_code != 200:
            logger.error("❌ [%s] 提交失败: %s", task_id, resp.text)
            return None

        logger.info("✅ [%s] 已提交特征任务，响应: %s", task_id, resp.json())
        return task_id

    def _poll_and_fetch(self, task_id: str) -> None:
        start = time.time()
        logger.info("⏳ [%s] 开始轮询特征任务状态...", task_id)
        finished = False

        while time.time() - start < CONFIG["TIMEOUT"]:
            try:
                resp = requests.get(f"{self.base_url}/task_status", params={"task_id": task_id}, timeout=10)
                data = resp.json().get("data") or {}
                status = data.get("status")
                msg = data.get("message")
                logger.info("📡 [%s] status=%s, message=%s", task_id, status, msg)
                if status in ("SUCCESS", "FAILED", "STUCK", "CANCELLED"):
                    finished = True
                    break
            except Exception as e:
                logger.warning("⚠️ [%s] 轮询异常: %s", task_id, e)
            time.sleep(CONFIG["POLL_INTERVAL"])

        if not finished:
            logger.error("❌ [%s] 轮询超时，放弃下载结果", task_id)
            return

        try:
            resp = requests.post(
                f"{self.base_url}/select_features_and_download",
                data={"task_id": task_id, "fdr": 0.05},
                timeout=120,
            )
            if resp.status_code == 200:
                path = self._save_binary(resp.content, f"{task_id}_selected_features.csv")
                logger.info("✅ [%s] 特征结果已保存: %s", task_id, path)
            else:
                logger.info("ℹ️ [%s] 无法下载特征文件: %s", task_id, resp.text)
        except Exception as e:
            logger.error("❌ [%s] 下载特征文件异常: %s", task_id, e)

        try:
            log_resp = requests.get(f"{self.base_url}/task_logs", params={"task_id": task_id}, timeout=30)
            lines = log_resp.json().get("data", [])
            log_path = self._save_text(lines, f"{task_id}.log.txt")
            logger.info("📝 [%s] 日志已保存: %s", task_id, log_path)
        except Exception as e:
            logger.warning("⚠️ [%s] 获取日志失败: %s", task_id, e)

    def run_multi_features(self, num_jobs: int = 3):
        if not os.path.exists(self.input_csv):
            logger.error("❌ 找不到输入文件: %s", self.input_csv)
            return

        logger.info("=== 并发特征生成测试：%s 个任务 ===", num_jobs)

        task_ids: list[str] = []
        with ThreadPoolExecutor(max_workers=num_jobs) as exe:
            futures = [exe.submit(self._start_one_feature_job, i) for i in range(num_jobs)]
            for fut in as_completed(futures):
                tid = fut.result()
                if tid:
                    task_ids.append(tid)

        if not task_ids:
            logger.error("❌ 无成功提交的特征任务")
            return

        logger.info("✅ 成功提交的特征任务: %s", task_ids)

        with ThreadPoolExecutor(max_workers=len(task_ids)) as exe:
            futures = [exe.submit(self._poll_and_fetch, tid) for tid in task_ids]
            for fut in as_completed(futures):
                _ = fut.result()

        logger.info("✨ 并发特征测试完成，结果目录: %s", self.output_dir)


if __name__ == "__main__":
    tester = MultiFeatureTester()
    if tester.check_health():
        tester.run_multi_features(num_jobs=CONFIG["NUM_JOBS"])
    else:
        logger.error("服务不可用，终止并发特征测试")

