import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


CONFIG = {
    "BASE_URL": "http://localhost:8000/api",
    "INPUT_CSV": "just.csv",
    # 请把这里改成一个真实存在的模型 zip 路径
    "MODEL_ZIP": "test_results_multi_train/example_model.zip",
    "OUTPUT_DIR": "test_results_multi_infer",
    "NUM_JOBS": 3,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultiInferTester:
    def __init__(self):
        self.base_url = CONFIG["BASE_URL"]
        self.input_csv = CONFIG["INPUT_CSV"]
        self.model_zip = CONFIG["MODEL_ZIP"]
        self.output_dir = CONFIG["OUTPUT_DIR"]
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_binary(self, content: bytes, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "wb") as f:
            f.write(content)
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

    def _run_one_infer(self, index: int) -> None:
        task_id = f"infer_multi_{int(time.time())}_{index}"
        logger.info("🚀 [%s] 提交推理任务: %s", index, task_id)

        if not os.path.exists(self.model_zip):
            logger.error("❌ 模型包不存在: %s", self.model_zip)
            return

        files = None
        try:
            files = [
                ("model_zip", (os.path.basename(self.model_zip), open(self.model_zip, "rb"), "application/zip")),
                ("data_csv", (os.path.basename(self.input_csv), open(self.input_csv, "rb"), "text/csv")),
            ]
            data = {
                "task_id": task_id,
                # 如果你知道具体最佳模型名，可以在这里填；为空时由服务端选择默认模型
                "best_model": "",
            }
            resp = requests.post(f"{self.base_url}/inference", files=files, data=data, timeout=600)
        except Exception as e:
            logger.error("❌ [%s] 推理请求异常: %s", task_id, e)
            return
        finally:
            if files:
                for _, f_tuple in files:
                    try:
                        f_tuple[1].close()
                    except Exception:
                        pass

        if resp.status_code == 200:
            # 保存预测结果
            pred_path = self._save_binary(resp.content, f"{task_id}_predictions.csv")
            logger.info("✅ [%s] 推理成功，预测结果已保存: %s", task_id, pred_path)
            
            # 获取并保存日志文件
            try:
                log_resp = requests.get(f"{self.base_url}/inference/{task_id}/log", timeout=30)
                if log_resp.status_code == 200:
                    log_path = self._save_binary(log_resp.content, f"{task_id}.log")
                    logger.info("📝 [%s] 日志文件已保存: %s", task_id, log_path)
            except Exception as e:
                logger.warning("⚠️  [%s] 获取日志失败: %s", task_id, e)
            
            # 获取并保存元数据
            try:
                meta_resp = requests.get(f"{self.base_url}/inference/{task_id}/metadata", timeout=30)
                if meta_resp.status_code == 200:
                    meta_path = os.path.join(self.output_dir, f"{task_id}_metadata.json")
                    with open(meta_path, "w", encoding="utf-8") as f:
                        f.write(meta_resp.text)
                    logger.info("📊 [%s] 元数据已保存: %s", task_id, meta_path)
            except Exception as e:
                logger.warning("⚠️  [%s] 获取元数据失败: %s", task_id, e)
        else:
            logger.error("❌ [%s] 推理失败: %s", task_id, resp.text)

    def run_multi_infer(self, num_jobs: int = 3):
        if not os.path.exists(self.input_csv):
            logger.error("❌ 找不到输入 CSV: %s", self.input_csv)
            return
        if not os.path.exists(self.model_zip):
            logger.error("❌ 找不到模型包: %s", self.model_zip)
            return

        logger.info("=== 并发推理测试：%s 个任务 ===", num_jobs)

        with ThreadPoolExecutor(max_workers=num_jobs) as exe:
            futures = [exe.submit(self._run_one_infer, i) for i in range(num_jobs)]
            for fut in as_completed(futures):
                _ = fut.result()

        logger.info("✨ 并发推理测试完成，结果目录: %s", self.output_dir)


if __name__ == "__main__":
    tester = MultiInferTester()
    if tester.check_health():
        logger.warning(
            "请先确认 CONFIG['MODEL_ZIP'] 指向一个真实存在的模型 zip（可用训练接口导出的包），当前配置为: %s",
            CONFIG["MODEL_ZIP"],
        )
        tester.run_multi_infer(num_jobs=CONFIG["NUM_JOBS"])
    else:
        logger.error("服务不可用，终止并发推理测试")

