import os
import shutil
import time
import uuid
import json
import logging
import threading
import pandas as pd
from fastapi import UploadFile
from configs.config import GLOBAL_LOGGER, LOG_DIR, BeijingFormatter, TEMP_DIR, TASK_STATE_DIR
from autoapex.timeseries import TimeSeriesDataFrame

# ================= 全局任务状态存储 =================
TASK_STORE = {}
_TASK_STORE_LOCK = threading.Lock()

def _task_state_path(task_id: str) -> str:
    return os.path.join(TASK_STATE_DIR, f"{task_id}.json")

def update_task_state(task_id, status, message=None, data=None):
    with _TASK_STORE_LOCK:
        if task_id not in TASK_STORE:
            TASK_STORE[task_id] = {}
        TASK_STORE[task_id]["status"] = status
        if message:
            TASK_STORE[task_id]["message"] = message
        if data:
            if "data" not in TASK_STORE[task_id]:
                TASK_STORE[task_id]["data"] = {}
            TASK_STORE[task_id]["data"].update(data)
        # 持久化到文件，便于子进程与主进程共享（如特征生成在子进程中更新状态）
        try:
            path = _task_state_path(task_id)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(TASK_STORE[task_id], f, ensure_ascii=False, indent=0)
        except Exception as e:
            GLOBAL_LOGGER.warning(f"Failed to persist task state for {task_id}: {e}")

def get_task_state(task_id):
    with _TASK_STORE_LOCK:
        # 优先从文件读取，保证子进程的更新对主进程可见
        path = _task_state_path(task_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                TASK_STORE[task_id] = state
                return state.copy()
            except Exception as e:
                GLOBAL_LOGGER.warning(f"Failed to read task state file for {task_id}: {e}")
        return TASK_STORE.get(task_id)


# ================= 并发长任务：按 task_id 注册进程，支持心跳检测卡住 =================
# task_id -> {"process": Process, "service": str}
RUNNING_PROCESSES = {}
_RUNNING_PROCESSES_LOCK = threading.Lock()
HEARTBEAT_TIMEOUT_SEC = 600  # 10 分钟无心跳则标记 STUCK
_MONITOR_STARTED = False

def register_running_process(task_id: str, process, service: str = "training") -> None:
    with _RUNNING_PROCESSES_LOCK:
        RUNNING_PROCESSES[task_id] = {"process": process, "service": service}


def unregister_running_process(task_id: str) -> None:
    with _RUNNING_PROCESSES_LOCK:
        RUNNING_PROCESSES.pop(task_id, None)


def _heartbeat_monitor_loop():
    """后台线程：检查各任务心跳，超时则标记 STUCK 并终止进程"""
    import time as _time
    while True:
        _time.sleep(60)
        now = _time.time()
        with _RUNNING_PROCESSES_LOCK:
            snapshot = list(RUNNING_PROCESSES.items())
        to_remove = []
        for task_id, info in snapshot:
            proc = info.get("process")
            if not proc or not getattr(proc, "is_alive", lambda: False)():
                to_remove.append(task_id)
                continue
            state = get_task_state(task_id)  # 不持 _RUNNING_PROCESSES_LOCK 调用，避免死锁
            if not state:
                continue
            data = state.get("data") or {}
            last = data.get("last_heartbeat")
            if last is not None and (now - last) > HEARTBEAT_TIMEOUT_SEC:
                try:
                    proc.terminate()
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=2)
                except Exception as e:
                    GLOBAL_LOGGER.warning(f"Stuck monitor kill failed for {task_id}: {e}")
                update_task_state(task_id, "STUCK", "任务无响应超时，已终止（可能因 CPU/资源占满导致卡住）")
                GLOBAL_LOGGER.info(f"已标记并终止卡住任务: task_id={task_id}")
                to_remove.append(task_id)
        with _RUNNING_PROCESSES_LOCK:
            for task_id in to_remove:
                RUNNING_PROCESSES.pop(task_id, None)


def start_heartbeat_monitor():
    global _MONITOR_STARTED
    with _RUNNING_PROCESSES_LOCK:
        if _MONITOR_STARTED:
            return
        _MONITOR_STARTED = True
    t = threading.Thread(target=_heartbeat_monitor_loop, daemon=True)
    t.start()
    GLOBAL_LOGGER.info("Heartbeat monitor started (stuck detection).")


# ================= 单例长任务（可选）：新请求时强制终止上一次 =================
# service_name -> {"process": Process, "task_id": str, "file_path": str | None}
RUNNING_SERVICE_JOBS = {}
_RUNNING_JOBS_LOCK = threading.Lock()

def cancel_running_service(service_name: str) -> None:
    """若该服务有正在运行的任务，则强制终止并标记为 CANCELLED，并清理其临时文件。"""
    with _RUNNING_JOBS_LOCK:
        current = RUNNING_SERVICE_JOBS.get(service_name)
        if not current:
            return
        proc = current.get("process")
        old_task_id = current.get("task_id")
        file_path = current.get("file_path")
        if proc and getattr(proc, "is_alive", lambda: False)():
            try:
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
            except Exception as e:
                GLOBAL_LOGGER.warning(f"Terminate previous {service_name} task failed: {e}")
            update_task_state(old_task_id, "CANCELLED", "被新请求终止")
            GLOBAL_LOGGER.info(f"已终止上一任务: service={service_name}, task_id={old_task_id}")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                GLOBAL_LOGGER.info(f"已清理被终止任务的临时文件: {file_path}")
            except OSError as e:
                GLOBAL_LOGGER.warning(f"清理临时文件失败: {file_path}, {e}")
        try:
            del RUNNING_SERVICE_JOBS[service_name]
        except KeyError:
            pass


def register_running_service(service_name: str, process, task_id: str, file_path: str = None) -> None:
    with _RUNNING_JOBS_LOCK:
        RUNNING_SERVICE_JOBS[service_name] = {"process": process, "task_id": task_id, "file_path": file_path}


def unregister_running_service(service_name: str, task_id: str) -> None:
    """任务正常结束时由调度线程调用，仅当当前注册的任务仍是本 task_id 时才删除。"""
    with _RUNNING_JOBS_LOCK:
        current = RUNNING_SERVICE_JOBS.get(service_name)
        if current and current.get("task_id") == task_id:
            try:
                del RUNNING_SERVICE_JOBS[service_name]
            except KeyError:
                pass


# ================= 辅助函数 =================

def get_or_create_task_id(task_id: str = None) -> str:
    """如果未提供task_id，则基于时间戳生成"""
    if not task_id:
        return f"{int(time.time())}_{uuid.uuid4().hex[:4]}"
    return task_id

def save_upload_file(file: UploadFile, service_name: str, task_id: str = None) -> tuple[str, str]:
    """
    通用上传文件保存逻辑
    路径结构: outputs/temp/{service_name}/{task_id}/{filename}
    返回: (保存后的绝对路径, 最终的task_id)
    """
    # 1. 确定 Task ID
    final_task_id = get_or_create_task_id(task_id)
    
    # 2. 构建目录: outputs/temp/{service_name}/{task_id}
    # TEMP_DIR 在 config.py 中定义为 outputs/temp
    target_dir = os.path.join(TEMP_DIR, service_name, final_task_id)
    os.makedirs(target_dir, exist_ok=True)
    
    # 3. 确定文件路径
    file_path = os.path.join(target_dir, file.filename)
    
    # 4. 保存文件
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        GLOBAL_LOGGER.info(f"File saved to temp: {file_path}")
    except Exception as e:
        GLOBAL_LOGGER.error(f"Failed to save upload file: {e}")
        raise e
        
    return file_path, final_task_id

def prepare_output_dir(base_dir: str, task_id: str):
    """确保 outputs/{base}/{task_id} 存在"""
    path = os.path.join(base_dir, task_id)
    os.makedirs(path, exist_ok=True)
    return path

def cleanup_files(paths: list):
    """清理临时文件"""
    for path in paths:
        try:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                GLOBAL_LOGGER.info(f"Cleaned up: {path}")
                
                # 尝试清理空的父目录 (可选优化)
                parent_dir = os.path.dirname(path)
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
                    
        except Exception as e:
            GLOBAL_LOGGER.error(f"Error cleaning up {path}: {e}")

# def setup_task_logger(task_id: str):
#     """为每个任务创建独立日志，并桥接 tsfresh 日志"""
#     log_file = os.path.join(LOG_DIR, f"{task_id}.log")
#     logger = logging.getLogger(f"task_{task_id}")
#     logger.setLevel(logging.INFO)
    
#     file_handler = None
#     if not logger.handlers:
#         file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8') 
#         formatter = BeijingFormatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)
#     else:
#         file_handler = logger.handlers[0]

#     tsfresh_logger = logging.getLogger("tsfresh_progress")
#     tsfresh_logger.setLevel(logging.INFO)
#     tsfresh_logger.handlers = [] 
#     tsfresh_logger.addHandler(file_handler)
    
#     ag_logger = logging.getLogger("autoapex")
#     ag_logger.setLevel(logging.INFO)
#     if file_handler not in ag_logger.handlers:
#         ag_logger.addHandler(file_handler)
    
#     return logger


def setup_task_logger(task_id: str, capture_autoapex: bool = True):
    """为每个任务创建独立日志，并桥接 tsfresh 日志。

    capture_autoapex: 若为 True，将 autoapex 根 logger 的日志也写入本任务日志文件（适合训练任务）；
                       若为 False，不捕获 autoapex 日志，仅保留本任务与 tsfresh 的日志（适合特征生成等非训练任务，避免 time_limit、训练进度等混入）。
    """
    log_file = os.path.join(LOG_DIR, f"{task_id}.log")
    logger = logging.getLogger(f"task_{task_id}")
    logger.setLevel(logging.INFO)

    file_handler = None
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        formatter = BeijingFormatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        file_handler = logger.handlers[0]

    tsfresh_logger = logging.getLogger("tsfresh_progress")
    tsfresh_logger.setLevel(logging.INFO)
    tsfresh_logger.handlers = []
    tsfresh_logger.addHandler(file_handler)

    if capture_autoapex:
        ag_logger = logging.getLogger("autoapex")
        ag_logger.setLevel(logging.INFO)
        if file_handler not in ag_logger.handlers:
            ag_logger.addHandler(file_handler)

    return logger

def prepare_tsdf(file_path: str, id_col="item_id", time_col="timestamp", target_col="value", default_id="default_item") -> TimeSeriesDataFrame:
    """读取 CSV 并转换为 TimeSeriesDataFrame"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    if 'date' in df.columns and time_col not in df.columns:
        df.rename(columns={'date': time_col}, inplace=True)
    if 'time' in df.columns and time_col not in df.columns:
        df.rename(columns={'time': time_col}, inplace=True)
    
    if id_col not in df.columns:
        df[id_col] = default_id

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=[id_col, time_col])
    
    return TimeSeriesDataFrame.from_data_frame(df, id_column=id_col, timestamp_column=time_col)