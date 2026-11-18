import time
import os
import json
import shutil
from multiprocessing import Process
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from api.services import train_service, data_service
from configs.config import LOG_DIR, TRAIN_DATA_DIR

router = APIRouter(prefix="/api", tags=["Training"])


def parse_json(val, default=None):
    if val is None or val == "":
        return default
    try:
        return json.loads(val)
    except Exception:
        return val


@router.post("/train")
def train_model(
    task_id: str = Form(...),
    file: UploadFile = File(...),
    target: str = Form(...),
    prediction_length: int = Form(...),
    freq: Optional[str] = Form(None),
    eval_metric: Optional[str] = Form(None),
    use_quantile: Optional[str] = Form(None),
    use_tsfresh: bool = Form(...),
    time_limit: int = Form(...),
    enable_ensemble: bool = Form(...),
    hyperparameters: Optional[str] = Form(None),
    ensemble_types: Optional[str] = Form(None),
    hyperparameter_tune_kwargs: Optional[str] = Form(None),
    known_covariates_names: Optional[str] = Form(None),
    presets: Optional[str] = Form(None),
    num_val_windows: Optional[int] = Form(None),
    enable_shap: Optional[bool] = Form(False),
):
    """
    训练接口设计说明：
    - 每个请求各自启动一个独立训练子进程
    - 当前 HTTP 请求只等待自己的任务
    - 哪个任务先训练完，哪个请求就先返回自己的 zip 文件流
    - 日志与状态通过 task_id 分离，可由 /task_logs /task_status 单独轮询展示
    """

    # 1. 防止同一个 task_id 重复提交运行中任务
    old_state = data_service.get_task_state(task_id)
    if old_state and old_state.get("status") in {"PENDING", "RUNNING"}:
        raise HTTPException(
            status_code=409,
            detail=f"Task '{task_id}' is already running"
        )

    # 2. 初始化任务状态
    data_service.update_task_state(task_id, "PENDING", "Task accepted")

    # 3. 保存上传文件到临时目录
    temp_path, _ = data_service.save_upload_file(
        file,
        service_name="training",
        task_id=task_id
    )

    # 4. 备份训练数据
    backup_path = os.path.join(TRAIN_DATA_DIR, f"{task_id}.csv")
    shutil.copy(temp_path, backup_path)

    # 5. 分位数配置
    quantile_levels = [0.1, 0.5, 0.9] if use_quantile == "interval" else [0.5]

    # 6. 参数组装
    params = {
        "target": target,
        "prediction_length": prediction_length,
        "freq": freq,
        "eval_metric": eval_metric,
        "quantile_levels": quantile_levels,
        "use_tsfresh": use_tsfresh,
        "time_limit": time_limit,
        "enable_ensemble": enable_ensemble,
        "enable_shap": enable_shap,
        "hyperparameters": parse_json(hyperparameters, []),
        "ensemble_types": parse_json(ensemble_types, []),
        "hyperparameter_tune_kwargs": parse_json(hyperparameter_tune_kwargs, None),
        "known_covariates_names": parse_json(known_covariates_names, None),
        "presets": presets,
        "num_val_windows": num_val_windows,
    }

    # 7. 启动独立子进程训练
    proc = Process(
        target=train_service.run_training_worker,
        args=(task_id, backup_path, params),
        daemon=False
    )
    proc.start()

    # # 8. 当前请求只等待自己的进程结束
    # proc.join()

    # # 9. 读取任务状态
    # state = data_service.get_task_state(task_id)
    # status = (state or {}).get("status")

    # if status != "SUCCESS":
    #     msg = (state or {}).get("message", "Training failed")
    #     raise HTTPException(status_code=500, detail=msg)

    # zip_path = ((state or {}).get("data") or {}).get("zip_path")
    # if not zip_path or not os.path.isfile(zip_path):
    #     raise HTTPException(status_code=500, detail="Model zip not found after training")

    # return FileResponse(
    #     zip_path,
    #     filename=os.path.basename(zip_path),
    #     media_type="application/zip"
    # )
    while True:
        state = data_service.get_task_state(task_id) or {}
        status = state.get("status")
        data = state.get("data") or {}

        if status == "SUCCESS":
            zip_path = data.get("zip_path")
            if zip_path and os.path.isfile(zip_path):
                return FileResponse(
                    zip_path,
                    filename=os.path.basename(zip_path),
                    media_type="application/zip"
                )

        if status == "FAILED":
            msg = state.get("message", "Training failed")
            raise HTTPException(status_code=500, detail=msg)

        # 防止子进程异常退出但状态没写进去
        if proc.exitcode is not None and proc.exitcode != 0:
            msg = state.get("message", f"Training process exited abnormally: {proc.exitcode}")
            raise HTTPException(status_code=500, detail=msg)

        time.sleep(1)

@router.get("/task_status")
def get_status(task_id: str):
    state = data_service.get_task_state(task_id)
    if not state:
        if os.path.exists(os.path.join(LOG_DIR, f"{task_id}.log")):
            return {
                "code": 200,
                "data": {
                    "status": "UNKNOWN",
                    "message": "Task lost from memory but log exists."
                }
            }
        return {"code": 404, "message": "Task ID not found"}
    return {"code": 200, "data": state}


@router.get("/task_logs")
def get_logs(task_id: str):
    log_file = os.path.join(LOG_DIR, f"{task_id}.log")
    if not os.path.exists(log_file):
        return {"code": 200, "data": []}

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            return {"code": 200, "data": [line.strip() for line in f.readlines()]}
    except Exception:
        return {"code": 500, "message": "Failed to read logs"}


@router.post("/generate_leaderboard")
def generate_leaderboard(task_id: str = Form(...)):
    metrics_list = [
        "MASE", "MSE", "RMSE", "MAPE", "SQL",
        "WQL", "SMAPE", "RMSSE", "RMSLE", "WAPE"
    ]
    try:
        result_zip = train_service.process_leaderboard_generation(task_id, metrics_list)
        filename = os.path.basename(result_zip)
        return FileResponse(result_zip, filename=filename, media_type="application/zip")
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")