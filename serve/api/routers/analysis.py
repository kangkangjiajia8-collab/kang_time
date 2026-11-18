import os
import base64
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Body
from fastapi.responses import FileResponse
from api.services import analysis_service, data_service
import json
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from configs.config import TEMP_DIR
import binascii
router = APIRouter(prefix="/api", tags=["Analysis & Features"])

class FeatureConfigItem(BaseModel):
    name: str          # 例如 "c3"
    params: Dict[str, Any] = {}  # <--- 修改这里：接收字符串列表，例如 ["lag"]



@router.post("/analyze_csv")
async def analyze_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form("value")
):
    # 使用新方法保存：temp/analysis/{generated_id}/filename
    try:
        input_path, request_id = data_service.save_upload_file(file, service_name="analysis")
        
        output_path = analysis_service.compute_boxplot(input_path, target_column, request_id)
        
        # 清理
        background_tasks.add_task(data_service.cleanup_files, [input_path])
        
        return FileResponse(output_path, media_type='text/csv', filename="distribution_boxplot.csv")
    except Exception as e:
        # 如果 save_upload_file 成功但后面失败了，尝试清理 (如果 input_path 已定义)
        # 简单起见，这里抛出异常即可，临时文件会留存供排查，或依赖系统定期清理
        raise HTTPException(500, detail=str(e))

@router.post("/fill_missing_values")
async def fill_missing(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    method: str = Form("auto")
):
    try:
        input_path, request_id = data_service.save_upload_file(file, service_name="analysis")
        
        # 这里的 method 会传给 AutoApex 的 fill_missing_values
        output_path, filled_status = analysis_service.fill_missing_values(input_path, method, request_id)
        
        background_tasks.add_task(data_service.cleanup_files, [input_path])
        
        headers = {
            "X-Filled-Status": str(filled_status).lower(),
            "X-File-Name": f"processed_{file.filename}",
            "Access-Control-Expose-Headers": "X-Filled-Status, X-File-Name"
        }
        return FileResponse(output_path, media_type="text/csv", filename=f"processed_{file.filename}", headers=headers)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@router.post("/get_characteristics")
async def get_characteristics(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form("value")
):
    try:
        input_path, request_id = data_service.save_upload_file(file, service_name="analysis")
        
        scores = analysis_service.compute_characteristics(input_path, target_column)
        
        background_tasks.add_task(data_service.cleanup_files, [input_path])
        return {"status": "success", "data": scores}
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    

@router.post("/generate_all_features")
async def generate_features(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV 文件流"),
    task_id: str = Form(..., description="任务 ID"),
    max_timeshift: int = Form(29, description="最大时移"),
    min_timeshift: int = Form(29, description="最小时移"),
    custom_params: Optional[str] = Form(None, description="自定义特征参数，JSON 字符串，与 hyperparameters 格式一致"),
    # col_old: Optional[str] = Form(None, description="旧列名")
):

    try:
        save_path, task_id = data_service.save_upload_file(
            file, service_name="feature_engineering", task_id=task_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # 2. 解析 custom_params（str -> List[Dict]）
    try:
        params_list = json.loads(custom_params) if custom_params else []
    except (TypeError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid custom_params JSON: {e}")
    if not isinstance(params_list, list):
        raise HTTPException(status_code=400, detail="custom_params must be a JSON array")

    # 3. 子进程执行特征生成（支持并发，每任务独立日志 + 心跳防卡住）
    data_service.start_heartbeat_monitor()
    background_tasks.add_task(
        analysis_service.run_feature_generation_concurrent,
        task_id,
        save_path,
        max_timeshift,
        min_timeshift,
        params_list,
    )

    return {
        "code": 200,
        "message": "Feature generation started",
        "data": {
            "task_id": task_id,
            "status": "PENDING",
            "info": "Poll /api/task_status for progress, /api/task_logs for logs",
        },
    }

@router.post("/select_features_and_download")
def select_features(
    task_id: str = Form(...),
    fdr: float = Form(0.05)
):
    try:
        output_path = analysis_service.select_features(task_id, fdr)
        filename = f"selected_features_{task_id}_fdr{fdr}.csv"
        return FileResponse(output_path, media_type='text/csv', filename=filename)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@router.post("/calculate_correlation")
async def calc_correlation(
    background_tasks: BackgroundTasks,
    method: str = Form("pearson"), 
    file: UploadFile = File(...)
):
    try:
        input_path, _ = data_service.save_upload_file(file, service_name="analysis")
        
        output_path = analysis_service.calculate_correlation_matrix(input_path, method)
        
        background_tasks.add_task(data_service.cleanup_files, [input_path])
        return FileResponse(output_path, media_type='text/csv', filename="correlation_matrix.csv")
    except Exception as e:
        raise HTTPException(500, detail=str(e))