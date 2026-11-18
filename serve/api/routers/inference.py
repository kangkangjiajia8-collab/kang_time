import os
import json
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from api.services import predict_service, data_service
from configs.config import GLOBAL_LOGGER

router = APIRouter(prefix="/api", tags=["Prediction"])

@router.post("/inference")
async def inference(
    background_tasks: BackgroundTasks,
    task_id: str = Form(...),
    best_model: str = Form(...),
    model_zip: UploadFile = File(...),
    data_csv: UploadFile = File(...)
):
    zip_path = None
    csv_path = None
    
    try:
        # 1. 保存模型包: temp/prediction/{task_id}/xxx.zip
        zip_path, _ = data_service.save_upload_file(model_zip, service_name="prediction", task_id=task_id)
        
        # 2. 保存数据 CSV: temp/prediction/{task_id}/xxx.csv
        csv_path, _ = data_service.save_upload_file(data_csv, service_name="prediction", task_id=task_id)
        
        # 3. 执行推理
        output_path, log_file, metadata_file, inference_metadata = predict_service.run_inference(task_id, best_model, zip_path, csv_path)
        
        # 4. 清理上传的临时文件
        background_tasks.add_task(data_service.cleanup_files, [zip_path, csv_path])
        
        # 5. 记录推理完成信息
        GLOBAL_LOGGER.info(f"✅ 推理任务完成 - Task ID: {task_id}")
        GLOBAL_LOGGER.info(f"   预测结果: {output_path}")
        GLOBAL_LOGGER.info(f"   日志文件: {log_file}")
        GLOBAL_LOGGER.info(f"   元数据: {metadata_file}")
        
        # 6. 返回预测结果 CSV 文件
        return FileResponse(output_path, filename=f"{task_id}_predictions.csv", media_type='text/csv')
        
    except Exception as e:
        error_msg = f"Inference failed: {str(e)}"
        GLOBAL_LOGGER.error(f"❌ 推理任务失败 - Task ID: {task_id}, Error: {error_msg}")
        
        # 清理已保存的上传文件
        if zip_path or csv_path:
            background_tasks.add_task(data_service.cleanup_files, [p for p in [zip_path, csv_path] if p])
        
        raise HTTPException(500, detail=error_msg)


@router.post("/inference/v1")
async def inference_v1(
    background_tasks: BackgroundTasks,
    task_id: str = Form(...),
    best_model: str = Form(...),
    model_id: str = Form(...),
    data_csv: UploadFile = File(...),
):
    """
    推理 v1：通过 model_id（训练完成后保存的 UUID）加载模型包，无需上传 zip。
    模型包路径：outputs/model_packs/{model_id}.zip
    """
    csv_path = None

    try:
        zip_path = predict_service.resolve_model_zip_path(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        csv_path, _ = data_service.save_upload_file(
            data_csv, service_name="prediction", task_id=task_id
        )

        output_path, log_file, metadata_file, inference_metadata = predict_service.run_inference(
            task_id, best_model, zip_path, csv_path, model_id=model_id
        )

        background_tasks.add_task(data_service.cleanup_files, [csv_path])

        GLOBAL_LOGGER.info(f"✅ 推理 v1 任务完成 - Task ID: {task_id}, model_id: {model_id}")
        GLOBAL_LOGGER.info(f"   预测结果: {output_path}")

        return FileResponse(
            output_path,
            filename=f"{task_id}_predictions.csv",
            media_type="text/csv",
        )

    except Exception as e:
        error_msg = f"Inference v1 failed: {str(e)}"
        GLOBAL_LOGGER.error(
            f"❌ 推理 v1 失败 - Task ID: {task_id}, model_id: {model_id}, Error: {error_msg}"
        )
        if csv_path:
            background_tasks.add_task(data_service.cleanup_files, [csv_path])
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/inference/{task_id}/log")
async def get_inference_log(task_id: str):
    """获取推理任务的日志文件"""
    from configs.config import INFERENCE_DIR
    log_file = os.path.join(INFERENCE_DIR, task_id, f"{task_id}.log")
    
    if not os.path.exists(log_file):
        raise HTTPException(404, detail=f"Log file not found for task {task_id}")
    
    return FileResponse(log_file, filename=f"{task_id}.log", media_type='text/plain')

@router.get("/inference/{task_id}/metadata")
async def get_inference_metadata(task_id: str):
    """获取推理任务的元数据"""
    from configs.config import INFERENCE_DIR
    metadata_file = os.path.join(INFERENCE_DIR, task_id, "inference_metadata.json")
    
    if not os.path.exists(metadata_file):
        raise HTTPException(404, detail=f"Metadata file not found for task {task_id}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return JSONResponse(metadata)

@router.get("/inference/{task_id}/predictions")
async def get_inference_predictions(task_id: str):
    """获取推理任务的预测结果"""
    from configs.config import INFERENCE_DIR
    predictions_file = os.path.join(INFERENCE_DIR, task_id, "predictions.csv")
    
    if not os.path.exists(predictions_file):
        raise HTTPException(404, detail=f"Predictions file not found for task {task_id}")
    
    return FileResponse(predictions_file, filename=f"{task_id}_predictions.csv", media_type='text/csv')