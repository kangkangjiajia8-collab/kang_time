import os
import shutil
import zipfile
import traceback
import uuid
import time
import re
import json
from autoapex.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from configs.config import INFERENCE_DIR, TEMP_DIR, MODEL_PACKS_DIR, GLOBAL_LOGGER
from api.services.data_service import get_or_create_task_id, prepare_output_dir, prepare_tsdf, setup_task_logger
import pandas as pd






def _split_inference_data_for_known_covariates(
    data: TimeSeriesDataFrame,
    prediction_length: int,
    known_covariates_names: list[str],
):
    data_frame = data.reset_index()
    history_frames = []
    future_covariate_frames = []

    for _, item_df in data_frame.groupby("item_id", sort=False):
        if len(item_df) <= prediction_length:
            raise ValueError(
                f"Each series must contain more than prediction_length={prediction_length} rows when future known covariates are provided."
            )
        history_frames.append(item_df.iloc[:-prediction_length])
        future_slice = item_df.iloc[-prediction_length:][["item_id", "timestamp", *known_covariates_names]].copy()
        future_covariate_frames.append(future_slice)

    history_df = pd.concat(history_frames, ignore_index=True)
    future_covariates_df = pd.concat(future_covariate_frames, ignore_index=True)

    history_data = TimeSeriesDataFrame.from_data_frame(history_df, id_column="item_id", timestamp_column="timestamp")
    future_known_covariates = TimeSeriesDataFrame.from_data_frame(
        future_covariates_df,
        id_column="item_id",
        timestamp_column="timestamp",
    )
    return history_data, future_known_covariates

def find_predictor_home(extract_root: str) -> str:
    """
    递归查找 predictor.pkl 文件，返回其所在目录。
    """
    for root, dirs, files in os.walk(extract_root):
        if "predictor.pkl" in files:
            return root
    
    raise FileNotFoundError("Could not find 'predictor.pkl' in the uploaded zip.")

_UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def resolve_model_zip_path(model_id: str) -> str:
    """根据训练阶段保存的 model_id（UUID）解析模型包路径。"""
    model_id = (model_id or "").strip()
    if not _UUID_PATTERN.match(model_id):
        raise ValueError(f"Invalid model_id format: {model_id}")
    zip_path = os.path.join(MODEL_PACKS_DIR, f"{model_id}.zip")
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Model pack not found for model_id: {model_id}")
    return zip_path


def run_inference(
    task_id: str,
    best_model: str,
    zip_path: str,
    csv_path: str,
    model_id: str | None = None,
):
    task_id = get_or_create_task_id(task_id)
    logger = setup_task_logger(task_id, capture_autoapex=False)

    # 1. 临时解压目录
    extract_path = os.path.join(TEMP_DIR, f"{task_id}_model_extract")
    
    # 2. 结果保存目录
    result_dir = prepare_output_dir(INFERENCE_DIR, task_id)
    output_path = os.path.join(result_dir, "predictions.csv")
    log_file = os.path.join(result_dir, f"{task_id}.log")
    metadata_file = os.path.join(result_dir, "inference_metadata.json")
    
    inference_metadata = {
        "task_id": task_id,
        "best_model": best_model,
        "status": "running",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_files": {
            "model_zip": zip_path,
            "data_csv": csv_path,
        }
    }
    if model_id:
        inference_metadata["model_id"] = model_id
    
    try:
        logger.info(f"=== 推理任务开始 ===")
        logger.info(f"Task ID: {task_id}")
        logger.info(f"Best Model: {best_model}")
        logger.info(f"Model ZIP: {zip_path}")
        logger.info(f"Data CSV: {csv_path}")
        
        # 解压
        logger.info(f"正在解压模型包...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info(f"模型包解压完成: {extract_path}")
            
        real_model_path = find_predictor_home(extract_path)
        logger.info(f"找到预测器路径: {real_model_path}")
        
        predictor = TimeSeriesPredictor.load(real_model_path, require_version_match=False)
        logger.info(f"预测器加载成功")
        
        data = prepare_tsdf(csv_path)
        logger.info(f"数据加载成功，样本数: {len(data)}")

        known_covariates_names = predictor.known_covariates_names or []

        if known_covariates_names:
            logger.info("检测到训练阶段存在未来已知变量标记，开始切分历史数据与未来已知变量数据。")
            history_data, future_known_covariates = _split_inference_data_for_known_covariates(
                data=data,
                prediction_length=predictor.prediction_length,
                known_covariates_names=known_covariates_names,
            )
            predictions = predictor.predict(
                history_data,
                known_covariates=future_known_covariates,
                model=best_model,
            )
        else:
            logger.info("未检测到未来已知变量标记，走当前默认预测流程。")
            predictions = predictor.predict(data, model=best_model)
        logger.info(f"预测完成，结果行数: {len(predictions)}")
    
        quantile_cols = []
        for col in predictions.columns:
            if col in ["mean", "item_id", "timestamp","time"]:
                continue
            try:
                val = float(col)
                quantile_cols.append((col, val))
            except ValueError:
                continue 

        if len(quantile_cols) == 1 and quantile_cols[0][1] == 0.5:
            predictions.drop(columns=[quantile_cols[0][0]], inplace=True)
        elif len(quantile_cols) > 0:
            rename_map = {}
            for col_name, val in quantile_cols:
                if val < 0.5:
                    rename_map[col_name] = "lower_quantile"
                elif val == 0.5:
                    rename_map[col_name] = "median"
                elif val > 0.5:
                    rename_map[col_name] = "upper_quantile"
            predictions.rename(columns=rename_map, inplace=True)
        
        predictions.to_csv(output_path)
        logger.info(f"预测结果已保存: {output_path}")
        
        # 更新元数据
        inference_metadata["status"] = "success"
        inference_metadata["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        inference_metadata["output_files"] = {
            "predictions_csv": output_path,
            "log_file": log_file,
            "metadata_file": metadata_file
        }
        inference_metadata["predictions_shape"] = {
            "rows": len(predictions),
            "columns": list(predictions.columns)
        }
        
        logger.info(f"=== 推理任务完成 ===")
        return output_path, log_file, metadata_file, inference_metadata
        
    except Exception as e:
        logger.error(f"推理失败: {traceback.format_exc()}")
        inference_metadata["status"] = "failed"
        inference_metadata["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        inference_metadata["error"] = str(e)
        raise e
    finally:
        # 保存元数据
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(inference_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存元数据失败: {e}")
        
        # 清理临时解压目录
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
            logger.info(f"已清理临时解压目录")