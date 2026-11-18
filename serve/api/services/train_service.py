import os
import shutil
import traceback
import zipfile
import logging
import time
import threading
import uuid
import pandas as pd
from autoapex.timeseries import TimeSeriesPredictor
from autoapex.common import space as ag

from configs.config import TRAIN_MODELS_DIR, TEMP_DIR, TRAIN_DATA_DIR, MODEL_PACKS_DIR
from api.services.data_service import prepare_tsdf, update_task_state, setup_task_logger

# 导入分离出去的 HPO 配置解析引擎
from configs.hpo_template import parse_hpo_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def prepare_training_data_with_known_covariates(train_data, prediction_length: int, known_covariates_names: list[str] | None):
    known_covariates_names = known_covariates_names or []
    if len(known_covariates_names) == 0:
        return train_data

    data_frame = train_data.reset_index()
    history_frames = []

    for _, item_df in data_frame.groupby("item_id", sort=False):
        if len(item_df) <= prediction_length:
            raise ValueError(
                f"Each series must contain more than prediction_length={prediction_length} rows when known_covariates_names is provided."
            )
        history_frames.append(item_df.iloc[:-prediction_length])

    history_df = pd.concat(history_frames, ignore_index=True)
    return type(train_data).from_data_frame(history_df, id_column="item_id", timestamp_column="timestamp")



def get_ensemble_configs(ensemble_types):
    """解析集成学习配置"""
    ensemble_hps = {}
    if not ensemble_types:
        return ensemble_hps
    if "weighted" in ensemble_types:
        ensemble_hps["WeightedEnsemble"] = {}
    if "per_item" in ensemble_types:
        ensemble_hps["PerItemGreedyEnsemble"] = {}
    if "simple" in ensemble_types:
        ensemble_hps["SimpleAverage"] = {}
    if "median" in ensemble_types:
        ensemble_hps["MedianEnsemble"] = {}
    return ensemble_hps


def zip_folder(folder_path, output_path):
    """通用文件夹打包"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)


def _generate_leaderboard_files(predictor, data, output_dir, logger, metrics=None):
    """
    在训练完成后直接生成榜单和预测文件到指定目录
    """
    if metrics is None:
        metrics = [
            "MASE", "MAE", "MSE", "RMSE", "MAPE",
            "SQL", "WQL", "SMAPE", "RMSSE", "RMSLE", "WAPE"
        ]

    try:
        prediction_length = predictor.prediction_length
        target_column = predictor.target

        # 1. 创建结果存放目录
        results_dir = os.path.join(output_dir, "evaluation_results")
        preds_dir = os.path.join(results_dir, "predictions")
        os.makedirs(preds_dir, exist_ok=True)

        logger.info(f"Generating leaderboard and predictions in {results_dir}...")

        # 2. 切分验证数据（known 协变量场景需同时切出 future known，否则 predict 会失败）
        try:
            known_cov_names = getattr(predictor, "known_covariates_names", None) or []
            data_past, known_covariates = data.get_model_inputs_for_scoring(
                prediction_length=prediction_length,
                known_covariates_names=known_cov_names if known_cov_names else None,
            )
            data_truth = data.slice_by_timestep(-prediction_length, None)
        except Exception as e:
            logger.warning(f"Data too short for validation/leaderboard generation: {e}")
            return

        # 3. 遍历模型生成预测文件
        all_models = predictor.model_names()
        for model_name in all_models:
            try:
                predictions = predictor.predict(
                    data_past,
                    known_covariates=known_covariates,
                    model=model_name,
                )
                predictions["ground_truth"] = data_truth[target_column]

                quantile_cols = []
                for col in predictions.columns:
                    if col in ["ground_truth", "mean", "item_id", "timestamp"]:
                        continue
                    try:
                        quantile_cols.append((col, float(col)))
                    except Exception:
                        continue

                rename_map = {}
                if len(quantile_cols) == 1 and quantile_cols[0][1] == 0.5:
                    col_to_drop = str(quantile_cols[0][1])
                    if col_to_drop in predictions.columns:
                        predictions.drop(columns=[col_to_drop], inplace=True)
                elif len(quantile_cols) > 0:
                    for col_name, val in quantile_cols:
                        if val < 0.5:
                            rename_map[col_name] = "lower_quantile"
                        elif val == 0.5:
                            rename_map[col_name] = "median"
                        elif val > 0.5:
                            rename_map[col_name] = "upper_quantile"
                    predictions.rename(columns=rename_map, inplace=True)

                safe_name = model_name.replace(os.path.sep, "_").replace(" ", "_")
                predictions.to_csv(os.path.join(preds_dir, f"{safe_name}.csv"))
            except Exception as e:
                logger.warning(f"Failed to predict for {model_name}: {e}")

        # 4. 生成 Leaderboard CSV
        lb_df = predictor.leaderboard(data=data, extra_metrics=metrics, silent=True)
        lb_df.to_csv(
            os.path.join(results_dir, "leaderboard.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        logger.info("Leaderboard generation completed.")

    except Exception as e:
        logger.error(f"Error during leaderboard generation: {e}")


def _normalize_ensemble_types(e_types):
    """兼容 list / json / 逗号字符串"""
    if not e_types:
        return ["weighted"]

    if isinstance(e_types, list):
        return [str(x).strip() for x in e_types if str(x).strip()]

    if isinstance(e_types, str):
        return [x.strip() for x in e_types.split(",") if x.strip()]

    return ["weighted"]


def run_training_logic(task_id: str, file_path: str, params: dict) -> str:
    """
    真正训练逻辑：
    - 只负责训练、生成模型目录
    - 不在这里写最终 SUCCESS，最终状态由 worker 统一写入
    """
    if not task_id:
        raise ValueError("Task ID is required for training.")

    logger = setup_task_logger(task_id)
    autoapex_logger = logging.getLogger("autoapex")
    autoapex_logger.setLevel(logging.INFO)

    # 将当前 task logger 的 handler 临时挂到 autoapex logger，确保日志写到 task_id 对应文件
    task_handlers = []
    for handler in logger.handlers:
        if handler not in autoapex_logger.handlers:
            autoapex_logger.addHandler(handler)
            task_handlers.append(handler)

    try:
        update_task_state(task_id, "RUNNING", "Initializing training...")
        logger.info(f"Start Training Task: {task_id}")

        # 1. 准备数据
        target_col = params.get("target", "value")
        prediction_length = params.get("prediction_length")
        known_covariates_names = params.get("known_covariates_names") or []

        train_data = prepare_tsdf(file_path, default_id=task_id, target_col=target_col)
        logger.info("Training data prepared.")

        if known_covariates_names:
            logger.info(
                "检测到 known_covariates_names 非空，训练数据将切除最后 prediction_length 作为未来已知变量保留区。"
            )
            train_data = prepare_training_data_with_known_covariates(
                train_data=train_data,
                prediction_length=prediction_length,
                known_covariates_names=known_covariates_names,
            )

        # 2. 模型输出目录
        model_path = os.path.join(TRAIN_MODELS_DIR, task_id)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        os.makedirs(model_path, exist_ok=True)

        # 3. 初始化 Predictor
        predictor_args = {
            "path": model_path,
            "target": target_col,
            "prediction_length": params.get("prediction_length"),
            "freq": params.get("freq"),
            "eval_metric": params.get("eval_metric"),
            "use_tsfresh": params.get("use_tsfresh", False),
            "verbosity": 4,
        }
        if params.get("quantile_levels"):
            predictor_args["quantile_levels"] = params.get("quantile_levels")
        if known_covariates_names:
            predictor_args["known_covariates_names"] = known_covariates_names

        logger.info(f"Predictor args: {predictor_args}")
        predictor = TimeSeriesPredictor(**predictor_args)

        # 4. HPO / preset 配置
        presets = params.get("presets")
        user_hyperparameters = params.get("hyperparameters") or []
        user_tune_kwargs = params.get("hyperparameter_tune_kwargs")

        fit_kwargs = {
            "train_data": train_data,
            "time_limit": params.get("time_limit"),
        }

        if presets:
            fit_kwargs["presets"] = presets
            logger.info(f"使用 preset: {presets}")
            if user_hyperparameters:
                logger.info("preset 模式下同时收到 hyperparameters，将作为覆盖项解析")
                fit_kwargs["hyperparameters"] = parse_hpo_config(user_hyperparameters)
        else:
            logger.info("正在解析并注入经验 HPO 超参空间...")
            fit_kwargs["hyperparameters"] = parse_hpo_config(user_hyperparameters)

        if params.get("num_val_windows") is not None:
            fit_kwargs["num_val_windows"] = params["num_val_windows"]

        if user_tune_kwargs is not None:
            fit_kwargs["hyperparameter_tune_kwargs"] = user_tune_kwargs

        if params.get("enable_ensemble"):
            e_list = _normalize_ensemble_types(params.get("ensemble_types"))
            fit_kwargs["enable_ensemble"] = True
            fit_kwargs["ensemble_hyperparameters"] = get_ensemble_configs(e_list)
            logger.info(f"Ensemble enabled. Types: {e_list}")
        else:
            fit_kwargs["enable_ensemble"] = False
            logger.info("Ensemble disabled.")

        logger.info("Training started...")
        predictor.fit(**fit_kwargs)

        # 6. 生成 leaderboard
        update_task_state(task_id, "RUNNING", "Generating Leaderboard...")
        _generate_leaderboard_files(predictor, train_data, model_path, logger)

        # 7. SHAP
        if params.get("enable_shap", False):
            update_task_state(task_id, "RUNNING", "Generating SHAP explanations...")
            try:
                shap_output_dir = os.path.join(model_path, "shap_results")
                predictor.explain(data=train_data, output_dir=shap_output_dir)
                logger.info("SHAP generation finished.")
            except Exception as e:
                logger.warning(f"SHAP explain() failed, skipping: {e}")
        else:
            logger.info("SHAP disabled, skipping explain().")

        logger.info("Training finished.")
        return model_path

    except Exception as e:
        logger.error(f"Training failed: {traceback.format_exc()}")
        update_task_state(task_id, "FAILED", f"Training failed: {str(e)}")
        raise

    finally:
        for handler in task_handlers:
            try:
                autoapex_logger.removeHandler(handler)
            except Exception:
                pass


def process_leaderboard_generation(task_id: str, metrics_list: list) -> str:
    """根据已训练模型重新生成榜单 zip，供 /api/generate_leaderboard 使用"""
    model_path = os.path.join(TRAIN_MODELS_DIR, task_id)
    data_path = os.path.join(TRAIN_DATA_DIR, f"{task_id}.csv")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model not found for task {task_id}")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Train data not found for task {task_id}")

    predictor = TimeSeriesPredictor.load(model_path, require_version_match=False)
    train_data = prepare_tsdf(data_path, default_id=task_id, target_col=predictor.target)

    logger = setup_task_logger(task_id)
    _generate_leaderboard_files(predictor, train_data, model_path, logger, metrics=metrics_list)

    zip_filename = f"{task_id}_leaderboard.zip"
    zip_path = os.path.join(TEMP_DIR, zip_filename)

    results_dir = os.path.join(model_path, "evaluation_results")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(results_dir):
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, results_dir)
                zipf.write(abs_path, rel_path)

    return zip_path


def _heartbeat_thread_fn(task_id: str, stop_event: threading.Event):
    """子进程内后台线程：定期写心跳，便于主进程检测卡住"""
    while not stop_event.wait(60):
        try:
            update_task_state(
                task_id,
                "RUNNING",
                "Training in progress",
                data={"last_heartbeat": time.time()}
            )
        except Exception:
            break


def run_training_worker(task_id: str, file_path: str, params: dict) -> None:
    """
    在子进程中执行训练：
    - 每个任务独立进程，日志天然隔离
    - 当前 /train 请求只等待自己的这个进程
    - 训练完成后将 zip_path 写入状态，供主请求返回文件流
    """
    logger = setup_task_logger(task_id)

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_thread_fn,
        args=(task_id, stop_event),
        daemon=True
    )
    heartbeat_thread.start()

    try:
        model_path = run_training_logic(task_id, file_path, params)

        model_id = str(uuid.uuid4())
        pack_zip_path = os.path.join(MODEL_PACKS_DIR, f"{model_id}.zip")
        zip_folder(model_path, pack_zip_path)

        zip_filename = f"{task_id}_model_pack.zip"
        zip_path = os.path.join(TEMP_DIR, zip_filename)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        shutil.copy2(pack_zip_path, zip_path)

        update_task_state(
            task_id,
            "SUCCESS",
            "Training Completed",
            data={"zip_path": zip_path, "model_id": model_id}
        )
        logger.info(f"Model pack saved: {pack_zip_path} (model_id={model_id})")

    except Exception as e:
        logger.error(f"Training failed in worker: {traceback.format_exc()}")
        update_task_state(task_id, "FAILED", f"Training failed: {str(e)}")

    finally:
        stop_event.set()
        try:
            heartbeat_thread.join(timeout=1)
        except Exception:
            pass