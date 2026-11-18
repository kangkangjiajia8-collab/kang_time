import os
import json
import traceback
import time
import threading
import pandas as pd
import numpy as np
from multiprocessing import Process
from autoapex.timeseries.utils.features import TimeSeriesFeatureGenerator
from autoapex.timeseries import TimeSeriesDataFrame
from configs.config import ANALYSIS_DIR, FEATURE_DIR, TEMP_DIR
from configs.fe_template import parse_feature_config
from api.services.data_service import (
    prepare_tsdf, update_task_state, setup_task_logger,
    get_or_create_task_id, prepare_output_dir,
    cancel_running_service, register_running_service, unregister_running_service,
    register_running_process, unregister_running_process,
)

def compute_boxplot(file_path: str, target_column: str, request_id: str = None) -> str:
    """计算月度箱线图"""
    task_id = get_or_create_task_id(request_id)
    save_dir = prepare_output_dir(ANALYSIS_DIR, task_id)
    
    ts_df = prepare_tsdf(file_path, default_id="default_item")
    
    real_target = target_column if target_column in ts_df.columns else 'value'
    if real_target not in ts_df.columns:
        real_target = ts_df.columns[-1]

    # 调用 AutoApex 扩展方法
    boxplot_df = ts_df._compute_monthly_boxplot_data(target=real_target)
    
    output_path = os.path.join(save_dir, "distribution_boxplot.csv")
    boxplot_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return output_path

def compute_characteristics(file_path: str, target_column: str):
    """计算时间序列特征评分 (周期性、趋势性等)"""
    ts_df = prepare_tsdf(file_path)
    real_target = target_column if target_column in ts_df.columns else 'value'
    if real_target not in ts_df.columns:
        real_target = ts_df.columns[-1]
    
    # 调用 AutoApex 扩展方法
    scores = ts_df._compute_heterogeneity_scores(target=real_target)
    return scores


def fill_missing_values(file_path: str, method: str, request_id: str = None):
    """
    缺失值填补入口函数
    """
    task_id = get_or_create_task_id(request_id)
    save_dir = prepare_output_dir(ANALYSIS_DIR, task_id)
    logger = setup_task_logger(task_id, capture_autoapex=False)
    
    # 1. 读取数据
    df = pd.read_csv(file_path)
    
    # 2. 标准化列名 (Service 层负责适配用户上传的杂乱列名)
    time_col = 'timestamp'
    target_col = 'value'
    id_col = 'item_id'
    
    # 3. 自动检测时间戳列
    potential_time_cols = []
    # 读取前几行数据进行判断
    sample_rows = min(5, len(df))
    sample_df = df.head(sample_rows)
    logger.info(f"正在检测时间戳列，使用前 {sample_rows} 行数据进行判断")
    
    for col in df.columns:
        # 先尝试转换为 datetime 类型
        try:
            # 尝试转换前几行数据
            pd.to_datetime(sample_df[col])
            # 如果转换成功，认为是时间戳列
            potential_time_cols.append(col)
            logger.info(f"检测到潜在的时间戳列: {col}")
        except:
            continue
    
    # 如果找到潜在的时间戳列，选择第一个作为时间列
    if potential_time_cols:
        logger.info(f"找到 {len(potential_time_cols)} 个潜在的时间戳列: {potential_time_cols}")
        # 如果时间列已经存在，先检查是否在潜在列表中
        if time_col in df.columns:
            if time_col not in potential_time_cols:
                # 如果现有的 timestamp 列不是时间戳类型，使用找到的第一个潜在时间列
                logger.info(f"现有 timestamp 列不是时间戳类型，使用 {potential_time_cols[0]} 作为时间列")
                df.rename(columns={potential_time_cols[0]: time_col}, inplace=True)
            else:
                logger.info("使用现有的 timestamp 列作为时间列")
        else:
            # 如果没有 timestamp 列，使用找到的第一个潜在时间列
            logger.info(f"使用 {potential_time_cols[0]} 作为时间列")
            df.rename(columns={potential_time_cols[0]: time_col}, inplace=True)
    else:
        # 如果没有找到潜在的时间戳列，尝试使用常见的时间列名
        logger.info("未检测到潜在的时间戳列，尝试使用常见的时间列名")
        if 'date' in df.columns:
            logger.info("使用 date 列作为时间列")
            df.rename(columns={'date': time_col}, inplace=True)
        elif 'time' in df.columns:
            logger.info("使用 time 列作为时间列")
            df.rename(columns={'time': time_col}, inplace=True)
        else:
            logger.warning("未找到时间戳列")
    
    # 4. 检查时间列是否按顺序排列，若不是则排序
    if time_col in df.columns:
        logger.info(f"正在检查时间列 {time_col} 是否按顺序排列")
        # 尝试将时间列转换为 datetime 类型
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            logger.info("时间列转换为 datetime 类型成功")
            # 检查是否按时间顺序排列
            if not df[time_col].is_monotonic_increasing:
                logger.info("时间列不是按顺序排列，正在排序")
                # 按时间列排序
                df.sort_values(by=time_col, inplace=True)
                # 重置索引
                df.reset_index(drop=True, inplace=True)
                logger.info("时间列排序完成")
            else:
                logger.info("时间列已经按顺序排列")
        except Exception as e:
            logger.warning(f"时间列转换失败: {str(e)}")
            # 如果转换失败，继续执行，不影响后续处理
            pass
        
    if id_col not in df.columns:
        df[id_col] = "default_item"
    
    if target_col not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[-1]
            df.rename(columns={target_col: 'value'}, inplace=True)
            target_col = 'value'
            
    if time_col not in df.columns:
        raise ValueError("Could not identify timestamp column.")

    # 3. 调用底层 classmethod
    df_result = TimeSeriesDataFrame.clean_and_fill(
        df=df,
        time_col=time_col,
        target_col=target_col,
        id_col=id_col,
        fill_method=method,
    )
    
    if id_col in df_result.columns:
        df_result.drop(columns=[id_col], inplace=True)
        
    output_path = os.path.join(save_dir, "filled_data.csv")
    df_result.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    # 只要 flag 不是 3，就算有填充发生
    has_filled = (df_result['fill_flag'] != 3).any()
    
    return output_path, has_filled


def calculate_correlation_matrix(file_path: str, method: str):
    """相关性分析"""
    task_id = get_or_create_task_id(None)
    save_dir = prepare_output_dir(ANALYSIS_DIR, task_id)
    
    df = pd.read_csv(file_path)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found.")
    
    corr_matrix = numeric_df.corr(method=method)
    
    save_path = os.path.join(save_dir, "correlation_matrix.csv")
    corr_matrix.to_csv(save_path, index=True)
    return save_path

def select_features(task_id: str, fdr: float):
    """根据 FDR 进行特征筛选"""
    # 1. 寻找全量特征文件
    logger = setup_task_logger(task_id, capture_autoapex=False)
    update_task_state(task_id, "RUNNING", f"正在筛选特征 (FDR={fdr})...")
    logger.info(f"【筛选启动】开始进行特征筛选，设定 FDR = {fdr}")
    try:
        full_feat_dir = os.path.join(FEATURE_DIR, task_id)
        full_feat_path = os.path.join(full_feat_dir, "full_features.csv")
        logger.info(f"正在读取全量特征文件...")
        if not os.path.exists(full_feat_path):
            raise FileNotFoundError(f"Feature file for task {task_id} not found. Please run generate_all_features first.")
        
        df = pd.read_csv(full_feat_path)
        logger.info(f"数据读取成功，当前特征维度: {df.shape}")
        target_col = "value"
        if target_col not in df.columns: 
            raise ValueError("Target column 'value' missing in features file.")
        
        # 准备 X, y
        y = df[target_col]
        X = df.drop(columns=[target_col, "item_id", "timestamp"], errors='ignore').select_dtypes(include=[np.number])
        
        # 计算相关性
        logger.info("开始计算特征相关性 (Relevance Calculation)...")
        relevance = TimeSeriesFeatureGenerator.calculate_relevance_standalone(X, y, fdr_level=fdr)
        
        # 获取保留列
        if 'feature' in relevance.columns:
            keep = relevance[relevance['relevant'] == True]['feature'].tolist()
        else:
            keep = relevance[relevance['relevant'] == True].index.tolist()
        logger.info(f"筛选计算完成。保留特征数量: {len(keep)} / {len(X.columns)}")
        final_cols = ["item_id", "timestamp", "value"] + keep
        final_cols = [c for c in final_cols if c in df.columns]
        
        # 保存结果
        out_dir = prepare_output_dir(FEATURE_DIR, task_id)
        out_path = os.path.join(out_dir, f"selected_fdr_{fdr}.csv")
        df[final_cols].to_csv(out_path, index=False)

        update_task_state(task_id, "SUCCESS", "特征筛选完成", {
            "selected_count": len(keep),
            "dataset_path": out_path
        })
        logger.info("【筛选结束】任务成功完成。")
        return out_path
    except Exception as e:
        update_task_state(task_id, "FAILED", f"特征筛选失败: {str(e)}")
        raise e

def _feature_heartbeat_thread_fn(task_id: str, stop_event: threading.Event):
    """子进程内：定期写心跳，便于主进程检测卡住"""
    while not stop_event.wait(60):
        try:
            update_task_state(task_id, "RUNNING", "Feature generation in progress", data={"last_heartbeat": time.time()})
        except Exception:
            break


def worker_generate_features(
    task_id: str, 
    file_path: str, 
    max_timeshift: int, 
    min_timeshift: int, 
    tsfresh_custom_params: list # <--- 修改类型提示：接收 list
):
    """后台任务：生成全量特征（可在子进程中运行，每任务独立日志）"""
    task_id = get_or_create_task_id(task_id)
    logger = setup_task_logger(task_id, capture_autoapex=False)
    stop_heartbeat = threading.Event()
    heartbeat_thread = threading.Thread(target=_feature_heartbeat_thread_fn, args=(task_id, stop_heartbeat), daemon=True)
    heartbeat_thread.start()
    try:
        logger.info(f"【任务启动】开始执行特征生成 (TaskID: {task_id})")
        update_task_state(task_id, "RUNNING", "Initializing data...", data={"last_heartbeat": time.time()})
        
        # 1. 准备数据
        logger.info(f"正在读取并校验数据文件: {file_path}")
        logger.info(f"tsfresh_custom_params: {tsfresh_custom_params}")
        df = pd.read_csv(file_path)
        
        # 1. 统一列名 (容错处理)
        if 'date' in df.columns:
            df.rename(columns={'date': 'timestamp'}, inplace=True)
        if 'time' in df.columns:
            df.rename(columns={'time': 'timestamp'}, inplace=True)
        if 'item_id' not in df.columns:
            df['item_id'] = task_id
        if 'value' not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df.rename(columns={numeric_cols[-1]: 'value'}, inplace=True)

        # 2. 剔除 CSV 中的无效完全空行
        df.dropna(how='all', inplace=True)

        # # 3. 核心：调用你写好的清洗和断档填补逻辑
        # logger.info("修复缺失时间戳与数值断档...")
        # df_cleaned = clean_and_fill_timeseries(
        #     df=df,
        #     time_col='timestamp',
        #     target_col='value',
        #     id_col='item_id',
        #     fill_method='auto'
        # )

        # 4. 安全地转换为 AutoApex TSDF
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df, 
            id_column='item_id', 
            timestamp_column='timestamp'
        )
        # ================= 修改结束 =================

        logger.info(f"数据加载成功，时间序列形状: {ts_df.shape}, 实体数量: {ts_df.num_items}")

        # 2. 解析 tsfresh 参数 (还原了你原版代码中的 eval 逻辑)
        parsed_params = "comprehensive"
        if tsfresh_custom_params and isinstance(tsfresh_custom_params, list):
            # 传入的已经是 List[Dict] 了，直接解析
            final_fc_parameters = parse_feature_config(tsfresh_custom_params)

        tsfresh_config = {
            "impute_strategy": "mean",
            "window_size": 30,
            "max_timeshift": max_timeshift,
            "min_timeshift": min_timeshift,
            "do_selection": False, 
            "fc_parameters": final_fc_parameters
        }

        # 3. 初始化生成器
        logger.info("初始化特征生成器...")
        feature_generator = TimeSeriesFeatureGenerator(
            target='value',
            known_covariates_names=[],
            use_tsfresh=True,
            tsfresh_settings=tsfresh_config,
            correlation_analysis=False 
        )

        update_task_state(task_id, "RUNNING", "Running Feature Generator...")
        logger.info(">>> AutoApex开始运行特征生成引擎...")
        logger.info("提示：此过程涉及大量计算，请耐心等待...") 
        full_tsdf_result = feature_generator.fit_transform(ts_df)
        logger.info("<<< 特征生成完成！")
        # 4. 保存结果
        save_dir = prepare_output_dir(FEATURE_DIR, task_id)
        save_path = os.path.join(save_dir, "full_features.csv")
        
        result_df = full_tsdf_result.reset_index()
        result_df.to_csv(save_path, index=False)
        
        # 生成的特征字段名（排除 item_id, timestamp, value）
        feature_columns = [c for c in result_df.columns if c not in ("item_id", "timestamp", "value")]
        feature_count = len(feature_columns)
        logger.info(f"任务结束。共生成特征数: {feature_count}")
        logger.info("生成的特征字段: " + str(feature_columns))
        update_task_state(task_id, "SUCCESS", "Done", {
            "feature_count": feature_count,
            "download_path": save_path,
            "feature_columns": feature_columns,
        })

    except Exception as e:
        logger.error(f"特征生成任务失败: {traceback.format_exc()}")
        update_task_state(task_id, "FAILED", str(e))
    finally:
        stop_heartbeat.set()
        # 清理原始上传的 CSV
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


# 服务单例键（可选）：run_feature_generation_task 会先终止上一任务
FEATURE_GENERATION_SERVICE_NAME = "feature_generation"


def run_feature_generation_task(
    task_id: str,
    file_path: str,
    max_timeshift: int,
    min_timeshift: int,
    tsfresh_custom_params: list,
) -> None:
    """
    在子进程中执行特征生成；若已有任务在跑则先强制终止再启动本次。
    由路由层通过 BackgroundTasks.add_task 调用，不阻塞请求。
    """
    cancel_running_service(FEATURE_GENERATION_SERVICE_NAME)
    proc = Process(
        target=worker_generate_features,
        args=(task_id, file_path, max_timeshift, min_timeshift, tsfresh_custom_params),
    )
    proc.start()
    register_running_service(FEATURE_GENERATION_SERVICE_NAME, proc, task_id, file_path=file_path)
    proc.join()
    unregister_running_service(FEATURE_GENERATION_SERVICE_NAME, task_id)


def run_feature_generation_concurrent(
    task_id: str,
    file_path: str,
    max_timeshift: int,
    min_timeshift: int,
    tsfresh_custom_params: list,
) -> None:
    """
    在子进程中执行特征生成，支持与其它任务并发（不取消已有任务）；
    子进程内独立日志 + 心跳，主进程可检测卡住并向前端反馈。
    """
    proc = Process(
        target=worker_generate_features,
        args=(task_id, file_path, max_timeshift, min_timeshift, tsfresh_custom_params),
    )
    proc.start()
    register_running_process(task_id, proc, service="feature_generation")
    proc.join()
    unregister_running_process(task_id)