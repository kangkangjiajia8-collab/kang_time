# import os
# import time
# import warnings
# import numpy as np
# import pandas as pd  # <---【修复1】必须放在文件最顶部，解决 UnboundLocalError
# import torch
# from torch.utils.data import DataLoader, Dataset

# # AutoApex 基础类
# from autoapex.timeseries.models.abstract import AbstractTimeSeriesModel
# from autoapex.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame

# # 假设你的 TimeXer 模型和工具类在这个路径下
# # 请根据你实际的项目结构调整这些 import
# try:
#     from .timexer_lib.utils.timefeatures import time_features
#     # 假设你的模型定义在这里，如果类名不同请修改
#     from .timexer_lib.models import TimeXer as DETMModel 
# except ImportError:
#     # 仅作为示例，如果找不到库则忽略，实际运行时会报错
#     pass

# class TimeSeriesDataset(Dataset):
#     def __init__(self, data, df_stamp, freq='h', flag='train', size=None,
#                  features='S', target='value', scale=True, timeenc=0, freq_map=None):
#         # 初始化参数
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]
#         self.flag = flag
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq

#         # -----------------------------------------------------------
#         # 【修复2】健壮的时间特征处理
#         # 确保传入 time_features 的是 DatetimeIndex 或 Series，而不是 DataFrame
#         # -----------------------------------------------------------
#         if isinstance(df_stamp, pd.DataFrame):
#             if 'date' in df_stamp.columns:
#                 time_vals = pd.to_datetime(df_stamp['date'].values)
#             elif isinstance(df_stamp.index, pd.DatetimeIndex):
#                 time_vals = df_stamp.index
#             else:
#                 # 尝试取第一列
#                 time_vals = pd.to_datetime(df_stamp.iloc[:, 0].values)
#         else:
#             time_vals = pd.to_datetime(df_stamp)

#         # 生成时间特征 (dayofweek, month, etc.)
#         self.data_stamp = time_features(time_vals, freq=freq)
        
#         # 处理实际数据 X 和 Y
#         # 注意：这里假设 data 已经是 numpy array 或者处理好的 DataFrame
#         if isinstance(data, pd.DataFrame):
#             self.data_x = data.values
#             self.data_y = data.values
#         else:
#             self.data_x = data
#             self.data_y = data

#     def __getitem__(self, index):
#         # 标准的时间序列切片逻辑
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1


# class DETMModel(AbstractTimeSeriesModel):
#     def __init__(self, freq=None, prediction_length: int = 24, path: str = None, 
#                  name: str = "TimeXer", eval_metric: str = None, 
#                  hyperparameters: dict = None, **kwargs):
#         """
#         初始化模型
#         """
#         # 修正 hyperparameters 默认值
#         hyperparameters = hyperparameters if hyperparameters is not None else {}
        
#         super().__init__(
#             path=path,
#             freq=freq,
#             prediction_length=prediction_length,
#             name=name,
#             eval_metric=eval_metric,
#             hyperparameters=hyperparameters,
#             **kwargs
#         )
        
#         # 从超参数中读取模型配置（基类用 get_hyperparameters()，无 params 属性）
#         params = self.get_hyperparameters()
#         self.seq_len = params.get('seq_len', 96)
#         self.label_len = params.get('label_len', 48)
#         self.pred_len = prediction_length  # 预测长度通常由 AutoApex 指定
#         self.batch_size = params.get('batch_size', 32)
#         self.features = params.get('features', 'S')  # S:Univariate, M:Multivariate
#         self.model = None  # 实际的 torch 模型

#     def _fit(self, train_data, val_data=None, time_limit=None, **kwargs):
#         """
#         训练入口函数
#         """
#         # -----------------------------------------------------------
#         # 【修复3】数据预处理：将 AutoApex 的 MultiIndex 数据转换为模型可用的格式
#         # -----------------------------------------------------------
        
#         # 1. 打印调试信息（可选）
#         # print(f"DEBUG: Input train_data shape: {train_data.shape}")

#         # 2. 重置索引，将 item_id 和 timestamp 变为普通列
#         train_df = train_data.reset_index()

#         # 3. 提取时间戳列
#         if 'timestamp' in train_df.columns:
#             timestamps = train_df['timestamp']
#         else:
#             # 兜底：寻找 datetime 类型的列
#             timestamps = train_df.select_dtypes(include=[np.datetime64, 'datetime']).iloc[:, 0]

#         # 4. 准备 df_stamp (用于生成时间特征)
#         # 这里解决了 'UnboundLocalError: local variable 'pd' referenced before assignment'
#         # 因为 pd 已经在文件顶部导入了
#         df_stamp = pd.DataFrame({"date": timestamps})

#         # 5. 准备目标数据 (Target)
#         # 确保使用用户指定的 target 列 (通常是 'value')
#         target_col = self.target if self.target is not None else 'value'
#         if target_col not in train_df.columns:
#             raise ValueError(f"Target column '{target_col}' not found in train_data")
            
#         data_values = train_df[[target_col]].values # 转为 numpy array
        
#         # -----------------------------------------------------------
#         # 实例化 Dataset 和 DataLoader
#         # -----------------------------------------------------------
#         train_dataset = TimeSeriesDataset(
#             data=data_values,
#             df_stamp=df_stamp,
#             freq=self.freq,
#             flag='train',
#             size=[self.seq_len, self.label_len, self.pred_len],
#             features=self.features,
#             target=target_col
#         )

#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=0, # 避免多进程报错，生产环境可调大
#             drop_last=True
#         )

#         # 初始化你的模型 (这里需要根据你的 DETM 实际参数调整)
#         # self.model = DETM(
#         #     self.seq_len, self.label_len, self.pred_len, ...
#         # ).float()
        
#         # if torch.cuda.is_available():
#         #     self.model.cuda()

#         # 模拟训练循环 (你需要填入实际的训练代码)
#         print(f"Training TimeXer with {len(train_dataset)} samples...")
#         # for epoch in range(self.get_hyperparameters().get('epochs', 10)):
#         #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
#         #         optimizer.zero_grad()
#         #         outputs = self.model(batch_x, batch_x_mark)
#         #         loss = criterion(outputs, batch_y)
#         #         loss.backward()
#         #         optimizer.step()
        
#         # 训练完成后返回自身
#         return self

#     def _predict(
#         self,
#         data: TimeSeriesDataFrame,
#         known_covariates: TimeSeriesDataFrame | None = None,
#         **kwargs,
#     ) -> TimeSeriesDataFrame:
#         """
#         抽象方法：在预处理后的 data 上做预测，返回带 mean 与分位数列的 TimeSeriesDataFrame。
#         基类 predict() 会先做缩放/预处理，再调用本方法，再做逆变换。
#         """
#         forecast_index = self.get_forecast_horizon_index(data)
#         item_ids = data.item_ids
#         indptr = data.get_indptr()
#         target_col = self.target if self.target is not None else "value"

#         rows = []
#         for i in range(len(indptr) - 1):
#             start, end = indptr[i], indptr[i + 1]
#             ts = data.iloc[start:end][target_col]
#             last_val = float(ts.iloc[-1]) if len(ts) > 0 and pd.notna(ts.iloc[-1]) else 0.0
#             n_steps = self.prediction_length
#             # 每条约 n_steps 行：mean + 各分位数
#             row_dict = {"mean": last_val}
#             for q in self.quantile_levels:
#                 row_dict[str(q)] = last_val
#             for _ in range(n_steps):
#                 rows.append(row_dict.copy())

#         predictions_df = pd.DataFrame(rows)
#         if len(predictions_df) > 0:
#             predictions_df.index = forecast_index
#         return TimeSeriesDataFrame(predictions_df)

    
import os
import sys
import time
import types
import warnings
import numpy as np
import pandas as pd  # <---【修复1】必须放在文件最顶部，解决 UnboundLocalError
import torch
from torch.utils.data import DataLoader, Dataset

# AutoApex 基础类
from autoapex.timeseries.models.abstract import AbstractTimeSeriesModel
from autoapex.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame

# 假设你的 TimeXer 模型和工具类在这个路径下
# 请根据你实际的项目结构调整这些 import
try:
    from .timexer_lib.utils.timefeatures import time_features
    # 假设你的模型定义在这里，如果类名不同请修改
    from .timexer_lib.models import TimeXer as DETMModel 
except ImportError:
    # 仅作为示例，如果找不到库则忽略，实际运行时会报错
    pass

class TimeSeriesDataset(Dataset):
    def __init__(self, data, df_stamp, freq='h', flag='train', size=None,
                 features='S', target='value', scale=True, timeenc=0, freq_map=None):
        # 初始化参数
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # -----------------------------------------------------------
        # 【修复2】健壮的时间特征处理
        # 确保传入 time_features 的是 DatetimeIndex 或 Series，而不是 DataFrame
        # -----------------------------------------------------------
        if isinstance(df_stamp, pd.DataFrame):
            if 'date' in df_stamp.columns:
                time_vals = pd.to_datetime(df_stamp['date'].values)
            elif isinstance(df_stamp.index, pd.DatetimeIndex):
                time_vals = df_stamp.index
            else:
                # 尝试取第一列
                time_vals = pd.to_datetime(df_stamp.iloc[:, 0].values)
        else:
            time_vals = pd.to_datetime(df_stamp)

        # 生成时间特征 (dayofweek, month, etc.)
        self.data_stamp = time_features(time_vals, freq=freq)
        
        # 处理实际数据 X 和 Y
        # 注意：这里假设 data 已经是 numpy array 或者处理好的 DataFrame
        if isinstance(data, pd.DataFrame):
            self.data_x = data.values
            self.data_y = data.values
        else:
            self.data_x = data
            self.data_y = data

    def __getitem__(self, index):
        # 标准的时间序列切片逻辑
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class DETMModel(AbstractTimeSeriesModel):
    def __init__(self, freq=None, prediction_length: int = 24, path: str = None, 
                 name: str = "TimeXer", eval_metric: str = None, 
                 hyperparameters: dict = None, **kwargs):
        """
        初始化模型
        """
        # 修正 hyperparameters 默认值
        hyperparameters = hyperparameters if hyperparameters is not None else {}
        
        super().__init__(
            path=path,
            freq=freq,
            prediction_length=prediction_length,
            name=name,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            **kwargs
        )
        
        # 从超参数中读取模型配置（基类用 get_hyperparameters()，无 params 属性）
        params = self.get_hyperparameters()
        self.seq_len = params.get('seq_len', 96)
        self.label_len = params.get('label_len', 48)
        self.pred_len = prediction_length  # 预测长度通常由 AutoApex 指定
        self.batch_size = params.get('batch_size', 32)
        self.features = params.get('features', 'S')  # S:Univariate, M:Multivariate
        self.use_orthogonal = params.get('use_orthogonal', False)  # 是否启用正交法模块
        self.model = None  # 实际的 torch 模型

    def _get_default_hyperparameters(self) -> dict:
        """默认超参，包含 use_orthogonal 及 TimeXer 所需字段。"""
        return {
            "seq_len": 96,
            "label_len": 48,
            "batch_size": 32,
            "features": "S",
            "use_orthogonal": False,
            # TimeXer 内部 configs 的其余默认值（构建 configs 时用）
            "task_name": "long_term_forecast",
            "use_norm": True,
            "patch_len": 16,
            "enc_in": 1,
            "d_model": 512,
            "dropout": 0.1,
            "factor": 3,
            "n_heads": 8,
            "d_ff": 512,
            "activation": "gelu",
            "e_layers": 2,
            "embed": "timeF",
            "freq": "h",
        }

    @property
    def allowed_hyperparameters(self) -> list:
        """允许的超参列表，包含 use_orthogonal 及 TimeXer 相关参数。"""
        return list(super().allowed_hyperparameters) + [
            "seq_len",
            "label_len",
            "batch_size",
            "features",
            "use_orthogonal",
            "task_name",
            "use_norm",
            "patch_len",
            "enc_in",
            "d_model",
            "dropout",
            "factor",
            "n_heads",
            "d_ff",
            "activation",
            "e_layers",
            "embed",
            "freq",
        ]

    def _build_timexer_configs(self):
        """从当前超参构建 TimeXer Model 所需的 configs，保证 use_orthogonal 传入。"""
        p = self.get_hyperparameters()
        configs = types.SimpleNamespace(
            task_name=p.get("task_name", "long_term_forecast"),
            features=p.get("features", "S"),
            seq_len=p.get("seq_len", 96),
            pred_len=self.prediction_length,
            use_norm=p.get("use_norm", True),
            patch_len=p.get("patch_len", 16),
            enc_in=p.get("enc_in", 1),
            d_model=p.get("d_model", 512),
            dropout=p.get("dropout", 0.1),
            factor=p.get("factor", 3),
            n_heads=p.get("n_heads", 8),
            d_ff=p.get("d_ff", 512),
            activation=p.get("activation", "gelu"),
            e_layers=p.get("e_layers", 2),
            embed=p.get("embed", "timeF"),
            freq=self.freq or "h",
            use_orthogonal=p.get("use_orthogonal", False),
        )
        return configs

    def _fit(self, train_data, val_data=None, time_limit=None, **kwargs):
        """
        训练入口函数
        """
        # -----------------------------------------------------------
        # 【修复3】数据预处理：将 AutoApex 的 MultiIndex 数据转换为模型可用的格式
        # -----------------------------------------------------------
        
        # 1. 打印调试信息（可选）
        # print(f"DEBUG: Input train_data shape: {train_data.shape}")

        # 2. 重置索引，将 item_id 和 timestamp 变为普通列
        train_df = train_data.reset_index()

        # 3. 提取时间戳列
        if 'timestamp' in train_df.columns:
            timestamps = train_df['timestamp']
        else:
            # 兜底：寻找 datetime 类型的列
            timestamps = train_df.select_dtypes(include=[np.datetime64, 'datetime']).iloc[:, 0]

        # 4. 准备 df_stamp (用于生成时间特征)
        # 这里解决了 'UnboundLocalError: local variable 'pd' referenced before assignment'
        # 因为 pd 已经在文件顶部导入了
        df_stamp = pd.DataFrame({"date": timestamps})

        # 5. 准备目标数据 (Target)
        # 确保使用用户指定的 target 列 (通常是 'value')
        target_col = self.target if self.target is not None else 'value'
        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in train_data")
            
        data_values = train_df[[target_col]].values # 转为 numpy array
        
        # -----------------------------------------------------------
        # 实例化 Dataset 和 DataLoader
        # -----------------------------------------------------------
        train_dataset = TimeSeriesDataset(
            data=data_values,
            df_stamp=df_stamp,
            freq=self.freq,
            flag='train',
            size=[self.seq_len, self.label_len, self.pred_len],
            features=self.features,
            target=target_col
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0, # 避免多进程报错，生产环境可调大
            drop_last=True
        )

        # 从超参构建 configs（含 use_orthogonal），并初始化 TimeXer Model
        # TimeXer 内部使用 from layers / models / utils，需把 timexer_lib 加入 path 才能解析
        _here = os.path.dirname(os.path.abspath(__file__))
        _timexer_root = os.path.join(_here, "timexer_lib")
        if _timexer_root not in sys.path:
            sys.path.insert(0, _timexer_root)
        from .timexer_lib.models.TimeXer import Model as TimeXerModel
        configs = self._build_timexer_configs()
        if self.features in ("M", "MS") and data_values.ndim > 1 and data_values.shape[1] > 1:
            configs.enc_in = data_values.shape[1]
        self.model = TimeXerModel(configs)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # 模拟训练循环 (你需要填入实际的训练代码)
        print(f"Training TimeXer with {len(train_dataset)} samples (use_orthogonal={getattr(configs, 'use_orthogonal', False)})...")
        # for epoch in range(self.get_hyperparameters().get('epochs', 10)):
        #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        #         optimizer.zero_grad()
        #         outputs = self.model(batch_x, batch_x_mark)
        #         loss = criterion(outputs, batch_y)
        #         loss.backward()
        #         optimizer.step()
        
        # 训练完成后返回自身
        return self

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame | None = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        """
        使用 TimeXer 模型对每条序列做前向推理，返回 pred_len 步的预测（每步可不同）。
        若序列过短或模型推理失败，则回退为最后观测值常数预测。
        """
        from .timexer_lib.utils.timefeatures import time_features

        forecast_index = self.get_forecast_horizon_index(data)
        indptr = data.get_indptr()
        target_col = self.target if self.target is not None else "value"
        n_steps = self.prediction_length
        device = next(self.model.parameters()).device
        seq_len = self.seq_len
        label_len = self.label_len

        # 时间特征维度：用于构造 x_mark
        _sample_dates = pd.DatetimeIndex([pd.Timestamp("2020-01-01")])
        n_time_feat = time_features(_sample_dates, freq=self.freq or "h").shape[0]

        rows = []
        for i in range(len(indptr) - 1):
            start, end = indptr[i], indptr[i + 1]
            block = data.iloc[start:end]
            ts = block[target_col]
            if hasattr(block.index, "get_level_values"):
                timestamps = block.index.get_level_values(TimeSeriesDataFrame.TIMESTAMP)
            else:
                timestamps = block.index

            last_val = float(ts.iloc[-1]) if len(ts) > 0 and pd.notna(ts.iloc[-1]) else 0.0
            use_model = (
                self.model is not None
                and len(ts) >= seq_len
                and ts.notna().sum() >= seq_len
            )

            if use_model:
                try:
                    # 取最后 seq_len 个点
                    tail = ts.iloc[-seq_len:]
                    stamp_tail = timestamps[-seq_len:]
                    vals = pd.to_numeric(tail, errors="coerce").ffill().bfill().values
                    if vals.size < seq_len or np.any(~np.isfinite(vals)):
                        use_model = False
                    else:
                        x_enc = torch.from_numpy(vals.astype(np.float32)).reshape(1, seq_len, -1)
                        if x_enc.shape[2] == 1 and self.features == "S":
                            pass
                        x_enc = x_enc.to(device)

                        dates = pd.to_datetime(stamp_tail)
                        if not hasattr(dates, "to_pydatetime"):
                            dates = pd.DatetimeIndex(dates)
                        mark = time_features(dates, freq=self.freq or "h")
                        x_mark_enc = torch.from_numpy(mark.T.astype(np.float32)).unsqueeze(0).to(device)

                        dec_len = label_len + n_steps
                        x_dec = torch.zeros(1, dec_len, x_enc.shape[2], device=device, dtype=x_enc.dtype)
                        x_mark_dec = torch.zeros(1, dec_len, n_time_feat, device=device, dtype=x_mark_enc.dtype)

                        with torch.no_grad():
                            out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                        # out: [1, pred_len, D]
                        pred_vals = out.cpu().numpy().ravel()
                        pred_vals = pred_vals[:n_steps]
                        if pred_vals.size < n_steps:
                            pred_vals = np.resize(pred_vals, n_steps)
                        if not np.isfinite(pred_vals).all():
                            use_model = False
                except Exception:
                    use_model = False

            if not use_model:
                pred_vals = np.full(n_steps, last_val)

            for k in range(n_steps):
                v = float(pred_vals[k])
                row_dict = {"mean": v}
                for q in self.quantile_levels:
                    row_dict[str(q)] = v
                rows.append(row_dict)

        predictions_df = pd.DataFrame(rows)
        if len(predictions_df) > 0:
            predictions_df.index = forecast_index
        return TimeSeriesDataFrame(predictions_df)

    
