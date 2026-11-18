import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
from typing import Optional
warnings.filterwarnings("ignore")

from autoapex.timeseries.dataset import TimeSeriesDataFrame
from autoapex.timeseries.models.abstract import AbstractTimeSeriesModel


class ErrorCompensationModel(AbstractTimeSeriesModel):
    """Residual error-compensation model using autoregressive + past covariate features.

    This model is designed to support:
    1) training with historical covariates (past covariates),
    2) prediction without future covariates.

    It learns one-step residual deltas per item with a ridge regressor and performs
    recursive multi-step forecasting.
    """

    ag_priority = 80
    _supports_past_covariates = True

    def _get_default_hyperparameters(self) -> dict:
        return {
            **super()._get_default_hyperparameters(),
            "target_lags": [1, 2, 3, 7, 14],
            "past_cov_lags": [0, 1, 2, 7],
            "ridge_alpha": 1.0,
            "min_train_samples": 40,
        }

    @property
    def allowed_hyperparameters(self) -> list[str]:
        return list(super().allowed_hyperparameters) + [
            "target_lags",
            "past_cov_lags",
            "ridge_alpha",
            "min_train_samples",
        ]

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: TimeSeriesDataFrame | None = None,
        time_limit: float | None = None,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        verbosity: int = 2,
        **kwargs,
    ) -> None:
        params = self.get_hyperparameters()
        target_lags = sorted(set(int(x) for x in params["target_lags"]))
        cov_lags = sorted(set(int(x) for x in params["past_cov_lags"]))
        ridge_alpha = float(params["ridge_alpha"])
        min_train_samples = int(params["min_train_samples"])

        df = pd.DataFrame(train_data)
        df = df.reset_index().sort_values([TimeSeriesDataFrame.ITEMID, TimeSeriesDataFrame.TIMESTAMP])
        past_covs = [c for c in self.covariate_metadata.past_covariates if c in df.columns]

        self._ec_target_lags = target_lags
        self._ec_cov_lags = cov_lags
        self._ec_past_covs = past_covs
        self._ec_item_models: dict[str, Ridge] = {}
        self._ec_item_sigma: dict[str, float] = {}
        self._ec_item_last_cov: dict[str, np.ndarray] = {}

        max_lag = max(target_lags + cov_lags) if (target_lags or cov_lags) else 1

        for item_id, gdf in df.groupby(TimeSeriesDataFrame.ITEMID, sort=False):
            gdf = gdf.sort_values(TimeSeriesDataFrame.TIMESTAMP)
            y = pd.to_numeric(gdf[self.target], errors="coerce").astype(float).values
            if np.isnan(y).any():
                y = pd.Series(y).ffill().bfill().values

            if len(y) < max(min_train_samples, max_lag + 2):
                continue

            if len(past_covs) > 0:
                cov_arr = gdf[past_covs].apply(pd.to_numeric, errors="coerce").astype(float).values
                cov_arr = pd.DataFrame(cov_arr).ffill().bfill().values
                if np.isnan(cov_arr).any():
                    cov_arr = np.nan_to_num(cov_arr, copy=False)
            else:
                cov_arr = np.empty((len(y), 0), dtype=float)

            X_rows, y_rows = [], []
            for t in range(max_lag, len(y)):
                feat = []
                for lag in target_lags:
                    feat.append(y[t - lag])
                for lag in cov_lags:
                    if cov_arr.shape[1] > 0:
                        feat.extend(cov_arr[t - lag, :].tolist())
                # Learn next-step increment to stabilize non-stationary level.
                delta = y[t] - y[t - 1]
                X_rows.append(feat)
                y_rows.append(delta)

            X = np.asarray(X_rows, dtype=float)
            y_delta = np.asarray(y_rows, dtype=float)
            if len(X) < min_train_samples:
                continue

            model = Ridge(alpha=ridge_alpha)
            model.fit(X, y_delta)
            pred_delta = model.predict(X)
            sigma = float(np.sqrt(np.mean((y_delta - pred_delta) ** 2)))
            if not np.isfinite(sigma):
                sigma = 0.0

            self._ec_item_models[str(item_id)] = model
            self._ec_item_sigma[str(item_id)] = sigma
            if cov_arr.shape[1] > 0:
                self._ec_item_last_cov[str(item_id)] = cov_arr[-1].copy()
            else:
                self._ec_item_last_cov[str(item_id)] = np.empty((0,), dtype=float)

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: TimeSeriesDataFrame | None = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        # Intentionally ignore known_covariates: model supports past covariates only.
        df = pd.DataFrame(data).reset_index().sort_values([TimeSeriesDataFrame.ITEMID, TimeSeriesDataFrame.TIMESTAMP])
        pred_index = self.get_forecast_horizon_index(data)

        out_frames: list[pd.DataFrame] = []
        item_order = list(data.item_ids)
        target_lags = self._ec_target_lags
        cov_lags = self._ec_cov_lags
        past_covs = self._ec_past_covs
        max_lag = max(target_lags + cov_lags) if (target_lags or cov_lags) else 1

        for item_id in item_order:
            gdf = df[df[TimeSeriesDataFrame.ITEMID] == item_id].sort_values(TimeSeriesDataFrame.TIMESTAMP)
            model = self._ec_item_models.get(str(item_id))

            y_hist = pd.to_numeric(gdf[self.target], errors="coerce").astype(float).values
            y_hist = pd.Series(y_hist).ffill().bfill().values
            if len(y_hist) == 0:
                y_hist = np.array([0.0], dtype=float)

            if len(past_covs) > 0 and len(gdf) > 0:
                cov_hist = gdf[past_covs].apply(pd.to_numeric, errors="coerce").astype(float).values
                cov_hist = pd.DataFrame(cov_hist).ffill().bfill().values
                if np.isnan(cov_hist).any():
                    cov_hist = np.nan_to_num(cov_hist, copy=False)
            else:
                cov_hist = np.empty((len(y_hist), 0), dtype=float)

            if model is None or len(y_hist) < max_lag + 1:
                mean_forecast = np.full(self.prediction_length, y_hist[-1], dtype=float)
                sigma = 0.0
            else:
                mean_vals = []
                sigma = self._ec_item_sigma.get(str(item_id), 0.0)
                # Recursive prediction using only historical covariates.
                # For future steps, covariates are held at the last observed value.
                for _ in range(self.prediction_length):
                    feat = []
                    cur_t = len(y_hist) - 1
                    for lag in target_lags:
                        idx = max(cur_t - lag + 1, 0)
                        idx = min(idx, len(y_hist) - 1)
                        feat.append(float(y_hist[idx]))
                    for lag in cov_lags:
                        if cov_hist.shape[1] > 0:
                            idx = max(cur_t - lag + 1, 0)
                            idx = min(idx, cov_hist.shape[0] - 1)
                            feat.extend(cov_hist[idx, :].tolist())
                    delta = float(model.predict(np.asarray(feat, dtype=float).reshape(1, -1))[0])
                    next_y = float(y_hist[-1] + delta)
                    mean_vals.append(next_y)
                    y_hist = np.append(y_hist, next_y)
                    if cov_hist.shape[1] > 0:
                        cov_hist = np.vstack([cov_hist, cov_hist[-1, :]])
                mean_forecast = np.asarray(mean_vals, dtype=float)

            frame = pd.DataFrame({"mean": mean_forecast})
            sigma_per_step = sigma * np.sqrt(np.arange(1, self.prediction_length + 1, dtype=float))
            for q in self.quantile_levels:
                z = float(torch.distributions.Normal(0, 1).icdf(torch.tensor(q)).cpu().item())
                frame[str(q)] = mean_forecast + z * sigma_per_step
            out_frames.append(frame)

        pred_df = pd.concat(out_frames, ignore_index=True)
        pred_df.index = pred_index
        return TimeSeriesDataFrame(pred_df)


# ======================
# 参数设置
# ======================
HISTORY_CSV_PATH = "/mnt/raid/forecast/shiweiyou/双视图/国网山东省电力公司_居民电量完整保留平均和日期.csv"
FUTURE_EXOG_PATH = "/mnt/raid/forecast/shiweiyou/双视图/future.csv"   # 必须包含: date + 所有外生变量列
DATE_COL = "date"
TARGET_COL = "load"

SEQ_LEN = 90
HORIZON = 31                # 直接预测未来 30 天
VAL_RATIO = 0.2
EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 2023
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STL_PERIOD = 365

SAVE_RESULT_PATH = "future_direct_forecast.csv"
SAVE_TREND_MODEL_PATH = "direct_trend_model.pth"
SAVE_SEASONAL_MODEL_PATH = "direct_seasonal_model.pth"
SAVE_RESID_MODEL_PATH = "direct_resid_model.pth"


# ======================
# 固定随机种子
# ======================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================
# STL 分解
# ======================
def safe_stl_decompose(series: np.ndarray, period: int):
    """
    对序列做 STL 分解。
    长度不足时返回 [0, 0, 原序列]，避免报错。
    """
    if len(series) < 2 * period:
        print(f"警告：序列长度 {len(series)} < 2*{period}，跳过 STL 分解。")
        zeros = np.zeros_like(series)
        return zeros, zeros, series.copy()

    stl = STL(series, period=period)
    result = stl.fit()
    return result.trend, result.seasonal, result.resid


# ======================
# 多步样本构造
# ======================
def create_direct_sequences(
    component_scaled: np.ndarray,
    load_scaled: np.ndarray,
    exog_scaled: np.ndarray,
    seq_len: int,
    horizon: int,
):
    """
    直接多步预测样本构造。

    历史输入:
      [component_t, load_t, exog_t...]

    未来已知输入:
      future_exog_{t+1:t+horizon}

    目标:
      future_component_{t+1:t+horizon}

    返回:
      X_hist:        (B, seq_len, 2+E)
      X_future_exog: (B, horizon, E)
      y:             (B, horizon)
    """
    if component_scaled.ndim != 2 or component_scaled.shape[1] != 1:
        raise ValueError("component_scaled 必须是 shape=(N,1)")

    if load_scaled.ndim != 2 or load_scaled.shape[1] != 1:
        raise ValueError("load_scaled 必须是 shape=(N,1)")

    if exog_scaled.ndim != 2:
        raise ValueError("exog_scaled 必须是二维数组")

    feat = np.concatenate([component_scaled, load_scaled, exog_scaled], axis=1)

    X_hist, X_future_exog, y = [], [], []
    n = len(component_scaled)

    for i in range(n - seq_len - horizon + 1):
        hist_block = feat[i:i + seq_len]
        future_exog_block = exog_scaled[i + seq_len:i + seq_len + horizon]
        target_block = component_scaled[i + seq_len:i + seq_len + horizon, 0]

        X_hist.append(hist_block)
        X_future_exog.append(future_exog_block)
        y.append(target_block)

    if len(X_hist) == 0:
        raise ValueError(
            f"无法构造样本。请检查数据量是否足够。当前长度={n}, SEQ_LEN={seq_len}, HORIZON={horizon}"
        )

    return np.array(X_hist), np.array(X_future_exog), np.array(y)


# ======================
# 模型：直接多步 Attention
# ======================


class DirectMAttention(nn.Module):
    def __init__(self, input_dim: int, horizon: int, exog_dim: int,
                 d_model: int = 128, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.exog_dim = exog_dim
        self.use_future_exog = exog_dim > 0

        self.q_proj = nn.Linear(1, d_model)
        self.kv_proj = nn.Linear(input_dim - 1, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        if self.use_future_exog:
            self.future_exog_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(horizon * exog_dim, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            head_in_dim = d_model * 2
        else:
            self.future_exog_proj = None
            head_in_dim = d_model

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x_hist: torch.Tensor, x_future_exog: Optional[torch.Tensor] = None):
        component = x_hist[:, :, 0:1]
        other = x_hist[:, :, 1:]

        q = self.q_proj(component)
        kv = self.kv_proj(other)

        attn_out, _ = self.attn(q, kv, kv)
        hist_repr = self.norm(q + attn_out)[:, -1, :]

        if self.use_future_exog:
            if x_future_exog is None:
                raise ValueError("当前模型需要 x_future_exog，但收到 None")
            fut_repr = self.future_exog_proj(x_future_exog)
            final_repr = torch.cat([hist_repr, fut_repr], dim=1)
        else:
            final_repr = hist_repr

        return self.head(final_repr)

# ======================
# 模型：直接多步 CNNLSTM
# ======================
class DirectCNNLSTM(nn.Module):
    def __init__(self, input_dim: int, horizon: int, exog_dim: int, dropout: float = 0.2):
        super().__init__()
        self.horizon = horizon
        self.exog_dim = exog_dim
        self.use_future_exog = exog_dim > 0

        self.conv = nn.Conv1d(input_dim, 128, kernel_size=3)
        self.bn = nn.BatchNorm1d(128)
        self.conv_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True,
            num_layers=3,
            dropout=dropout,
        )
        self.fc_dropout = nn.Dropout(dropout)

        if self.use_future_exog:
            self.future_exog_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(horizon * exog_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            head_in_dim = 64 + 64
        else:
            self.future_exog_proj = None
            head_in_dim = 64

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon),
        )

    def forward(self, x_hist: torch.Tensor, x_future_exog: Optional[torch.Tensor] = None):
        x = x_hist.permute(0, 2, 1)
        x = torch.relu(self.bn(self.conv(x)))
        x = self.conv_dropout(x)
        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)
        hist_repr = self.fc_dropout(out[:, -1, :])

        if self.use_future_exog:
            if x_future_exog is None:
                raise ValueError("当前模型需要 x_future_exog，但收到 None")
            fut_repr = self.future_exog_proj(x_future_exog)
            final_repr = torch.cat([hist_repr, fut_repr], dim=1)
        else:
            final_repr = hist_repr

        return self.head(final_repr)


# ======================
# 训练函数
# ======================
def train_direct_model(
    model: nn.Module,
    optimizer,
    scheduler,
    X_hist_train: torch.Tensor,
    X_future_train: torch.Tensor,
    y_train: torch.Tensor,
    X_hist_val: torch.Tensor,
    X_future_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    batch_size: int,
    model_name: str,
) -> None:
    criterion = nn.MSELoss()

    print(f"\n训练 {model_name} ...")
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_hist_train.size(0))

        for i in range(0, X_hist_train.size(0), batch_size):
            idx = perm[i:i + batch_size]
            bx_hist = X_hist_train[idx]
            bx_fut = X_future_train[idx]
            by = y_train[idx]

            optimizer.zero_grad()
            pred = model(bx_hist, bx_fut)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_hist_val, X_future_val)
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"{model_name} | Epoch {epoch + 1:03d}/{epochs} | Val Loss = {val_loss:.6f}")


# ======================
# Tensor 工具
# ======================
def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32).to(DEVICE)


# ======================
# 主流程
# ======================
def main():
    set_seed(SEED)

    if not os.path.exists(HISTORY_CSV_PATH):
        raise FileNotFoundError(f"历史数据文件不存在: {HISTORY_CSV_PATH}")

    if not os.path.exists(FUTURE_EXOG_PATH):
        raise FileNotFoundError(f"未来外生变量文件不存在: {FUTURE_EXOG_PATH}")

    # 1) 读取历史数据
    df = pd.read_csv(HISTORY_CSV_PATH)
    if DATE_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"历史数据必须包含列: {DATE_COL}, {TARGET_COL}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    history_df = df[df[TARGET_COL].notna()].copy().reset_index(drop=True)
    exog_cols = [c for c in history_df.columns if c not in [DATE_COL, TARGET_COL]]

    print("识别到外生变量列:", exog_cols)

    # 2) 读取未来外生变量
    future_df = pd.read_csv(FUTURE_EXOG_PATH)
    if DATE_COL not in future_df.columns:
        raise ValueError(f"未来外生变量文件必须包含列: {DATE_COL}")

    future_df[DATE_COL] = pd.to_datetime(future_df[DATE_COL])
    future_df = future_df.sort_values(DATE_COL).reset_index(drop=True)

    missing_cols = [c for c in exog_cols if c not in future_df.columns]
    if missing_cols:
        raise ValueError(f"未来外生变量缺少列: {missing_cols}")

    future_df = future_df[[DATE_COL] + exog_cols].copy()

    if len(future_df) != HORIZON:
        raise ValueError(
            f"future_exog.csv 行数必须等于 HORIZON。当前 future_df={len(future_df)}, HORIZON={HORIZON}"
        )

    # 3) Prophet 训练
    prophet_train = history_df[[DATE_COL, TARGET_COL] + exog_cols].copy()
    prophet_train = prophet_train.rename(columns={DATE_COL: "ds", TARGET_COL: "y"})

    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    for col in exog_cols:
        prophet_model.add_regressor(col)

    print("\n开始训练 Prophet ...")
    prophet_model.fit(prophet_train)

    prophet_fitted = prophet_model.predict(prophet_train)
    fitted_values = prophet_fitted["yhat"].values
    residuals_raw = history_df[TARGET_COL].values - fitted_values

    # 4) STL 分解 Prophet 残差
    trend_raw, seasonal_raw, resid_raw = safe_stl_decompose(residuals_raw, STL_PERIOD)
    trend_raw = np.nan_to_num(trend_raw)
    seasonal_raw = np.nan_to_num(seasonal_raw)
    resid_raw = np.nan_to_num(resid_raw)

    # 5) 归一化
    scaler_trend = MinMaxScaler()
    scaler_seasonal = MinMaxScaler()
    scaler_resid = MinMaxScaler()
    scaler_load = MinMaxScaler()

    trend_scaled_only = scaler_trend.fit_transform(trend_raw.reshape(-1, 1))
    seasonal_scaled_only = scaler_seasonal.fit_transform(seasonal_raw.reshape(-1, 1))
    resid_scaled_only = scaler_resid.fit_transform(resid_raw.reshape(-1, 1))
    load_scaled = scaler_load.fit_transform(history_df[TARGET_COL].values.reshape(-1, 1))

    if len(exog_cols) > 0:
        scaler_exog = MinMaxScaler()
        exog_scaled = scaler_exog.fit_transform(history_df[exog_cols].values)
    else:
        scaler_exog = None
        exog_scaled = np.empty((len(history_df), 0))

    # 6) 构造直接多步样本
    Xh_trend, Xf_trend, y_trend = create_direct_sequences(
        trend_scaled_only, load_scaled, exog_scaled, SEQ_LEN, HORIZON
    )
    Xh_seasonal, Xf_seasonal, y_seasonal = create_direct_sequences(
        seasonal_scaled_only, load_scaled, exog_scaled, SEQ_LEN, HORIZON
    )
    Xh_resid, Xf_resid, y_resid = create_direct_sequences(
        resid_scaled_only, load_scaled, exog_scaled, SEQ_LEN, HORIZON
    )

    val_size = max(1, int(len(Xh_trend) * VAL_RATIO))
    if len(Xh_trend) - val_size < 1:
        raise ValueError("训练样本太少，无法划分训练/验证集。")

    # 划分训练/验证
    Xh_trend_train, Xh_trend_val = Xh_trend[:-val_size], Xh_trend[-val_size:]
    Xf_trend_train, Xf_trend_val = Xf_trend[:-val_size], Xf_trend[-val_size:]
    y_trend_train, y_trend_val = y_trend[:-val_size], y_trend[-val_size:]

    Xh_seasonal_train, Xh_seasonal_val = Xh_seasonal[:-val_size], Xh_seasonal[-val_size:]
    Xf_seasonal_train, Xf_seasonal_val = Xf_seasonal[:-val_size], Xf_seasonal[-val_size:]
    y_seasonal_train, y_seasonal_val = y_seasonal[:-val_size], y_seasonal[-val_size:]

    Xh_resid_train, Xh_resid_val = Xh_resid[:-val_size], Xh_resid[-val_size:]
    Xf_resid_train, Xf_resid_val = Xf_resid[:-val_size], Xf_resid[-val_size:]
    y_resid_train, y_resid_val = y_resid[:-val_size], y_resid[-val_size:]

    # tensor
    Xh_trend_train = to_tensor(Xh_trend_train)
    Xf_trend_train = to_tensor(Xf_trend_train)
    y_trend_train = to_tensor(y_trend_train)
    Xh_trend_val = to_tensor(Xh_trend_val)
    Xf_trend_val = to_tensor(Xf_trend_val)
    y_trend_val = to_tensor(y_trend_val)

    Xh_seasonal_train = to_tensor(Xh_seasonal_train)
    Xf_seasonal_train = to_tensor(Xf_seasonal_train)
    y_seasonal_train = to_tensor(y_seasonal_train)
    Xh_seasonal_val = to_tensor(Xh_seasonal_val)
    Xf_seasonal_val = to_tensor(Xf_seasonal_val)
    y_seasonal_val = to_tensor(y_seasonal_val)

    Xh_resid_train = to_tensor(Xh_resid_train)
    Xf_resid_train = to_tensor(Xf_resid_train)
    y_resid_train = to_tensor(y_resid_train)
    Xh_resid_val = to_tensor(Xh_resid_val)
    Xf_resid_val = to_tensor(Xf_resid_val)
    y_resid_val = to_tensor(y_resid_val)

    # 7) 模型初始化
    exog_dim = len(exog_cols)

    model_trend = DirectMAttention(
        input_dim=Xh_trend_train.shape[2],
        horizon=HORIZON,
        exog_dim=exog_dim,
    ).to(DEVICE)

    model_seasonal = DirectMAttention(
        input_dim=Xh_seasonal_train.shape[2],
        horizon=HORIZON,
        exog_dim=exog_dim,
    ).to(DEVICE)

    model_resid = DirectCNNLSTM(
        input_dim=Xh_resid_train.shape[2],
        horizon=HORIZON,
        exog_dim=exog_dim,
    ).to(DEVICE)

    optimizer_trend = torch.optim.AdamW(model_trend.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer_seasonal = torch.optim.AdamW(model_seasonal.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer_resid = torch.optim.AdamW(model_resid.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler_trend = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_trend, patience=3)
    scheduler_seasonal = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_seasonal, patience=3)
    scheduler_resid = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_resid, patience=3)

    # 8) 训练
    train_direct_model(
        model_trend, optimizer_trend, scheduler_trend,
        Xh_trend_train, Xf_trend_train, y_trend_train,
        Xh_trend_val, Xf_trend_val, y_trend_val,
        EPOCHS, BATCH_SIZE, "趋势模型"
    )

    train_direct_model(
        model_seasonal, optimizer_seasonal, scheduler_seasonal,
        Xh_seasonal_train, Xf_seasonal_train, y_seasonal_train,
        Xh_seasonal_val, Xf_seasonal_val, y_seasonal_val,
        EPOCHS, BATCH_SIZE, "季节模型"
    )

    train_direct_model(
        model_resid, optimizer_resid, scheduler_resid,
        Xh_resid_train, Xf_resid_train, y_resid_train,
        Xh_resid_val, Xf_resid_val, y_resid_val,
        EPOCHS, BATCH_SIZE, "余项模型"
    )

    # 9) 保存模型
    torch.save(model_trend.state_dict(), SAVE_TREND_MODEL_PATH)
    torch.save(model_seasonal.state_dict(), SAVE_SEASONAL_MODEL_PATH)
    torch.save(model_resid.state_dict(), SAVE_RESID_MODEL_PATH)

    # 10) Prophet 直接预测未来 HORIZON 天
    prophet_future = future_df.rename(columns={DATE_COL: "ds"})
    prophet_pred = prophet_model.predict(prophet_future)
    prophet_pred_values = prophet_pred["yhat"].values  # (HORIZON,)

    # 11) 未来外生变量归一化
    if len(exog_cols) > 0:
        future_exog_scaled = scaler_exog.transform(future_df[exog_cols].values)
    else:
        future_exog_scaled = np.empty((HORIZON, 0))

    # 12) 取历史最后窗口，直接一次输出未来 HORIZON 天
    trend_feat = np.concatenate([trend_scaled_only, load_scaled, exog_scaled], axis=1)
    seasonal_feat = np.concatenate([seasonal_scaled_only, load_scaled, exog_scaled], axis=1)
    resid_feat = np.concatenate([resid_scaled_only, load_scaled, exog_scaled], axis=1)

    trend_hist_last = trend_feat[-SEQ_LEN:]
    seasonal_hist_last = seasonal_feat[-SEQ_LEN:]
    resid_hist_last = resid_feat[-SEQ_LEN:]

    trend_hist_input = to_tensor(trend_hist_last).unsqueeze(0)
    seasonal_hist_input = to_tensor(seasonal_hist_last).unsqueeze(0)
    resid_hist_input = to_tensor(resid_hist_last).unsqueeze(0)
    future_exog_input = to_tensor(future_exog_scaled).unsqueeze(0)

    model_trend.eval()
    model_seasonal.eval()
    model_resid.eval()

    with torch.no_grad():
        trend_pred_scaled = model_trend(trend_hist_input, future_exog_input).cpu().numpy().reshape(-1, 1)
        seasonal_pred_scaled = model_seasonal(seasonal_hist_input, future_exog_input).cpu().numpy().reshape(-1, 1)
        resid_pred_scaled = model_resid(resid_hist_input, future_exog_input).cpu().numpy().reshape(-1, 1)

    # 13) 反归一化
    future_trend_raw = scaler_trend.inverse_transform(trend_pred_scaled).ravel()
    future_seasonal_raw = scaler_seasonal.inverse_transform(seasonal_pred_scaled).ravel()
    future_resid_raw = scaler_resid.inverse_transform(resid_pred_scaled).ravel()

    # 14) 最终预测
    final_pred = prophet_pred_values + future_trend_raw + future_seasonal_raw + future_resid_raw

    future_result_df = pd.DataFrame({
        "date": future_df[DATE_COL].values,
        "prophet_pred": prophet_pred_values,
        "trend_pred_raw": future_trend_raw,
        "seasonal_pred_raw": future_seasonal_raw,
        "resid_pred_raw": future_resid_raw,
        "final_pred": final_pred,
    })

    future_result_df.to_csv(SAVE_RESULT_PATH, index=False)

    print("\n====================")
    print("直接多步未来预测完成")
    print("====================")
    print(f"历史样本数: {len(history_df)}")
    print(f"未来预测步数: {len(future_df)}")
    print(f"结果文件: {SAVE_RESULT_PATH}")
    print(f"趋势模型: {SAVE_TREND_MODEL_PATH}")
    print(f"季节模型: {SAVE_SEASONAL_MODEL_PATH}")
    print(f"余项模型: {SAVE_RESID_MODEL_PATH}")


if __name__ == "__main__":
    main()