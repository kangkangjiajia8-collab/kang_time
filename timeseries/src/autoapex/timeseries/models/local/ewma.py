"""
EWMA (Exponentially Weighted Moving Average) 模型。

对每条时间序列用指数加权移动平均得到平滑水平，预测期为该水平的常数外推；
分位数由一步预测残差的分布估计。
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from autoapex.timeseries.models.local.abstract_local_model import AbstractLocalModel


class EWMAModel(AbstractLocalModel):
    """
    指数加权移动平均（EWMA）预测模型。

    使用 EWMA 对历史序列平滑得到水平估计 S_t = alpha * x_t + (1 - alpha) * S_{t-1}，
    预测期内所有步的点预测为最后一个 S_t；分位数由一步预测残差的标准差估计。

    Other Parameters
    ----------------
    alpha : float, default = 0.3
        平滑系数，范围 (0, 1]。越大越重视近期观测。
        也可通过 span 间接指定：alpha = 2 / (span + 1)。
    span : int or None, default = None
        若提供，则忽略 alpha，使用 alpha = 2 / (span + 1)。例如 span=9 对应 alpha≈0.2。
    forecast_style : str, default = "constant"
        - "constant"：未来所有步的点预测都等于最后一个 EWMA 水平（经典做法，每步预测相同）。
        - "decay_to_level"：第 1 步 = 最后一个观测值，之后每步向 EWMA 水平衰减，
          即 step_1 = x_T，step_h = alpha * step_{h-1} + (1-alpha) * S_T，预测随步长变化。
    seasonal_period : int or None, default = None
        用于回退模型与季节性的推断，见 AbstractLocalModel。
    n_jobs : int or float, default = 见基类
        并行预测的 CPU 核心数。
    """

    ag_priority = 100
    allowed_local_model_args = ["seasonal_period", "alpha", "span", "forecast_style"]

    def _get_default_hyperparameters(self) -> dict:
        return {
            **super()._get_default_hyperparameters(),
            "alpha": 0.3,
            "span": None,
            "forecast_style": "constant",
        }

    def _predict_with_local_model(
        self,
        time_series: pd.Series,
        local_model_args: dict,
    ) -> pd.DataFrame:
        alpha = local_model_args.get("alpha", 0.3)
        span = local_model_args.get("span", None)
        if span is not None:
            alpha = 2.0 / (float(span) + 1.0)
        alpha = np.clip(float(alpha), 1e-6, 1.0)
        forecast_style = local_model_args.get("forecast_style", "constant") or "constant"

        # 清洗：转 float、去 inf、前向/后向填 NaN
        clean = pd.to_numeric(time_series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        clean = clean.ffill().bfill()
        arr = clean.dropna().values.ravel()

        if len(arr) == 0:
            last_val = 0.0
            last_obs = 0.0
            sigma = 0.0
        else:
            # EWMA: S_0 = x_0, S_t = alpha * x_t + (1-alpha) * S_{t-1}
            s = np.empty_like(arr, dtype=float)
            s[0] = arr[0]
            for t in range(1, len(arr)):
                s[t] = alpha * arr[t] + (1.0 - alpha) * s[t - 1]
            last_val = float(s[-1])
            last_obs = float(arr[-1])

            # 一步预测残差 r_t = x_t - S_{t-1}，用于分位数
            if len(arr) > 1:
                residuals = arr[1:] - s[:-1]
                residuals = residuals[np.isfinite(residuals)]
                sigma = np.sqrt(np.mean(residuals ** 2)) if len(residuals) > 0 else 0.0
            else:
                sigma = 0.0

        n_steps = self.prediction_length
        if forecast_style == "decay_to_level":
            # 第 1 步 = 最后观测值，之后每步向 EWMA 水平衰减，预测随步长变化
            mean_forecast = np.empty(n_steps, dtype=float)
            mean_forecast[0] = last_obs
            for h in range(1, n_steps):
                mean_forecast[h] = alpha * mean_forecast[h - 1] + (1.0 - alpha) * last_val
        else:
            # 经典：所有步点预测 = 最后 EWMA 值（每步相同）
            mean_forecast = np.full(n_steps, last_val)
        if sigma <= 0 or not np.isfinite(sigma):
            sigma = 0.0

        sigma_per_step = sigma * np.sqrt(np.arange(1, n_steps + 1, dtype=float))
        forecast = {"mean": mean_forecast}
        for q in self.quantile_levels:
            forecast[str(q)] = mean_forecast + norm.ppf(q) * sigma_per_step

        return pd.DataFrame(forecast)

    def _more_tags(self) -> dict:
        return {"allow_nan": True}
