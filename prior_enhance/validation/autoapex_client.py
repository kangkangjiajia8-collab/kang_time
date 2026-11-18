"""AutoApex serve 训练与推理客户端（/api/train, /api/inference/v1）。"""

from __future__ import annotations

import io
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import requests

from prior_enhance.config import AutoApexConfig, DataConfig
from prior_enhance.recipe.autoapex_adapter import (
    apply_sample_window,
    infer_best_model_name,
    recipe_to_train_payload,
)
from prior_enhance.recipe.schema import ForecastingRecipe
from prior_enhance.telemetry.run_logger import log

PHASE = "AUTOAPEX"

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# prior_enhance 频率码 → serve / AutoGluon 频率码
_SERVE_FREQ_MAP = {
    "h": "H",
    "H": "H",
    "5T": "5T",
    "15T": "15T",
    "D": "D",
    "W-SUN": "W-SUN",
    "M": "M",
}


def _resolve_serve_freq(
    df: pd.DataFrame,
    dcfg: DataConfig,
    recipe: ForecastingRecipe,
) -> str:
    """从 recipe / 配置 / 时间戳推断 serve 可接受的 freq（如 H）。"""
    raw = (recipe.freq or dcfg.freq or "").strip()
    if not raw:
        from prior_enhance.context.frequency import infer_freq_code

        raw = infer_freq_code(pd.to_datetime(df[dcfg.time_col]))
    mapped = _SERVE_FREQ_MAP.get(raw, raw)
    if not mapped or not str(mapped).strip():
        return "H"
    return str(mapped).strip()


def _form_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    return str(v)


def _ensure_infer_columns(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    out = df.copy()
    if cfg.id_col not in out.columns:
        out[cfg.id_col] = cfg.default_item_id
    return out.sort_values([cfg.id_col, cfg.time_col]).reset_index(drop=True)


def _extract_pred_series(
    pred_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: DataConfig,
) -> pd.Series:
    """将 serve 预测 CSV 对齐到 val/oos 时间戳。"""
    pred = pred_df.copy()
    val_col = None
    for col in ("mean", "median", cfg.target_col, "value"):
        if col in pred.columns:
            val_col = col
            break
    if val_col is None:
        raise ValueError(f"预测结果缺少目标列: {list(pred.columns)}")

    pred[cfg.time_col] = pd.to_datetime(pred[cfg.time_col])
    val = val_df.copy()
    val[cfg.time_col] = pd.to_datetime(val[cfg.time_col])

    merged = val[[cfg.time_col]].merge(
        pred[[cfg.time_col, val_col]],
        on=cfg.time_col,
        how="left",
    )
    series = merged[val_col]
    if series.isna().any():
        # 回退：按顺序取末尾 len(val) 行
        tail = pred.sort_values(cfg.time_col).tail(len(val_df))
        if len(tail) == len(val_df):
            series = tail[val_col].reset_index(drop=True)
        else:
            raise ValueError("预测长度与验证集不匹配")
    return series.astype(float)


@dataclass
class TrainResult:
    train_task_id: str
    model_id: str
    wall_time_sec: float


class AutoApexClient:
    """封装 serve 阻塞式训练与 inference/v1。"""

    def __init__(self, cfg: Optional[AutoApexConfig] = None):
        self.cfg = cfg or AutoApexConfig()
        self.base_url = self.cfg.base_url.rstrip("/")

    def check_health(self) -> bool:
        root = self.base_url.replace("/api", "")
        try:
            return requests.get(f"{root}/docs", timeout=10).status_code == 200
        except requests.RequestException:
            return False

    def train(
        self,
        recipe: ForecastingRecipe,
        train_df: pd.DataFrame,
        dcfg: DataConfig,
        *,
        task_id: Optional[str] = None,
    ) -> TrainResult:
        task_id = task_id or f"prior_{recipe.recipe_id}_{uuid.uuid4().hex[:8]}"
        windowed = _ensure_infer_columns(apply_sample_window(train_df, recipe), dcfg)
        freq = _resolve_serve_freq(windowed, dcfg, recipe)
        exec_recipe = recipe.model_copy(update={"freq": freq})
        payload = recipe_to_train_payload(exec_recipe, task_id, target=dcfg.target_col)

        buf = io.BytesIO()
        windowed.to_csv(buf, index=False)
        buf.seek(0)

        form = {k: _form_value(v) for k, v in payload.items()}
        t0 = time.time()
        resp = requests.post(
            f"{self.base_url}/train",
            files={"file": ("train.csv", buf, "text/csv")},
            data=form,
            timeout=self.cfg.train_timeout,
        )
        wall = time.time() - t0

        if resp.status_code == 409:
            raise RuntimeError(f"task_id 冲突: {task_id}")
        if resp.status_code != 200:
            raise RuntimeError(f"Train HTTP {resp.status_code}: {resp.text[:800]}")
        if "zip" not in (resp.headers.get("content-type") or ""):
            raise RuntimeError(f"Train 响应非 zip: {resp.text[:200]}")

        model_id = self.fetch_model_id(task_id)
        log.log_output("autoapex_train", {
            "task_id": task_id,
            "model_id": model_id,
            "model": recipe.M.name,
            "wall_time_sec": round(wall, 2),
        }, phase=PHASE)
        return TrainResult(task_id, model_id, wall)

    def fetch_model_id(self, train_task_id: str) -> str:
        resp = requests.get(
            f"{self.base_url}/task_status",
            params={"task_id": train_task_id},
            timeout=30,
        )
        resp.raise_for_status()
        inner = (resp.json().get("data") or {}).get("data") or {}
        model_id = (inner.get("model_id") or "").strip()
        if not _UUID_RE.match(model_id):
            raise ValueError(f"无效 model_id (task={train_task_id}): {model_id!r}")
        return model_id

    def predict(
        self,
        recipe: ForecastingRecipe,
        model_id: str,
        history_df: pd.DataFrame,
        target_df: pd.DataFrame,
        dcfg: DataConfig,
        *,
        infer_task_id: Optional[str] = None,
    ) -> pd.Series:
        """在 history 上加载模型，预测 target_df 对应 horizon。"""
        infer_task_id = infer_task_id or f"infer_{uuid.uuid4().hex[:12]}"
        best_model = infer_best_model_name(recipe)
        hist = _ensure_infer_columns(history_df, dcfg)

        buf = io.BytesIO()
        hist.to_csv(buf, index=False)
        buf.seek(0)

        resp = requests.post(
            f"{self.base_url}/inference/v1",
            files={"data_csv": ("infer.csv", buf, "text/csv")},
            data={
                "task_id": infer_task_id,
                "best_model": best_model,
                "model_id": model_id,
            },
            timeout=self.cfg.infer_timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Infer HTTP {resp.status_code}: {resp.text[:800]}")

        pred_df = pd.read_csv(io.BytesIO(resp.content))
        return _extract_pred_series(pred_df, target_df, dcfg)

    def train_and_predict(
        self,
        recipe: ForecastingRecipe,
        train_df: pd.DataFrame,
        target_df: pd.DataFrame,
        dcfg: DataConfig,
    ) -> Tuple[pd.Series, TrainResult]:
        """训练并在 target 时间范围上产出预测（prediction_length 自动对齐）。"""
        windowed = _ensure_infer_columns(apply_sample_window(train_df, recipe), dcfg)
        freq = _resolve_serve_freq(windowed, dcfg, recipe)
        exec_recipe = recipe.model_copy(
            update={"prediction_length": len(target_df), "freq": freq},
        )
        tr = self.train(exec_recipe, train_df, dcfg)
        pred = self.predict(exec_recipe, tr.model_id, train_df, target_df, dcfg)
        return pred, tr


_client: Optional[AutoApexClient] = None


def get_autoapex_client(cfg: Optional[AutoApexConfig] = None) -> AutoApexClient:
    global _client
    if cfg is not None:
        return AutoApexClient(cfg)
    if _client is None:
        _client = AutoApexClient()
    return _client
