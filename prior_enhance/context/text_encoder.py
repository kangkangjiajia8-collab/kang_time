"""中文文本编码器 g(·)：用于事件语义相似度与 episode 检索。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

HF_MODEL = "BAAI/bge-base-zh-v1.5"
MODELSCOPE_MODEL_ID = "Xorbits/bge-base-zh-v1.5"

_DEFAULT_LOCAL_CANDIDATES = [
    Path.home() / ".cache/modelscope/Xorbits/bge-base-zh-v1.5",
    Path.home() / ".cache/modelscope/Xorbits/bge-base-zh-v1___5",
    Path.home() / ".cache/huggingface/hub/models--BAAI--bge-base-zh-v1.5/snapshots",
]


def _local_modelscope_path() -> Optional[str]:
    """返回已下载的本地 BGE 目录，不存在则 None。"""
    for candidate in _DEFAULT_LOCAL_CANDIDATES:
        if candidate.name == "snapshots" and candidate.is_dir():
            for snap in sorted(candidate.iterdir()):
                if (snap / "config.json").is_file():
                    return str(snap)
        elif (candidate / "config.json").is_file():
            return str(candidate)
    return None


def resolve_encoder_model_path(model_name: Optional[str] = None) -> str:
    """
    解析编码器路径：环境变量 > 显式本地路径 > ModelScope/HF 本地缓存 > HuggingFace 名。

    当 model_name 为 HF repo id（如 BAAI/bge-base-zh-v1.5）时，优先使用本地缓存，
    避免离线环境触发网络下载。
    """
    if env := os.getenv("PRIOR_ENHANCE_TEXT_ENCODER_MODEL"):
        p = Path(env)
        if p.is_dir() and (p / "config.json").is_file():
            return str(p)
        return env
    if model_name:
        p = Path(model_name)
        if p.is_dir() and (p / "config.json").is_file():
            return str(p)
    local = _local_modelscope_path()
    if local:
        return local
    if model_name:
        return model_name
    return HF_MODEL


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """懒加载 SentenceTransformer（进程内单例）。"""
    from sentence_transformers import SentenceTransformer

    device = os.getenv("PRIOR_ENHANCE_TEXT_ENCODER_DEVICE", "cpu")
    local_only = os.getenv("PRIOR_ENHANCE_TEXT_ENCODER_LOCAL_ONLY", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    kwargs = {"device": device}
    if local_only and Path(model_name).is_dir():
        kwargs["local_files_only"] = True
    return SentenceTransformer(model_name, **kwargs)


class ChineseTextEncoder:
    """
    中文语义向量编码器，默认 BAAI/bge-base-zh-v1.5。

    BGE 系列在 C-MTEB 中文榜单表现优异，适合事件文本相似度计算。
    推荐通过 ModelScope 预下载至 ~/.cache/modelscope/Xorbits/bge-base-zh-v1.5。
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = resolve_encoder_model_path(model_name)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        *,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """将文本映射为语义向量；单条输入返回 shape (dim,)，多条返回 (n, dim)。"""
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)
        if not text_list:
            return np.zeros(0, dtype=np.float32)

        cleaned = [t.strip() if t else "" for t in text_list]
        vecs = self.model.encode(
            cleaned,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        arr = np.asarray(vecs, dtype=np.float32)
        return arr[0] if single else arr

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度，输入可为任意维向量（内部归一化）。"""
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        if a.size == 0 or b.size == 0:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))

    def max_similarity(self, query: str, candidates: Sequence[str]) -> float:
        """query 与 candidates 之间的最大余弦相似度 s^max。"""
        if not query or not query.strip():
            return 0.0
        valid = [c for c in candidates if c and c.strip()]
        if not valid:
            return 0.0
        q_vec = self.encode(query)
        c_vecs = self.encode(valid)
        if c_vecs.ndim == 1:
            return self.cosine_similarity(q_vec, c_vecs)
        return float(np.max(c_vecs @ q_vec))


_default_encoder: Optional[ChineseTextEncoder] = None


def get_text_encoder(model_name: Optional[str] = None) -> ChineseTextEncoder:
    """获取全局编码器实例。"""
    global _default_encoder
    resolved = resolve_encoder_model_path(model_name)
    if _default_encoder is None or _default_encoder.model_name != resolved:
        _default_encoder = ChineseTextEncoder(resolved)
    return _default_encoder


def get_text_encoder_from_config(cfg: Optional[object] = None) -> ChineseTextEncoder:
    """从 PriorEnhanceConfig.text_encoder 或环境变量初始化编码器。"""
    model_name: Optional[str] = None
    device: Optional[str] = None
    if cfg is not None:
        enc_cfg = getattr(cfg, "text_encoder", None)
        if enc_cfg is not None:
            model_name = getattr(enc_cfg, "model_name", None)
            device = getattr(enc_cfg, "device", None)
    if device:
        os.environ.setdefault("PRIOR_ENHANCE_TEXT_ENCODER_DEVICE", device)
    return get_text_encoder(model_name)
