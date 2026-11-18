"""prior_enhance 全局配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_PKG_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_CSV = str(_PKG_DIR / "data" / "代理购电_小时.csv")


def _load_dashscope_api_key() -> str:
    """优先级: 环境变量 > prior_enhance/secrets.py > 空字符串。"""
    if key := os.getenv("DASHSCOPE_API_KEY"):
        return key
    try:
        from prior_enhance import secrets

        return getattr(secrets, "DASHSCOPE_API_KEY", "") or ""
    except ImportError:
        return ""


@dataclass
class PriorConfig:
    """先验生成与置信度校准参数。"""

    confidence_lambda: float = 0.5  # λ：b = λ b^LLM + (1-λ) b^evidence
    episode_ref_k: int = 10  # |D_m| 归一化常数 K


@dataclass
class TextEncoderConfig:
    """中文文本编码器配置（事件新颖性 ν_m）。"""

    model_name: str = "BAAI/bge-base-zh-v1.5"
    device: str = "cpu"  # cpu | cuda


@dataclass
class LLMConfig:
    api_key: str = field(default_factory=_load_dashscope_api_key)
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "deepseek-r1-distill-qwen-1.5b"
    temperature: float = 0.2
    max_tokens: int = 8192
    stream: bool = True
    min_priors: int = 3
    max_priors: int = 8
    max_retries: int = 2


@dataclass
class WindowConfig:
    """
    预测窗口参数。

    默认 holdout 模式：全量 CSV 末尾留 H 点作测试 O_m，其余全部为历史 H_m（单一窗口）。
    H = forecast_days × points_per_day（小时数据默认 3 天 × 24 = 72）。
    """

    forecast_days: int = 3  # 预测未来天数
    points_per_day: int = 24  # 每天时点数（小时数据为 24）
    horizon: Optional[int] = None  # 若 None 则 H = forecast_days * points_per_day
    search_budget: int = 50
    loss_name: str = "mape"
    min_history_points: int = 48  # H_m 至少点数
    mode: str = "holdout"  # holdout | rolling（rolling 保留供实验）

    def effective_horizon(self) -> int:
        if self.horizon is not None:
            return self.horizon
        return self.forecast_days * self.points_per_day


@dataclass
class DataConfig:
    id_col: str = "item_id"
    time_col: str = "timestamp"
    target_col: str = "value"
    default_item_id: str = "default_item"
    fill_method: str = "auto"
    freq: Optional[str] = None  # 显式频率: 5T|15T|h|D|W-SUN|M；None 则自动推断


@dataclass
class AutoApexConfig:
    """AutoApex serve 训练/推理配置。"""

    base_url: str = field(
        default_factory=lambda: os.getenv("AUTOAPEX_BASE_URL", "http://127.0.0.1:8000/api")
    )
    eval_backend: str = field(
        default_factory=lambda: os.getenv("PRIOR_ENHANCE_EVAL_BACKEND", "autoapex")
    )  # autoapex | proxy
    train_timeout: int = field(
        default_factory=lambda: int(os.getenv("AUTOAPEX_TRAIN_TIMEOUT", "7200"))
    )
    infer_timeout: int = field(
        default_factory=lambda: int(os.getenv("AUTOAPEX_INFER_TIMEOUT", "600"))
    )
    fallback_proxy: bool = field(
        default_factory=lambda: os.getenv("PRIOR_ENHANCE_EVAL_FALLBACK_PROXY", "").lower()
        in ("1", "true", "yes")
    )


@dataclass
class SearchConfig:
    """窗口内 validation search 与 observation 记忆更新参数。"""

    max_trials: int = 12
    batch_size: int = 4
    explore_ratio: float = 0.15  # ρ_k：p_{m,k} = (1-ρ_k)q + ρ_k p_exp
    beta_k: float = 1.0  # softmax 逆温度
    use_softmax_proposal: bool = True  # q_{m,k} ∝ exp(β ψ)
    proposal_pool_size: int = 48  # 候选池规模（近似归一化）
    val_horizon: int = 48
    lambda_c: float = 0.1
    lambda_m: Optional[float] = None  # None → 排行榜/环境变量冷启动
    prior_similarity_min: float = 0.25  # s_min：相似 prior 检索阈值
    prior_semantic_weight: float = 0.15  # 文本编码器在 s(z,z') 中的权重
    prior_store_path: Optional[str] = None  # 默认 outputs/prior_store.jsonl
    min_samples_for_gain: int = 4
    chi_min: float = 0.25
    omega_val: float = 0.2
    eta_val: float = 0.15
    omega_obs_max: float = 0.35
    eta_obs: float = 0.1
    eta_lambda: float = 0.05
    retain_top_k: int = 3
    retain_explore_k: int = 2
    kappa: float = 0.0


@dataclass
class LoggingConfig:
    """透明日志配置。"""

    enabled: bool = True
    console: bool = True
    level: str = "INFO"  # DEBUG | INFO | WARNING
    log_dir: Optional[str] = None  # 默认 outputs/logs


@dataclass
class PriorEnhanceConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    data: DataConfig = field(default_factory=DataConfig)
    autoapex: AutoApexConfig = field(default_factory=AutoApexConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), "outputs"
        )
    )
    domain_knowledge: str = (
        "电力/负荷类时序：通常具有日周期与周周期；节假日与极端天气会导致峰值与分布漂移；"
        "非平稳场景下应谨慎使用全历史等权训练；"
        "低样本场景应避免高维 tsfresh 与大集成；"
        "强漂移场景应优先近期窗口与轻量模型。"
    )

    @classmethod
    def from_env(cls) -> "PriorEnhanceConfig":
        cfg = cls()
        if model := os.getenv("PRIOR_ENHANCE_LLM_MODEL"):
            cfg.llm.model = model
        if horizon := os.getenv("PRIOR_ENHANCE_HORIZON"):
            cfg.window.horizon = int(horizon)
        if days := os.getenv("PRIOR_ENHANCE_FORECAST_DAYS"):
            cfg.window.forecast_days = int(days)
        if freq := os.getenv("PRIOR_ENHANCE_FREQ"):
            cfg.data.freq = freq
        if min_p := os.getenv("PRIOR_ENHANCE_MIN_PRIORS"):
            cfg.llm.min_priors = int(min_p)
        if max_p := os.getenv("PRIOR_ENHANCE_MAX_PRIORS"):
            cfg.llm.max_priors = int(max_p)
        if max_r := os.getenv("PRIOR_ENHANCE_MAX_RETRIES"):
            cfg.llm.max_retries = int(max_r)
        if log_level := os.getenv("PRIOR_ENHANCE_LOG_LEVEL"):
            cfg.logging.level = log_level.upper()
        if enc_model := os.getenv("PRIOR_ENHANCE_TEXT_ENCODER_MODEL"):
            cfg.text_encoder.model_name = enc_model
        if enc_dev := os.getenv("PRIOR_ENHANCE_TEXT_ENCODER_DEVICE"):
            cfg.text_encoder.device = enc_dev
        if os.getenv("PRIOR_ENHANCE_LOG_CONSOLE", "").lower() in ("0", "false", "no"):
            cfg.logging.console = False
        if trials := os.getenv("PRIOR_ENHANCE_MAX_TRIALS"):
            cfg.search.max_trials = int(trials)
        if val_h := os.getenv("PRIOR_ENHANCE_VAL_HORIZON"):
            cfg.search.val_horizon = int(val_h)
        if backend := os.getenv("PRIOR_ENHANCE_EVAL_BACKEND"):
            cfg.autoapex.eval_backend = backend
        if base := os.getenv("AUTOAPEX_BASE_URL"):
            cfg.autoapex.base_url = base
        if s_min := os.getenv("PRIOR_ENHANCE_PRIOR_SIM_MIN"):
            cfg.search.prior_similarity_min = float(s_min)
        if lam := os.getenv("PRIOR_ENHANCE_LAMBDA_M"):
            cfg.search.lambda_m = float(lam)
        return cfg

    def default_prior_store_path(self) -> str:
        if self.search.prior_store_path:
            return self.search.prior_store_path
        return os.path.join(self.output_dir, "prior_store.jsonl")
