"""prior_enhance 包入口。"""

from prior_enhance._bootstrap import patch_random_randbits

patch_random_randbits()

from prior_enhance.config import PriorEnhanceConfig
from prior_enhance.schemas import ForecastTask, PriorSet, StructuredPrior

__all__ = [
    "PriorEnhanceConfig",
    "ForecastTask",
    "PriorSet",
    "StructuredPrior",
]


def __getattr__(name: str):
    if name == "PriorEnhancePipeline":
        from prior_enhance.pipeline import PriorEnhancePipeline
        return PriorEnhancePipeline
    raise AttributeError(name)
