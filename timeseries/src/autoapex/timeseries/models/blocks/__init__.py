"""
Timeseries 模型共享算法块 (blocks)。

各深度学习模型可统一从此包导入正交法等模块，实现外部添加、一处维护。
例如：
  from autoapex.timeseries.models.blocks import EE, RevIN, series_decomp
  # 在模型 forward 中：x = self.orthogonal_block(x)  # 输入/输出 (B, seq_len, C)

已接入：TimeXer/DETMModel（通过 hyperparameters.use_orthogonal 控制）。
GluonTS/Chronos/Toto 等自定义网络也可直接 import 使用。
"""
from .orthogonal import (
    CausalConvBlock,
    ConvCat,
    EE,
    RevIN,
    SCIBlock,
    Splitting,
    moving_avg,
    series_decomp,
)

__all__ = [
    "Splitting",
    "CausalConvBlock",
    "SCIBlock",
    "ConvCat",
    "RevIN",
    "moving_avg",
    "series_decomp",
    "EE",
]
