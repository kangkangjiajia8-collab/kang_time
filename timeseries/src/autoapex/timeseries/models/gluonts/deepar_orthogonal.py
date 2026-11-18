"""
DeepAR with optional orthogonal block integrated into training backbone.

This module keeps GluonTS training / prediction pipeline unchanged, and only
augments the underlying DeepAR network with an optional EE block from
`autoapex.timeseries.models.blocks`.
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gluonts.core.component import validated
from gluonts.itertools import select
from gluonts.torch.model.lightning_util import has_validation_loop
from gluonts.torch.model.deepar.estimator import DeepAREstimator
from gluonts.torch.model.deepar.lightning_module import DeepARLightningModule
from gluonts.torch.model.deepar.module import DeepARModel as GluonTSDeepARModel

from autoapex.timeseries.models.blocks import (
    CausalConvBlock,
    ConvCat,
    EE,
    SCIBlock,
)


class DeepAROrthogonalNetwork(GluonTSDeepARModel):
    """DeepAR network with optional orthogonal enhancement on past target."""

    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        use_orthogonal: bool = False,
        orthogonal_hidden_size: int = 128,
        orthogonal_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            freq=freq,
            context_length=context_length,
            prediction_length=prediction_length,
            **kwargs,
        )
        self.use_orthogonal = bool(use_orthogonal)
        self.orthogonal_hidden_size = int(orthogonal_hidden_size)
        self.orthogonal_dropout = float(orthogonal_dropout)

        self.orthogonal_block: EE | None = None
        self.orth_linear_in: nn.Linear | None = None
        self.orth_linear_mid: nn.Linear | None = None
        self.orth_linear_out: nn.Linear | None = None
        self.orth_activation = nn.ReLU()

        # 时域增强链路
        self.orth_causal: CausalConvBlock | None = None
        self.orth_sci: SCIBlock | None = None
        self.orth_convcat: ConvCat | None = None

        if self.use_orthogonal:
            self.orthogonal_block = EE(
                seq_len=self._past_length,
                pred_len=self._past_length,
                enc_in=1,
                hidden_size=self.orthogonal_hidden_size,
            )
            self.orth_linear_in = nn.Linear(1, self.orthogonal_hidden_size, bias=True)
            self.orth_linear_mid = nn.Linear(self.orthogonal_hidden_size, self.orthogonal_hidden_size, bias=True)
            self.orth_linear_out = nn.Linear(self.orthogonal_hidden_size, 1, bias=True)

            self.orth_causal = CausalConvBlock(
                d_model=self.orthogonal_hidden_size,
                kernel_size=5,
                dropout=self.orthogonal_dropout,
            )
            self.orth_sci = SCIBlock(
                d_model=self.orthogonal_hidden_size,
                kernel_size=5,
                dropout=self.orthogonal_dropout,
            )
            self.orth_convcat = ConvCat(c_in=self.orthogonal_hidden_size)

    def _apply_orthogonal_to_past_target(self, past_target: torch.Tensor) -> torch.Tensor:
        if self.orthogonal_block is None:
            return past_target

        # past_target shape: (batch_size, past_length)
        x = past_target.unsqueeze(-1)
        original_len = x.shape[1]
        expected_len = self.orthogonal_block.seq_len

        # 对齐到 EE 初始化的 seq_len，避免线性层维度不匹配
        if original_len < expected_len:
            x = F.pad(x, (0, 0, 0, expected_len - original_len), mode="replicate")
        elif original_len > expected_len:
            x = x[:, :expected_len, :]

        # 频域增强（EE，内部包含 RevIN/series_decomp/moving_avg）
        x = self.orthogonal_block(x)

        # 映射到隐藏维度后做时域增强（CausalConv + SCI + ConvCat）
        assert self.orth_linear_in is not None
        assert self.orth_linear_mid is not None
        assert self.orth_linear_out is not None
        assert self.orth_causal is not None
        assert self.orth_sci is not None
        assert self.orth_convcat is not None

        x_hidden = self.orth_linear_in(x)
        residual = self.orth_linear_mid(x_hidden)

        x_causal = self.orth_causal(x_hidden.permute(0, 2, 1)).permute(0, 2, 1)
        x_causal = self.orth_activation(x_causal)

        # SCIBlock 需要偶数长度，奇数长度时先裁掉最后一个时间步，之后再补回
        sci_input_len = x_causal.shape[1]
        if sci_input_len % 2 == 1:
            x_causal = x_causal[:, :-1, :]

        x_even, x_odd = self.orth_sci(x_causal)
        min_len = min(x_even.shape[1], x_odd.shape[1])
        x_merge = torch.cat([x_even[:, :min_len, :], x_odd[:, :min_len, :]], dim=1)

        if x_merge.shape[1] < sci_input_len:
            x_merge = F.pad(x_merge, (0, 0, 0, sci_input_len - x_merge.shape[1]), mode="replicate")
        x_merge = x_merge[:, : x_hidden.shape[1], :]

        x_down = self.orth_convcat(x_merge.permute(0, 2, 1)).permute(0, 2, 1)
        if x_down.shape[1] < x_hidden.shape[1]:
            pad_len = x_hidden.shape[1] - x_down.shape[1]
            x_down = F.pad(x_down, (0, 0, 0, pad_len), mode="replicate")
        x_down = x_down[:, : x_hidden.shape[1], :]

        x_hidden = x_down + residual
        x_out = self.orth_linear_out(x_hidden)

        # 还原回原始 past_target 长度
        if x_out.shape[1] < original_len:
            x_out = F.pad(x_out, (0, 0, 0, original_len - x_out.shape[1]), mode="replicate")
        x_out = x_out[:, :original_len, :]

        return x_out.squeeze(-1)

    def prepare_rnn_input(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor | None = None,
    ):
        if self.use_orthogonal:
            past_target = self._apply_orthogonal_to_past_target(past_target)

        return super().prepare_rnn_input(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )


class DeepAROrthogonalLightningModule(pl.LightningModule):
    """Lightning module that trains `DeepAROrthogonalNetwork`."""

    @validated()
    def __init__(
        self,
        model_kwargs: dict[str, Any],
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = DeepAROrthogonalNetwork(**model_kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.inputs = self.model.describe_inputs()
        self.example_input_array = self.inputs.zeros()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        train_loss = self.model.loss(
            **select(self.inputs, batch),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
        ).mean()

        self.log("train_loss", train_loss, on_epoch=True, on_step=False, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        val_loss = self.model.loss(
            **select(self.inputs, batch),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
        ).mean()

        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        monitor = "val_loss" if has_validation_loop(self.trainer) else "train_loss"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=0.5,
                    patience=self.patience,
                ),
                "monitor": monitor,
            },
        }


class DeepAROrthogonalEstimator(DeepAREstimator):
    """Drop-in DeepAR estimator with optional orthogonal backbone block."""

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: int | None = None,
        use_orthogonal: bool = False,
        orthogonal_hidden_size: int = 128,
        orthogonal_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            **kwargs,
        )
        self.use_orthogonal = bool(use_orthogonal)
        self.orthogonal_hidden_size = int(orthogonal_hidden_size)
        self.orthogonal_dropout = float(orthogonal_dropout)

    def create_lightning_module(self) -> DeepARLightningModule:
        return DeepAROrthogonalLightningModule(
            lr=self.lr,
            weight_decay=self.weight_decay,
            patience=self.patience,
            model_kwargs={
                "freq": self.freq,
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "num_feat_dynamic_real": (1 + self.num_feat_dynamic_real + len(self.time_features)),
                "num_feat_static_real": max(1, self.num_feat_static_real),
                "num_feat_static_cat": max(1, self.num_feat_static_cat),
                "cardinality": self.cardinality,
                "embedding_dimension": self.embedding_dimension,
                "num_layers": self.num_layers,
                "hidden_size": self.hidden_size,
                "distr_output": self.distr_output,
                "dropout_rate": self.dropout_rate,
                "lags_seq": self.lags_seq,
                "scaling": self.scaling,
                "default_scale": self.default_scale,
                "num_parallel_samples": self.num_parallel_samples,
                "nonnegative_pred_samples": self.nonnegative_pred_samples,
                "use_orthogonal": self.use_orthogonal,
                "orthogonal_hidden_size": self.orthogonal_hidden_size,
                "orthogonal_dropout": self.orthogonal_dropout,
            },
        )
