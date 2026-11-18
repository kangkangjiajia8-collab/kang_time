"""
TemporalFusionTransformer with optional orthogonal enhancement on past_target.

This wrapper keeps the GluonTS TFT training / prediction pipeline unchanged and
only augments the underlying TFT network with an optional EE + SCI/CausalConv
block applied to `past_target`.
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
from torch import nn
from gluonts.core.component import validated
from gluonts.itertools import select
from gluonts.torch.model.lightning_util import has_validation_loop
from gluonts.torch.model.tft.estimator import TemporalFusionTransformerEstimator as GluonTFTEstimator
from gluonts.torch.model.tft.module import (
    TemporalFusionTransformerModel as GluonTFTModel,
)

from autoapex.timeseries.models.blocks import CausalConvBlock, ConvCat, EE, SCIBlock


class TemporalFusionTransformerOrthogonalNetwork(GluonTFTModel):
    """TFT network with optional orthogonal enhancement on `past_target`."""

    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        use_orthogonal: bool = False,
        orthogonal_hidden_size: int = 128,
        orthogonal_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
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

        self.orth_causal: CausalConvBlock | None = None
        self.orth_sci: SCIBlock | None = None
        self.orth_convcat: ConvCat | None = None

        if self.use_orthogonal:
            # In TFT, `past_target` length should always equal `context_length`.
            self.orthogonal_block = EE(
                seq_len=self.context_length,
                pred_len=self.context_length,
                enc_in=1,
                hidden_size=self.orthogonal_hidden_size,
            )
            self.orth_linear_in = nn.Linear(1, self.orthogonal_hidden_size, bias=True)
            self.orth_linear_mid = nn.Linear(
                self.orthogonal_hidden_size, self.orthogonal_hidden_size, bias=True
            )
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

        # past_target shape: (batch_size, context_length)
        x = past_target.unsqueeze(-1)  # (B, T, 1)
        original_len = x.shape[1]
        expected_len = self.orthogonal_block.seq_len

        # Align to EE initialization length to avoid linear dimension mismatch.
        if original_len < expected_len:
            x = torch.nn.functional.pad(x, (0, 0, 0, expected_len - original_len), mode="replicate")
        elif original_len > expected_len:
            x = x[:, :expected_len, :]

        x = self.orthogonal_block(x)  # (B, T, 1)

        assert self.orth_linear_in is not None
        assert self.orth_linear_mid is not None
        assert self.orth_linear_out is not None
        assert self.orth_causal is not None
        assert self.orth_sci is not None
        assert self.orth_convcat is not None

        x_hidden = self.orth_linear_in(x)  # (B, T, H)
        residual = self.orth_linear_mid(x_hidden)  # (B, T, H)

        # Time-domain enhancement.
        x_causal = self.orth_causal(x_hidden.permute(0, 2, 1)).permute(0, 2, 1)
        x_causal = self.orth_activation(x_causal)

        # SCIBlock expects even temporal length; crop if needed.
        sci_input_len = x_causal.shape[1]
        if sci_input_len % 2 == 1:
            x_causal = x_causal[:, :-1, :]

        x_even, x_odd = self.orth_sci(x_causal)
        min_len = min(x_even.shape[1], x_odd.shape[1])
        x_merge = torch.cat(
            [x_even[:, :min_len, :], x_odd[:, :min_len, :]],
            dim=1,
        )  # (B, 2*min_len, H)

        if x_merge.shape[1] < sci_input_len:
            x_merge = torch.nn.functional.pad(
                x_merge, (0, 0, 0, sci_input_len - x_merge.shape[1]), mode="replicate"
            )
        x_merge = x_merge[:, : x_hidden.shape[1], :]

        x_down = self.orth_convcat(x_merge.permute(0, 2, 1)).permute(0, 2, 1)
        if x_down.shape[1] < x_hidden.shape[1]:
            pad_len = x_hidden.shape[1] - x_down.shape[1]
            x_down = torch.nn.functional.pad(x_down, (0, 0, 0, pad_len), mode="replicate")
        x_down = x_down[:, : x_hidden.shape[1], :]

        x_hidden = x_down + residual
        x_out = self.orth_linear_out(x_hidden)  # (B, T, 1)

        # Restore original past_target length.
        if x_out.shape[1] < original_len:
            x_out = torch.nn.functional.pad(x_out, (0, 0, 0, original_len - x_out.shape[1]), mode="replicate")
        x_out = x_out[:, :original_len, :]
        return x_out.squeeze(-1)

    def _preprocess(  # type: ignore[override]
        self,
        past_target: torch.Tensor,  # [N, T]
        past_observed_values: torch.Tensor,  # [N, T]
        feat_static_real: torch.Tensor | None,  # [N, D_sr]
        feat_static_cat: torch.Tensor | None,  # [N, D_sc]
        feat_dynamic_real: torch.Tensor | None,  # [N, T + H, D_dr]
        feat_dynamic_cat: torch.Tensor | None = None,  # [N, T + H, D_dc]
        past_feat_dynamic_real: torch.Tensor | None = None,  # [N, T, D_pr]
        past_feat_dynamic_cat: torch.Tensor | None = None,  # [N, T, D_pc]
    ):
        if self.use_orthogonal:
            past_target = self._apply_orthogonal_to_past_target(past_target)

        return super()._preprocess(
            past_target=past_target,
            past_observed_values=past_observed_values,
            feat_static_real=feat_static_real,
            feat_static_cat=feat_static_cat,
            feat_dynamic_real=feat_dynamic_real,
            feat_dynamic_cat=feat_dynamic_cat,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_feat_dynamic_cat=past_feat_dynamic_cat,
        )


class TemporalFusionTransformerOrthogonalLightningModule(pl.LightningModule):
    """Lightning module that trains `TemporalFusionTransformerOrthogonalNetwork`."""

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
        self.model = TemporalFusionTransformerOrthogonalNetwork(**model_kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.inputs = self.model.describe_inputs()
        self.example_input_array = self.inputs.zeros()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        train_loss = self.model.loss(
            **select(self.inputs, batch, ignore_missing=True),
            future_observed_values=batch["future_observed_values"],
            future_target=batch["future_target"],
        ).mean()
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        val_loss = self.model.loss(
            **select(self.inputs, batch, ignore_missing=True),
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
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=0.5,
                    patience=self.patience,
                ),
                "monitor": monitor,
            },
        }


class TemporalFusionTransformerOrthogonalEstimator(GluonTFTEstimator):
    """Drop-in TFT estimator with optional orthogonal backbone block."""

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: int | None = None,
        quantiles: list[float] | None = None,
        num_heads: int = 4,
        hidden_dim: int = 32,
        variable_dim: int = 32,
        dropout_rate: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
        batch_size: int = 32,
        use_orthogonal: bool = False,
        orthogonal_hidden_size: int = 128,
        orthogonal_dropout: float = 0.1,
        past_dynamic_dims: list[int] | None = None,
        dynamic_dims: list[int] | None = None,
        static_dims: list[int] | None = None,
        past_dynamic_cardinalities: list[int] | None = None,
        dynamic_cardinalities: list[int] | None = None,
        static_cardinalities: list[int] | None = None,
        **kwargs,
    ) -> None:
        # GluonTS hyperparams must be explicit: TemporalFusionTransformerModel._get_estimator
        # filters init_args by __init__ signature; **kwargs alone is not enough.
        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            quantiles=quantiles,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            variable_dim=variable_dim,
            dropout_rate=dropout_rate,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            batch_size=batch_size,
            past_dynamic_dims=past_dynamic_dims,
            dynamic_dims=dynamic_dims,
            static_dims=static_dims,
            past_dynamic_cardinalities=past_dynamic_cardinalities,
            dynamic_cardinalities=dynamic_cardinalities,
            static_cardinalities=static_cardinalities,
            **kwargs,
        )
        self.use_orthogonal = bool(use_orthogonal)
        self.orthogonal_hidden_size = int(orthogonal_hidden_size)
        self.orthogonal_dropout = float(orthogonal_dropout)

    def create_lightning_module(self) -> TemporalFusionTransformerOrthogonalLightningModule:
        return TemporalFusionTransformerOrthogonalLightningModule(
            lr=self.lr,
            patience=self.patience,
            weight_decay=self.weight_decay,
            model_kwargs={
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "d_var": self.variable_dim,
                "d_hidden": self.hidden_dim,
                "num_heads": self.num_heads,
                "distr_output": self.distr_output,
                "d_past_feat_dynamic_real": self.past_dynamic_dims,
                "c_past_feat_dynamic_cat": self.past_dynamic_cardinalities,
                "d_feat_dynamic_real": [1] * max(len(self.time_features), 1) + self.dynamic_dims,
                "c_feat_dynamic_cat": self.dynamic_cardinalities,
                "d_feat_static_real": self.static_dims or [1],
                "c_feat_static_cat": self.static_cardinalities or [1],
                "dropout_rate": self.dropout_rate,
                "use_orthogonal": self.use_orthogonal,
                "orthogonal_hidden_size": self.orthogonal_hidden_size,
                "orthogonal_dropout": self.orthogonal_dropout,
            },
        )

