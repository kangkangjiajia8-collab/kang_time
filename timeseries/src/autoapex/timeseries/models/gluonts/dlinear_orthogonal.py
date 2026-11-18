"""
DLinear with optional orthogonal enhancement on past_target.

This wrapper keeps the GluonTS DLinear training / prediction pipeline unchanged,
and only augments the underlying DLinear network with an optional orthogonal
block applied to `past_target`.
"""

from __future__ import annotations

from typing import Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn

from gluonts.core.component import validated
from gluonts.itertools import select
from gluonts.torch.model.d_linear.estimator import DLinearEstimator
from gluonts.torch.model.d_linear.module import DLinearModel as GluonTSDLinearModel

from autoapex.timeseries.models.blocks import CausalConvBlock, ConvCat, EE, SCIBlock


class DLinearOrthogonalNetwork(GluonTSDLinearModel):
    """DLinear network with optional orthogonal enhancement on past target."""

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        hidden_dimension: int,
        distr_output,
        kernel_size: int = 25,
        scaling: Optional[str] = "mean",
        use_orthogonal: bool = False,
        orthogonal_hidden_size: int = 128,
        orthogonal_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            hidden_dimension=hidden_dimension,
            distr_output=distr_output,
            kernel_size=kernel_size,
            scaling=scaling,
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
            self.orthogonal_block = EE(
                seq_len=self.context_length,
                pred_len=self.context_length,
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

        x = past_target.unsqueeze(-1)
        original_len = x.shape[1]
        expected_len = self.orthogonal_block.seq_len

        if original_len < expected_len:
            x = F.pad(x, (0, 0, 0, expected_len - original_len), mode="replicate")
        elif original_len > expected_len:
            x = x[:, :expected_len, :]

        x = self.orthogonal_block(x)

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
            x_down = F.pad(x_down, (0, 0, 0, x_hidden.shape[1] - x_down.shape[1]), mode="replicate")
        x_down = x_down[:, : x_hidden.shape[1], :]

        x_hidden = x_down + residual
        x_out = self.orth_linear_out(x_hidden)

        if x_out.shape[1] < original_len:
            x_out = F.pad(x_out, (0, 0, 0, original_len - x_out.shape[1]), mode="replicate")
        x_out = x_out[:, :original_len, :]
        return x_out.squeeze(-1)

    def forward(self, past_target: torch.Tensor, past_observed_values: torch.Tensor):
        if self.use_orthogonal:
            past_target = self._apply_orthogonal_to_past_target(past_target)

        return super().forward(
            past_target=past_target,
            past_observed_values=past_observed_values,
        )


class DLinearOrthogonalLightningModule(pl.LightningModule):
    """Lightning module that trains `DLinearOrthogonalNetwork`."""

    @validated()
    def __init__(self, model_kwargs: dict, lr: float = 1e-3, weight_decay: float = 1e-8):
        super().__init__()
        self.save_hyperparameters()
        self.model = DLinearOrthogonalNetwork(**model_kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.inputs = self.model.describe_inputs()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        train_loss = self.model.loss(
            **select(self.inputs, batch),
            future_target=batch["future_target"],
            future_observed_values=batch["future_observed_values"],
        ).mean()
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        val_loss = self.model.loss(
            **select(self.inputs, batch),
            future_target=batch["future_target"],
            future_observed_values=batch["future_observed_values"],
        ).mean()
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class DLinearOrthogonalEstimator(DLinearEstimator):
    """Drop-in DLinear estimator with optional orthogonal backbone block."""

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int | None = None,
        use_orthogonal: bool = False,
        orthogonal_hidden_size: int = 128,
        orthogonal_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            prediction_length=prediction_length,
            context_length=context_length,
            **kwargs,
        )
        self.use_orthogonal = bool(use_orthogonal)
        self.orthogonal_hidden_size = int(orthogonal_hidden_size)
        self.orthogonal_dropout = float(orthogonal_dropout)

    def create_lightning_module(self) -> pl.LightningModule:
        return DLinearOrthogonalLightningModule(
            lr=self.lr,
            weight_decay=self.weight_decay,
            model_kwargs={
                "prediction_length": self.prediction_length,
                "context_length": self.context_length,
                "hidden_dimension": self.hidden_dimension,
                "distr_output": self.distr_output,
                "kernel_size": self.kernel_size,
                "scaling": self.scaling,
                "use_orthogonal": self.use_orthogonal,
                "orthogonal_hidden_size": self.orthogonal_hidden_size,
                "orthogonal_dropout": self.orthogonal_dropout,
            },
        )
