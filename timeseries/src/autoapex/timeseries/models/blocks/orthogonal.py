"""
正交法 (Orthogonal) 算法模块。
供 timeseries 下所有深度学习模型统一调用的外部算法块。
包含：Splitting, CausalConvBlock, SCIBlock, RevIN, series_decomp, EE 等。
"""
import torch
from torch import nn


class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        return self.even(x), self.odd(x)


class CausalConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dropout=0.0):
        super(CausalConvBlock, self).__init__()
        module_list = [
            nn.ReplicationPad1d((kernel_size - 1, kernel_size - 1)),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.Tanh(),
        ]
        self.causal_conv = nn.Sequential(*module_list)

    def forward(self, x):
        return self.causal_conv(x)


class SCIBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dropout=0.0):
        super(SCIBlock, self).__init__()
        self.splitting = Splitting()
        self.modules_even = CausalConvBlock(d_model, kernel_size, dropout)
        self.modules_odd = CausalConvBlock(d_model, kernel_size, dropout)
        self.interactor_even = CausalConvBlock(d_model, kernel_size, dropout)
        self.interactor_odd = CausalConvBlock(d_model, kernel_size, dropout)

    def forward(self, x):
        x_even, x_odd = self.splitting(x)
        x_even = x_even.permute(0, 2, 1)
        x_odd = x_odd.permute(0, 2, 1)

        x_even_temp = x_even.mul(torch.exp(self.modules_even(x_odd)))
        x_odd_temp = x_odd.mul(torch.exp(self.modules_odd(x_even)))

        x_even_update = x_even_temp + self.interactor_even(x_odd_temp)
        x_odd_update = x_odd_temp - self.interactor_odd(x_even_temp)

        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1)


class ConvCat(nn.Module):
    def __init__(self, c_in):
        super(ConvCat, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            stride=2,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x


class RevIN(nn.Module):
    def __init__(
        self, num_features: int, eps=1e-5, affine=True, subtract_last=False
    ):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = (
            torch.sqrt(
                torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
                + self.eps
            )
        ).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EE(nn.Module):
    """
    正交法中的 Energy Enhancement 块。
    输入/输出形状: (batch, seq_len, channels)。
    可供各时序深度学习模型在外部统一调用。
    """

    def __init__(self, seq_len, pred_len, enc_in, hidden_size):
        super(EE, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.hidden_size = hidden_size
        self.revin_layer = RevIN(enc_in, affine=True, subtract_last=False)

        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        self.mask_matrix = nn.Parameter(
            torch.ones(int(self.seq_len / 2) + 1, self.channels)
        )
        self.freq_linear = nn.Linear(
            int(self.seq_len / 2) + 1, int(self.pred_len / 2) + 1
        ).to(torch.cfloat)

        self.extract_common_pattern = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.LeakyReLU(),
            nn.Linear(self.channels, 1),
        )
        self.model_common_pattern = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.seq_len),
        )
        self.model_spacific_pattern = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.seq_len),
        )

    def forward(self, x):
        _, seq_len, C = x.shape
        seasonal, trend = self.decompsition(x)
        z = seasonal
        z = self.revin_layer(z, "norm")
        x = z

        x_fft = torch.fft.rfft(x, dim=1)
        x_inverse_fft = torch.flip(x_fft, dims=[1])
        x_inverse_fft = x_inverse_fft * self.mask_matrix
        x_amplifier_fft = x_fft + x_inverse_fft
        # 关键：显式指定 n，避免原序列长度为奇数时 irfft 丢失 1 个时间步
        x_amplifier = torch.fft.irfft(x_amplifier_fft, n=seq_len, dim=1)
        x = x_amplifier

        common_pattern = self.extract_common_pattern(x)
        common_pattern = self.model_common_pattern(
            common_pattern.permute(0, 2, 1)
        ).permute(0, 2, 1)
        specififc_pattern = x - common_pattern.repeat(1, 1, C)
        specififc_pattern = self.model_spacific_pattern(
            specififc_pattern.permute(0, 2, 1)
        ).permute(0, 2, 1)
        x = specififc_pattern + common_pattern.repeat(1, 1, C)
        x_amplifier = x

        z = x
        z = self.revin_layer(z, "denorm")
        out = z + trend
        return out
