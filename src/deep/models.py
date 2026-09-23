"""Deep models: AttnSleep (Eldele et al. 2021) and TinySleepNet (Supratak & Guo 2020) ports, and seqctx_transformer.

AttnSleep (github.com/emadeldeen24/AttnSleep, model/model.py; repo defaults for Sleep-EDF). Quirks kept on
purpose so the model matches the original: the TCE treats the 30 AFR channels as the sequence and 80 time steps
as d_model; EncoderLayer's first sublayer ignores its normalised input; MultiHeadedAttention.convs[0] is unused;
SEBasicBlock passes `stride` as the kernel size (= 1). Input (B, C, 3000) -> logits (B, 5).

TinySleepNet (github.com/akaraspt/tinysleepnet): CNN per epoch + LSTM over L = 20 epochs. Deviations: TF 'SAME'
padding emulated (flattened size 128 x 16 = 2048); no LSTM peepholes; no LSTM state across windows (sliding
windows instead, src/deep/train.py). Input (B, L, C, 3000) -> logits (B, L, 5).

seqctx_transformer (our design): TinySleepNet's CNN per epoch -> Linear(128) -> 2-layer pre-norm Transformer with
sinusoidal positions over L = 20 epochs -> a classifier at every position. Input (B, L, C, 3000) -> (B, L, 5).
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


# ============================================================================ AttnSleep
class SELayer(nn.Module):
    """Squeeze-and-excitation over channels."""

    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        y = self.fc(self.avg_pool(x).view(x.shape[0], -1)).view(x.shape[0], -1, 1)
        return x * y.expand_as(x)  # expand_as, not broadcasting: same gradient summation order as the repo


class SEBasicBlock(nn.Module):
    """Residual SE block = adaptive feature recalibration (AFR)."""

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)  # sic: kernel_size = stride (= 1), as in the repo
        self.bn1, self.relu = nn.BatchNorm1d(planes), nn.ReLU(inplace=True)
        self.conv2, self.bn2, self.se = nn.Conv1d(planes, planes, 1), nn.BatchNorm1d(planes), SELayer(planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.se(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))
        return self.relu(out + (x if self.downsample is None else self.downsample(x)))


class MRCNN(nn.Module):
    """Multi-resolution CNN (small and large kernels) + AFR. (B, C, 3000) -> (B, 30, 80)."""

    def __init__(self, afr: int, n_channels: int = 1, drate: float = 0.5) -> None:
        super().__init__()
        g = nn.GELU()

        def branch(k, s, p, pool1, k2, pool2):
            return nn.Sequential(nn.Conv1d(n_channels, 64, k, s, bias=False, padding=p), nn.BatchNorm1d(64), g, nn.MaxPool1d(*pool1),
                                 nn.Dropout(drate), nn.Conv1d(64, 128, k2, 1, bias=False, padding=k2 // 2), nn.BatchNorm1d(128), g,
                                 nn.Conv1d(128, 128, k2, 1, bias=False, padding=k2 // 2), nn.BatchNorm1d(128), g, nn.MaxPool1d(*pool2))

        self.features1 = branch(50, 6, 24, (8, 2, 4), 8, (4, 4, 2))
        self.features2 = branch(400, 50, 200, (4, 2, 2), 7, (2, 2, 1))
        self.dropout = nn.Dropout(drate)
        self.AFR = nn.Sequential(SEBasicBlock(128, afr, 1, nn.Sequential(nn.Conv1d(128, afr, 1, bias=False), nn.BatchNorm1d(afr))))

    def forward(self, x):
        return self.AFR(self.dropout(torch.cat((self.features1(x), self.features2(x)), dim=2)))


class CausalConv1d(nn.Conv1d):
    """Conv1d padded on both sides then right-trimmed, so output t sees only inputs <= t."""

    def __init__(self, cin: int, cout: int, kernel_size: int) -> None:
        super().__init__(cin, cout, kernel_size, padding=kernel_size - 1)
        self.trim = kernel_size - 1

    def forward(self, x):
        return super().forward(x)[:, :, : -self.trim]


class MultiHeadedAttention(nn.Module):
    """Multi-head attention with causal-conv key / value projections."""

    def __init__(self, h: int, d_model: int, afr: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_k, self.h = d_model // h, h
        conv = CausalConv1d(afr, afr, 7)  # one conv copied 3 times, as the repo's clones(); convs[0] is unused
        self.convs = nn.ModuleList([copy.deepcopy(conv) for _ in range(3)])
        self.linear, self.dropout = nn.Linear(d_model, d_model), nn.Dropout(dropout)

    def forward(self, query, key, value):
        nb = query.size(0)  # operation order as in the repo (it fixes the gradient summation order)
        q = query.view(nb, -1, self.h, self.d_k).transpose(1, 2)
        k = self.convs[1](key).view(nb, -1, self.h, self.d_k).transpose(1, 2)
        v = self.convs[2](value).view(nb, -1, self.h, self.d_k).transpose(1, 2)
        p = self.dropout(F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1))
        return self.linear(torch.matmul(p, v).transpose(1, 2).contiguous().view(nb, -1, self.h * self.d_k))


class LayerNorm(nn.Module):
    """The repo's LayerNorm: unbiased std, eps added to the std."""

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.a_2, self.b_2, self.eps = nn.Parameter(torch.ones(features)), nn.Parameter(torch.zeros(features)), eps

    def forward(self, x):
        mean, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm, self.dropout = LayerNorm(size), nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.w_1, self.w_2, self.dropout = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model), nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """One TCE layer: causal-conv query, causal multi-head attention, feed-forward."""

    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, afr: int, dropout: float) -> None:
        super().__init__()
        self.self_attn, self.feed_forward, self.size = self_attn, feed_forward, size
        self.sublayer_output = nn.ModuleList([SublayerOutput(size, dropout) for _ in range(2)])
        self.conv = CausalConv1d(afr, afr, 7)

    def forward(self, x_in):
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda _x: self.self_attn(query, x_in, x_in))  # sic: ignores its argument
        return self.sublayer_output[1](x, self.feed_forward)


class TCE(nn.Module):
    def __init__(self, layer: EncoderLayer, n: int) -> None:
        super().__init__()
        self.layers, self.norm = nn.ModuleList([copy.deepcopy(layer) for _ in range(n)]), LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class AttnSleep(nn.Module):
    is_sequence = False

    def __init__(self, n_channels: int = 1, n_classes: int = 5, n_tce: int = 2, d_model: int = 80, d_ff: int = 120,
                 n_heads: int = 5, dropout: float = 0.1, afr_reduced_cnn_size: int = 30, mrcnn_dropout: float = 0.5) -> None:
        super().__init__()
        afr = afr_reduced_cnn_size
        self.mrcnn = MRCNN(afr, n_channels, mrcnn_dropout)
        self.tce = TCE(EncoderLayer(d_model, MultiHeadedAttention(n_heads, d_model, afr), PositionwiseFeedForward(d_model, d_ff, dropout),
                                    afr, dropout), n_tce)
        self.fc = nn.Linear(d_model * afr, n_classes)

    def forward(self, x):
        return self.fc(self.tce(self.mrcnn(x)).contiguous().flatten(1))


# ============================================================================ TinySleepNet
class SamePad1d(nn.Module):
    """Zero-pad (B, C, T) like TensorFlow padding='SAME' (output length ceil(T / stride))."""

    def __init__(self, kernel_size: int, stride: int, value: float = 0.0) -> None:
        super().__init__()
        self.kernel_size, self.stride, self.value = kernel_size, stride, value

    def forward(self, x):
        total = max((math.ceil(x.shape[-1] / self.stride) - 1) * self.stride + self.kernel_size - x.shape[-1], 0)
        return F.pad(x, (total // 2, total - total // 2), value=self.value)


class TinySleepNetCNN(nn.Module):
    """Per-epoch CNN encoder. (B, C, T) -> (B, 2048)."""

    def __init__(self, n_channels: int = 1, sfreq: float = 100.0, n_filters: int = 128, dropout: float = 0.5) -> None:
        super().__init__()
        conv = lambda cin, k, s: [SamePad1d(k, s), nn.Conv1d(cin, n_filters, k, s, bias=False), nn.BatchNorm1d(n_filters), nn.ReLU()]  # noqa: E731
        pool = lambda k: [SamePad1d(k, k, float("-inf")), nn.MaxPool1d(k, k)]  # noqa: E731  (-inf: padding never wins the max)
        self.net = nn.Sequential(*conv(n_channels, int(sfreq / 2), int(sfreq / 16)), *pool(8), nn.Dropout(dropout),
                                 *conv(n_filters, 8, 1), *conv(n_filters, 8, 1), *conv(n_filters, 8, 1), *pool(4),
                                 nn.Flatten(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class TinySleepNet(nn.Module):
    is_sequence = True

    def __init__(self, n_channels: int = 1, n_classes: int = 5, sfreq: float = 100.0, n_filters: int = 128,
                 n_rnn_units: int = 128, n_rnn_layers: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        self.cnn = TinySleepNetCNN(n_channels, sfreq, n_filters, dropout)
        with torch.no_grad():  # a train-mode pass (as in the original): it also fixes the random-number state for the LSTM init
            self.feat_dim = self.cnn(torch.zeros(1, n_channels, 3000)).shape[-1]
        self.rnn = nn.LSTM(self.feat_dim, n_rnn_units, num_layers=n_rnn_layers, batch_first=True)
        self.rnn_dropout, self.fc = nn.Dropout(dropout), nn.Linear(n_rnn_units, n_classes)

    def forward(self, x):
        b, l, c, t = x.shape
        feats = self.cnn(x.reshape(b * l, c, t)).reshape(b, l, -1)
        with torch.autocast(device_type=x.device.type, enabled=False):  # keep the LSTM in fp32 under bf16 autocast
            out, _ = self.rnn(feats.float())
        return self.fc(self.rnn_dropout(out))

    def conv_l2_params(self) -> list[nn.Parameter]:
        """Conv kernels: the only tensors the original applies L2 to."""
        return [m.weight for m in self.cnn.net if isinstance(m, nn.Conv1d)]


# ============================================================================ seqctx_transformer
class SinusoidalPositions(nn.Module):
    """Fixed sinusoidal position encoding (Vaswani et al. 2017) added to (B, L, D)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pos, div = torch.arange(max_len).unsqueeze(1), torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.shape[1]].to(x.dtype)


class SeqContextNet(nn.Module):
    is_sequence = True

    def __init__(self, n_channels: int = 1, n_classes: int = 5, encoder: str = "tinysleepnet_cnn", encoder_dropout: float = 0.5,
                 d_model: int = 128, seq_model: str = "transformer", n_layers: int = 2, n_heads: int = 4, ff_mult: int = 2,
                 dropout: float = 0.2) -> None:
        super().__init__()
        assert encoder == "tinysleepnet_cnn" and seq_model == "transformer"
        self.encoder = TinySleepNetCNN(n_channels, 100.0, 128, encoder_dropout)
        with torch.no_grad():
            feat_dim = self.encoder.eval()(torch.zeros(2, n_channels, 3000)).shape[-1]
        self.encoder.train()
        self.proj = nn.Sequential(nn.Linear(feat_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos = SinusoidalPositions(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_mult * d_model, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.seq = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x):
        b, l, c, t = x.shape
        return self.head(self.seq(self.pos(self.proj(self.encoder(x.reshape(b * l, c, t))).reshape(b, l, -1))))


MODELS: dict[str, type[nn.Module]] = {"attnsleep": AttnSleep, "tinysleepnet": TinySleepNet, "seq_context": SeqContextNet}


def build_model(name: str, **kwargs: Any) -> nn.Module:
    return MODELS[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
