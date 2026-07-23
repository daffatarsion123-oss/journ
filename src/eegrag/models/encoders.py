"""Representation encoders.

All encoders map an input window (or window-sequence) to an L2-normalizable
embedding of dimension ``cfg.embedding_dim``. They accept either:
  * a flat feature vector batch  ``[B, F]``  (default, sequence_len == 1), or
  * a window-sequence batch      ``[B, T, F]`` (sequence_len > 1).

Topology handling: the CNN encoders reshape the flat vector into a
``[n_feat_per_channel, n_channels]`` grid (via :class:`FeatureLayout`) and
convolve along the *electrode* axis, so the spatial montage is respected and
channel-dropout augmentation maps onto whole conv columns.

For sequence inputs, MLP/CNN encode each timestep and mean-pool embeddings;
TCN / Transformer consume the temporal axis directly.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import EncoderConfig
from ..data.feature_layout import FeatureLayout


def _mlp(dims, dropout: float) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class _BaseEncoder(nn.Module):
    embedding_dim: int

    def _flatten_seq(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, F] -> [B, T*F]; [B, F] unchanged
        if x.dim() == 3:
            return x.reshape(x.shape[0], -1)
        return x


class MLPEncoder(_BaseEncoder):
    def __init__(self, cfg: EncoderConfig, input_dim: int, layout: FeatureLayout):
        super().__init__()
        self.embedding_dim = cfg.embedding_dim
        flat_dim = input_dim * cfg.sequence_len
        dims = [flat_dim, *cfg.hidden_dims, cfg.embedding_dim]
        self.net = _mlp(dims, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self._flatten_seq(x))


class CNN1DEncoder(_BaseEncoder):
    """1D-CNN convolving along the electrode axis.

    Reshapes ``[B, F]`` to ``[B, n_feat_per_channel, n_channels]`` (conv input
    channels = feature types, length = electrodes). Falls back to treating the
    flat vector as a single channel when the layout is irregular.
    """

    def __init__(self, cfg: EncoderConfig, input_dim: int, layout: FeatureLayout):
        super().__init__()
        self.embedding_dim = cfg.embedding_dim
        self.layout = layout
        self.regular = layout.is_regular and input_dim == layout.n_features
        in_ch = layout.n_feat_per_channel if self.regular else 1

        convs = []
        prev = in_ch
        for ch in cfg.conv_channels:
            convs.append(
                nn.Conv1d(prev, ch, kernel_size=cfg.kernel_size, padding=cfg.kernel_size // 2)
            )
            convs.append(nn.BatchNorm1d(ch))
            convs.append(nn.ReLU(inplace=True))
            convs.append(nn.Dropout(cfg.dropout))
            prev = ch
        self.conv = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(prev, cfg.embedding_dim)

    def _to_grid(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F] -> [B, in_ch, L]
        b = x.shape[0]
        if self.regular:
            n_c = self.layout.n_channels
            n_f = self.layout.n_feat_per_channel
            # reorder columns into channel-major then view [B, n_c, n_f] -> [B, n_f, n_c]
            order = torch.as_tensor(
                _channel_major_order(self.layout), device=x.device, dtype=torch.long
            )
            x = x.index_select(1, order).reshape(b, n_c, n_f).transpose(1, 2)
            return x
        return x.unsqueeze(1)  # [B, 1, F]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # sequence: encode per-timestep, mean-pool embeddings
            b, t, f = x.shape
            emb = self.forward(x.reshape(b * t, f))
            return emb.reshape(b, t, -1).mean(dim=1)
        grid = self._to_grid(x)
        h = self.conv(grid)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


def _channel_major_order(layout: FeatureLayout):
    import numpy as np
    return np.concatenate(layout.channel_indices_array())


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _chomp(self, x):  # causal: remove right padding
        return x[..., : -self.pad] if self.pad > 0 else x

    def forward(self, x):
        out = self.drop(self.relu(self._chomp(self.conv1(x))))
        out = self.drop(self.relu(self._chomp(self.conv2(out))))
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCNEncoder(_BaseEncoder):
    """Temporal Convolutional Network over a window-sequence ``[B, T, F]``."""

    def __init__(self, cfg: EncoderConfig, input_dim: int, layout: FeatureLayout):
        super().__init__()
        self.embedding_dim = cfg.embedding_dim
        self.input_dim = input_dim
        blocks = []
        prev = input_dim
        for i, ch in enumerate(cfg.conv_channels):
            blocks.append(
                _TemporalBlock(prev, ch, cfg.kernel_size, dilation=2 ** i,
                               dropout=cfg.dropout)
            )
            prev = ch
        self.tcn = nn.Sequential(*blocks)
        self.proj = nn.Linear(prev, cfg.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:           # single window -> length-1 sequence
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)      # [B, F, T]
        h = self.tcn(x)            # [B, C, T]
        h = h.mean(dim=-1)         # global average over time
        return self.proj(h)


class CNN1DTransformerEncoder(_BaseEncoder):
    """1D-CNN stem (over electrodes, per timestep) + Transformer over time."""

    def __init__(self, cfg: EncoderConfig, input_dim: int, layout: FeatureLayout):
        super().__init__()
        self.embedding_dim = cfg.embedding_dim
        self.stem = CNN1DEncoder(cfg, input_dim, layout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.embedding_dim * 2,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_transformer_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)            # [B, 1, F]
        b, t, f = x.shape
        tok = self.stem(x.reshape(b * t, f)).reshape(b, t, -1)  # [B, T, E]
        cls = self.cls.expand(b, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        out = self.transformer(seq)
        return out[:, 0]                 # CLS embedding


_REGISTRY = {
    "mlp": MLPEncoder,
    "cnn1d": CNN1DEncoder,
    "tcn": TCNEncoder,
    "cnn1d_transformer": CNN1DTransformerEncoder,
}


def build_encoder(
    cfg: EncoderConfig, input_dim: int, layout: FeatureLayout
) -> _BaseEncoder:
    if cfg.arch not in _REGISTRY:
        raise ValueError(f"Unknown encoder arch {cfg.arch!r}")
    return _REGISTRY[cfg.arch](cfg, input_dim, layout)
