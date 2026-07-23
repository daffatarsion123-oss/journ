"""Projection / classifier heads and the combined contrastive model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import EncoderConfig
from ..data.feature_layout import FeatureLayout
from .encoders import build_encoder, _BaseEncoder


class ProjectionHead(nn.Module):
    """MLP projection head (SimCLR/SupCon). Output is L2-normalized."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class LinearClassifierHead(nn.Module):
    """Linear probe over (optionally frozen) embeddings."""

    def __init__(self, in_dim: int, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ContrastiveModel(nn.Module):
    """Encoder + projection head; exposes both embeddings and projections.

    ``embed`` returns the encoder representation used for retrieval / linear
    probing; ``project`` returns the normalized vector used by the contrastive
    loss.
    """

    def __init__(self, cfg: EncoderConfig, input_dim: int, layout: FeatureLayout):
        super().__init__()
        self.encoder: _BaseEncoder = build_encoder(cfg, input_dim, layout)
        self.projector = ProjectionHead(
            cfg.embedding_dim, cfg.projection_hidden, cfg.projection_dim
        )
        self.embedding_dim = cfg.embedding_dim

    def embed(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        h = self.encoder(x)
        return F.normalize(h, dim=-1) if normalize else h

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)
