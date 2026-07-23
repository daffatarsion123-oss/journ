"""Supervised contrastive losses, with imbalance-aware variants.

All losses consume features of shape ``[B, V, D]`` (V views per anchor, D the
projection dim, expected L2-normalized) and integer ``labels`` of shape ``[B]``.

Implemented:
  * :class:`SimCLRLoss`              -- unsupervised NT-Xent (sanity baseline).
  * :class:`SupConLoss`              -- Supervised Contrastive (Khosla et al. 2020).
  * :class:`BalancedContrastiveLoss` -- class-balanced contrastive: averages the
    positive log-prob *per class* so the majority class cannot dominate the
    gradient (a Balanced-Contrastive-Learning-style baseline).
  * :class:`ImbalanceAwareSupConLoss`-- SupCon with per-anchor class weighting
    (inverse-frequency or explicit minority weight) and optional focal-style
    down-weighting of easy anchors. This is the proposed loss to compare against
    the Balanced Contrastive baseline (RQ: does imbalance-aware weighting beat
    plain class balancing?).

NOTE on the comparison the user asked for: ``ImbalanceAwareSupConLoss`` and
``BalancedContrastiveLoss`` are *distinct* objectives so the journal can report
"Imbalance-Aware SupCon vs Balanced Contrastive" head-to-head.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import LossConfig


def _flatten_views(features: torch.Tensor):
    """[B, V, D] -> contrast features [B*V, D] and view count V."""
    assert features.dim() == 3, "features must be [B, V, D]"
    b, v, d = features.shape
    contrast = features.reshape(b * v, d)
    return contrast, b, v, d


def _supcon_core(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    base_temperature: float,
    anchor_weight: Optional[torch.Tensor] = None,
    class_balance: bool = False,
    return_per_anchor: bool = False,
):
    """Shared SupCon machinery (Khosla et al.), with optional weighting.

    Parameters
    ----------
    anchor_weight : [B] or None
        Per-anchor multiplicative weight (used by the imbalance-aware variant).
    class_balance : bool
        If True, average per-anchor losses within each class first, then average
        across classes (Balanced Contrastive behavior).
    """
    device = features.device
    contrast, b, v, d = _flatten_views(features)

    # similarity logits among all 2..V views
    anchor_dot = torch.matmul(contrast, contrast.T) / temperature
    # numerical stability
    logits_max, _ = anchor_dot.max(dim=1, keepdim=True)
    logits = anchor_dot - logits_max.detach()

    # label mask over the [B*V, B*V] grid
    labels = labels.contiguous().view(-1, 1)
    label_eq = torch.eq(labels, labels.T).float().to(device)         # [B, B]
    mask = label_eq.repeat(v, v)                                     # [B*V, B*V]

    # exclude self-contrast
    logits_mask = torch.ones_like(mask)
    idx = torch.arange(b * v, device=device)
    logits_mask[idx, idx] = 0
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_per_anchor = mask.sum(dim=1)
    valid = pos_per_anchor > 0
    mean_log_prob_pos = torch.zeros_like(pos_per_anchor)
    mean_log_prob_pos[valid] = (
        (mask * log_prob).sum(dim=1)[valid] / pos_per_anchor[valid]
    )

    per_anchor = -(base_temperature / temperature) * mean_log_prob_pos  # [B*V]

    if anchor_weight is not None:
        w = anchor_weight.repeat(v)
        per_anchor = per_anchor * w

    if class_balance:
        # average within class across the flattened anchors, then across classes
        flat_labels = labels.view(-1).repeat(v)
        uniq = torch.unique(flat_labels)
        losses = []
        for c in uniq:
            sel = (flat_labels == c) & valid
            if sel.any():
                losses.append(per_anchor[sel].mean())
        loss = torch.stack(losses).mean() if losses else per_anchor.new_zeros(())
    else:
        loss = per_anchor[valid].mean() if valid.any() else per_anchor.new_zeros(())

    if return_per_anchor:
        return loss, per_anchor, valid, mask, log_prob, pos_per_anchor
    return loss


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return _supcon_core(features, labels, self.temperature, self.base_temperature)


class BalancedContrastiveLoss(nn.Module):
    """Class-balanced contrastive baseline (BCL-style class averaging)."""

    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return _supcon_core(
            features, labels, self.temperature, self.base_temperature,
            class_balance=True,
        )


class ImbalanceAwareSupConLoss(nn.Module):
    """SupCon with minority weighting + optional focal-style modulation.

    Per-anchor weight:
        w_i = class_weight[y_i] * (focal term)
    where ``class_weight`` is inverse-frequency in the batch (or an explicit
    minority weight), and the focal term ``(1 - p_pos_i) ** gamma`` down-weights
    anchors whose positives are already well aligned (``p_pos`` = mean softmax
    mass on positives). Set ``use_focal_weighting=False`` to disable.
    """

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.temperature = cfg.temperature
        self.base_temperature = cfg.base_temperature

    def _class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        device = labels.device
        uniq, counts = torch.unique(labels, return_counts=True)
        inv = {int(c): 1.0 / float(n) for c, n in zip(uniq, counts)}
        # normalize so weights average ~1
        mean_inv = sum(inv.values()) / len(inv)
        inv = {c: w / mean_inv for c, w in inv.items()}
        if self.cfg.minority_weight > 0 and self.cfg.minority_class in inv:
            inv[self.cfg.minority_class] = self.cfg.minority_weight
        return torch.tensor(
            [inv[int(y)] for y in labels], device=device, dtype=torch.float32
        )

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        class_w = self._class_weights(labels)

        if not self.cfg.use_focal_weighting:
            return _supcon_core(
                features, labels, self.temperature, self.base_temperature,
                anchor_weight=class_w,
            )

        # First pass to get per-anchor positive mass for the focal term.
        _, _, valid, mask, log_prob, pos_per_anchor = _supcon_core(
            features, labels, self.temperature, self.base_temperature,
            return_per_anchor=True,
        )
        # p_pos_i = exp(mean_log_prob_pos_i) approximates avg prob on positives
        b = labels.shape[0]
        v = features.shape[1]
        mean_log_prob_pos = torch.zeros(b * v, device=features.device)
        valid_mask = pos_per_anchor > 0
        mean_log_prob_pos[valid_mask] = (
            (mask * log_prob).sum(dim=1)[valid_mask] / pos_per_anchor[valid_mask]
        )
        p_pos = torch.exp(mean_log_prob_pos).clamp(0.0, 1.0)
        focal = (1.0 - p_pos) ** self.cfg.focal_gamma           # [B*V]
        # collapse focal back to per-anchor [B] by averaging across views
        focal_b = focal.view(v, b).mean(dim=0)
        anchor_weight = class_w * focal_b
        return _supcon_core(
            features, labels, self.temperature, self.base_temperature,
            anchor_weight=anchor_weight,
        )


class SimCLRLoss(nn.Module):
    """Unsupervised NT-Xent: positives are the other views of the same anchor."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        # Treat each anchor as its own class -> SupCon with identity labels.
        b = features.shape[0]
        pseudo = torch.arange(b, device=features.device)
        return _supcon_core(features, pseudo, self.temperature, self.temperature)


def build_loss(cfg: LossConfig) -> nn.Module:
    if cfg.name == "supcon":
        return SupConLoss(cfg.temperature, cfg.base_temperature)
    if cfg.name == "balanced_supcon":
        return BalancedContrastiveLoss(cfg.temperature, cfg.base_temperature)
    if cfg.name == "imbalance_supcon":
        return ImbalanceAwareSupConLoss(cfg)
    if cfg.name == "simclr":
        return SimCLRLoss(cfg.temperature)
    raise ValueError(f"Unknown loss {cfg.name!r}")
