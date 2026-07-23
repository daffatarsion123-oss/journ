"""Contrastive losses for imbalanced seizure representation learning."""
from .supcon import (
    SupConLoss,
    BalancedContrastiveLoss,
    ImbalanceAwareSupConLoss,
    SimCLRLoss,
    build_loss,
)

__all__ = [
    "SupConLoss",
    "BalancedContrastiveLoss",
    "ImbalanceAwareSupConLoss",
    "SimCLRLoss",
    "build_loss",
]
