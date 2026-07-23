"""Torch datasets and physiology-preserving augmentations."""
from .augment import FeatureAugmenter, build_augmenter
from .dataset import (
    EEGWindowDataset,
    ContrastiveViewDataset,
    make_balanced_sampler,
)

__all__ = [
    "FeatureAugmenter",
    "build_augmenter",
    "EEGWindowDataset",
    "ContrastiveViewDataset",
    "make_balanced_sampler",
]
