"""Encoders, heads, and classical baselines."""
from .encoders import (
    build_encoder,
    MLPEncoder,
    CNN1DEncoder,
    TCNEncoder,
    CNN1DTransformerEncoder,
)
from .heads import ProjectionHead, LinearClassifierHead, ContrastiveModel
from .classical import build_classical_model, ClassicalModel, fit_classical_with_search

__all__ = [
    "build_encoder",
    "MLPEncoder",
    "CNN1DEncoder",
    "TCNEncoder",
    "CNN1DTransformerEncoder",
    "ProjectionHead",
    "LinearClassifierHead",
    "ContrastiveModel",
    "build_classical_model",
    "ClassicalModel",
    "fit_classical_with_search",
]
