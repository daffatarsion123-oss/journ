"""Configuration dataclasses and YAML loader."""
from .config import (
    DataConfig,
    PreprocessingConfig,
    AugmentationConfig,
    EncoderConfig,
    LossConfig,
    TrainingConfig,
    RetrievalConfig,
    QuantumConfig,
    ClassicalConfig,
    StatsConfig,
    ExperimentConfig,
)
from .loader import load_config, dump_config, merge_overrides

__all__ = [
    "DataConfig",
    "PreprocessingConfig",
    "AugmentationConfig",
    "EncoderConfig",
    "LossConfig",
    "TrainingConfig",
    "RetrievalConfig",
    "QuantumConfig",
    "ClassicalConfig",
    "StatsConfig",
    "ExperimentConfig",
    "load_config",
    "dump_config",
    "merge_overrides",
]
