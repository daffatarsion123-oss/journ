"""YAML <-> dataclass loading with safe nested overrides.

Usage::

    cfg = load_config("configs/contrastive.yaml",
                      overrides=["data.features_dir=/path/to/parquet",
                                 "training.epochs=2"])

Only keys that already exist on the dataclasses are accepted; unknown keys raise
so typos in YAML/CLI fail loudly instead of being silently ignored.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, get_type_hints

import yaml

from .config import (
    ExperimentConfig,
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
)

_SUBCONFIGS = {
    "data": DataConfig,
    "preprocessing": PreprocessingConfig,
    "augmentation": AugmentationConfig,
    "encoder": EncoderConfig,
    "loss": LossConfig,
    "training": TrainingConfig,
    "retrieval": RetrievalConfig,
    "quantum": QuantumConfig,
    "classical": ClassicalConfig,
    "stats": StatsConfig,
}


def _coerce_tuples(cls: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    """YAML lists -> tuples where the dataclass field annotates a Tuple."""
    if not dataclasses.is_dataclass(cls):
        return data
    out = dict(data)
    for f in dataclasses.fields(cls):
        if f.name in out and isinstance(out[f.name], list):
            ann = str(f.type)
            if "Tuple" in ann or "tuple" in ann:
                out[f.name] = tuple(out[f.name])
    return out


def _build_subconfig(cls: Any, raw: Optional[Dict[str, Any]]) -> Any:
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(raw)}")
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**_coerce_tuples(cls, raw))


def _parse_scalar(text: str) -> Any:
    """Parse a CLI override value using YAML scalar rules (int/float/bool/str)."""
    return yaml.safe_load(text)


def merge_overrides(cfg_dict: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` dotted overrides onto a nested dict, in place-ish."""
    out = dict(cfg_dict)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, _, value = item.partition("=")
        parts = key.strip().split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot descend into non-mapping at {p!r}")
        node[parts[-1]] = _parse_scalar(value.strip())
    return out


def load_config(
    path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    base: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from YAML with optional CLI overrides."""
    cfg_dict: Dict[str, Any] = dict(base or {})
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        cfg_dict.update(loaded)
    if overrides:
        cfg_dict = merge_overrides(cfg_dict, overrides)

    top_valid = {f.name for f in dataclasses.fields(ExperimentConfig)}
    unknown = set(cfg_dict) - top_valid
    if unknown:
        raise KeyError(f"Unknown top-level config keys: {sorted(unknown)}")

    kwargs: Dict[str, Any] = {}
    for key, value in cfg_dict.items():
        if key in _SUBCONFIGS:
            kwargs[key] = _build_subconfig(_SUBCONFIGS[key], value)
        else:
            kwargs[key] = value

    # Coerce top-level tuple fields (seeds) too.
    kwargs = _coerce_tuples(ExperimentConfig, kwargs)
    return ExperimentConfig(**kwargs)


def dump_config(cfg: ExperimentConfig, path: str) -> None:
    """Serialize a config to YAML (round-trippable through :func:`load_config`)."""
    def _clean(obj: Any) -> Any:
        if isinstance(obj, tuple):
            return [_clean(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        return obj

    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(_clean(cfg.to_dict()), fh, sort_keys=False)
