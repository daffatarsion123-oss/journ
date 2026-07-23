"""Filesystem IO helpers for metrics, predictions, embeddings, neighbors."""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, Optional

import numpy as np


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_npz(path: str, **arrays: np.ndarray) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    np.savez_compressed(path, **arrays)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def save_dataframe(df, path: str) -> None:
    """Save a pandas DataFrame, picking format from the extension."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported dataframe extension: {path}")


def git_revision(default: str = "unknown") -> str:
    """Best-effort current git short SHA, for provenance in saved metrics."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or default
    except (OSError, subprocess.SubprocessError):
        pass
    return default
