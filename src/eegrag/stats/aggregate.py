"""Aggregate fold/seed-level metrics into the 'Mean +/- Std' journal format."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "n": int(v.size),
    }


def format_mean_std(values: Sequence[float], digits: int = 4) -> str:
    ms = mean_std(values)
    if ms["n"] == 0:
        return "n/a"
    return f"{ms['mean']:.{digits}f} +/- {ms['std']:.{digits}f}"


def summarize_folds(
    per_fold_metrics: List[Dict[str, float]], keys: Sequence[str] = None
) -> Dict[str, Dict[str, float]]:
    """Given a list of per-fold metric dicts, return ``key -> {mean,std,n}``."""
    if not per_fold_metrics:
        return {}
    keys = list(keys) if keys else sorted(
        {k for d in per_fold_metrics for k in d}
    )
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        out[k] = mean_std([d.get(k) for d in per_fold_metrics])
    return out
