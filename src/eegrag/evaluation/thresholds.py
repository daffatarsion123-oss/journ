"""Decision-threshold tuning -- MUST be fit on training-subject scores only.

The runners compute scores for training queries (leave-self-out against the
memory bank, or an inner subject-held-out split) and call :func:`tune_threshold`
with THOSE scores. The chosen threshold is then frozen and applied to the
held-out test subject. The test subject's labels never influence the threshold.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def tune_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    criterion: str = "f1",
    target_sensitivity: Optional[float] = None,
    grid: int = 256,
) -> float:
    """Return a decision threshold maximizing the chosen criterion.

    criterion:
        "f1"             maximize seizure-class F1.
        "youden"         maximize Youden's J = sensitivity + specificity - 1.
        "sensitivity_at" smallest threshold achieving >= target_sensitivity
                         (raises if target_sensitivity is None).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5  # degenerate; nothing to tune

    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return lo
    thresholds = np.linspace(lo, hi, grid)

    pos = y_true == 1
    neg = ~pos
    n_pos = pos.sum()
    n_neg = neg.sum()

    best_t, best_val = 0.5, -np.inf
    for t in thresholds:
        pred = scores >= t
        tp = np.sum(pred & pos)
        fp = np.sum(pred & neg)
        fn = n_pos - tp
        tn = n_neg - fp
        sens = tp / n_pos if n_pos else 0.0
        spec = tn / n_neg if n_neg else 0.0

        if criterion == "f1":
            denom = 2 * tp + fp + fn
            val = (2 * tp / denom) if denom > 0 else 0.0
        elif criterion == "youden":
            val = sens + spec - 1.0
        elif criterion == "sensitivity_at":
            if target_sensitivity is None:
                raise ValueError("criterion 'sensitivity_at' needs target_sensitivity")
            # prefer the highest threshold that still meets the sensitivity floor
            # (fewest false positives) -> maximize threshold s.t. sens >= target
            val = t if sens >= target_sensitivity else -np.inf
        else:
            raise ValueError(f"Unknown criterion {criterion!r}")

        if val > best_val:
            best_val, best_t = val, float(t)
    return best_t


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores) >= threshold).astype(int)
