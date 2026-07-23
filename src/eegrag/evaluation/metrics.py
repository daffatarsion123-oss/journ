"""Imbalance-aware evaluation metrics for seizure detection.

PRIMARY metrics (the journal headline):
  * ``auc_pr``                 -- area under the precision-recall curve (AP).
  * ``sensitivity`` / recall   -- seizure-class recall = TP / (TP + FN).
  * ``f1_seizure``             -- per-class F1 for the seizure class.
  * ``precision_seizure``      -- seizure-class precision.
  * ``fp_per_hour``            -- false positives per hour (clinical relevance).

SECONDARY (reported but NOT headline, per the methodology constraints):
  * ``accuracy``, ``roc_auc`` -- included for completeness only.

Per-class F1 is reported separately for both classes so the majority-class
collapse seen in the conference paper is visible.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

METRIC_KEYS = (
    "auc_pr",
    "sensitivity",
    "precision_seizure",
    "f1_seizure",
    "f1_nonseizure",
    "f1_macro",
    "specificity",
    "fp_per_hour",
    "balanced_accuracy",
    "accuracy",        # secondary
    "roc_auc",         # secondary
    "n",
    "n_pos",
    "threshold",
)


def false_positives_per_hour(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_windows: Optional[int] = None,
    step_size_sec: float = 1.0,
    total_hours: Optional[float] = None,
) -> float:
    """False positives per hour.

    With a sliding window of stride ``step_size_sec`` (1 s here), each window is
    one decision; recording duration ~= n_windows * step / 3600 unless an exact
    ``total_hours`` is supplied. FP = predicted seizure on a non-seizure window.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    if total_hours is None:
        n = n_windows if n_windows is not None else len(y_true)
        total_hours = (n * step_size_sec) / 3600.0
    if total_hours <= 0:
        return float("nan")
    return fp / total_hours


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
    step_size_sec: float = 1.0,
    total_hours: Optional[float] = None,
) -> Dict[str, float]:
    """Compute the full metric panel from continuous ``scores`` and a threshold.

    ``scores`` are P(seizure)-like values in [0, 1] (or any monotone score for
    AUC-PR / ROC-AUC); the binary metrics use ``scores >= threshold``.
    """
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        precision_recall_fscore_support,
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
    )

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    y_pred = (scores >= threshold).astype(int)

    n = len(y_true)
    n_pos = int(y_true.sum())
    has_both = n_pos > 0 and n_pos < n

    out: Dict[str, float] = {"n": n, "n_pos": n_pos, "threshold": float(threshold)}

    out["auc_pr"] = (
        float(average_precision_score(y_true, scores)) if has_both else float("nan")
    )
    out["roc_auc"] = (
        float(roc_auc_score(y_true, scores)) if has_both else float("nan")
    )

    # per-class precision/recall/F1 (labels 0 and 1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    out["precision_seizure"] = float(prec[1])
    out["sensitivity"] = float(rec[1])           # recall of seizure class
    out["f1_seizure"] = float(f1[1])
    out["f1_nonseizure"] = float(f1[0])
    out["f1_macro"] = float((f1[0] + f1[1]) / 2.0)

    # specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["fp_per_hour"] = false_positives_per_hour(
        y_true, y_pred, n_windows=n, step_size_sec=step_size_sec,
        total_hours=total_hours,
    )
    return out
