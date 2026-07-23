"""Retrieval decision heads and neighbor analysis (RQ2).

Heads
-----
* :func:`knn_seizure_score` -- similarity-weighted fraction of seizure neighbors
  (the k-NN memory-bank head). Returns a score in [0, 1].
* :func:`hybrid_score`      -- convex blend of a linear-classifier probability and
  the k-NN seizure score.

Analysis (for RQ2 / neighbor study)
-----------------------------------
* :func:`neighbor_analysis` -- per-query and aggregate statistics:
    - seizure-neighbor ratio,
    - same-subject vs cross-subject neighbor fractions,
    - subject diversity among retrieved neighbors (unique subjects / k),
    - whether seizure predictions are supported by seizure-like neighbors.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

from .memory_bank import RetrievalResult


def knn_seizure_score(
    result: RetrievalResult, weighted: bool = True, eps: float = 1e-12
) -> np.ndarray:
    """Seizure score per query in [0, 1].

    weighted=True  -> similarity-weighted fraction of seizure neighbors.
    weighted=False -> plain fraction of seizure neighbors (majority-vote style).
    """
    labels = result.neighbor_labels.astype(np.float64)     # [Q, k]
    if not weighted:
        return labels.mean(axis=1)
    sim = result.similarity.astype(np.float64)
    # clip negative cosine similarities to 0 so they don't flip the weighting
    w = np.clip(sim, 0.0, None)
    num = (w * labels).sum(axis=1)
    den = w.sum(axis=1) + eps
    return num / den


def hybrid_score(
    linear_prob: np.ndarray, knn_score: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """alpha * linear_prob + (1 - alpha) * knn_score."""
    return alpha * np.asarray(linear_prob) + (1.0 - alpha) * np.asarray(knn_score)


@dataclass
class NeighborStats:
    """Aggregate neighbor statistics over a set of queries."""

    k: int
    metric: str
    n_queries: int
    mean_seizure_neighbor_ratio: float
    mean_same_subject_fraction: float
    mean_cross_subject_fraction: float
    mean_subject_diversity: float        # unique subjects / k, averaged
    # support analysis (conditioned on the model predicting seizure)
    seizure_pred_seizure_support: float  # mean seizure-neighbor ratio | pred==1
    nonseizure_pred_seizure_support: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def neighbor_analysis(
    result: RetrievalResult,
    query_subjects: np.ndarray,
    predictions: Optional[np.ndarray] = None,
) -> NeighborStats:
    """Compute aggregate neighbor statistics for the cross-subject study.

    Parameters
    ----------
    query_subjects : [Q]
        Subject id of each query window (the held-out test subject for LOSO).
        Under strict LOSO this never matches a bank subject, so
        ``same_subject_fraction`` is expected to be ~0 -- which is itself a
        finding (cross-subject retrieval is forced). The machinery still
        supports same-subject analysis for non-LOSO ablations.
    predictions : [Q] in {0,1}, optional
        Binarized seizure predictions for the support analysis.
    """
    labels = result.neighbor_labels                      # [Q, k]
    subs = result.neighbor_subjects                      # [Q, k]
    q = np.asarray(query_subjects).reshape(-1, 1)        # [Q, 1]
    k = result.k

    seizure_ratio = labels.mean(axis=1)                  # [Q]
    same_subject = (subs == q)                           # [Q, k] bool
    same_frac = same_subject.mean(axis=1)
    cross_frac = 1.0 - same_frac

    diversity = np.asarray(
        [len(set(row.tolist())) / max(k, 1) for row in subs], dtype=np.float64
    )

    if predictions is not None:
        predictions = np.asarray(predictions).astype(int)
        pos = predictions == 1
        neg = ~pos
        sup_pos = float(seizure_ratio[pos].mean()) if pos.any() else float("nan")
        sup_neg = float(seizure_ratio[neg].mean()) if neg.any() else float("nan")
    else:
        sup_pos = sup_neg = float("nan")

    return NeighborStats(
        k=k,
        metric=result.metric,
        n_queries=int(labels.shape[0]),
        mean_seizure_neighbor_ratio=float(seizure_ratio.mean()),
        mean_same_subject_fraction=float(same_frac.mean()),
        mean_cross_subject_fraction=float(cross_frac.mean()),
        mean_subject_diversity=float(diversity.mean()),
        seizure_pred_seizure_support=sup_pos,
        nonseizure_pred_seizure_support=sup_neg,
    )


def neighbor_overlap(
    result_a: RetrievalResult, result_b: RetrievalResult
) -> float:
    """Mean Jaccard overlap of retrieved neighbor sets between two metrics.

    Used to compare cosine / RBF / quantum retrieval (do they surface the same
    neighbors?). Both results must come from the SAME bank and query order.
    """
    a, b = result_a.neighbor_idx, result_b.neighbor_idx
    assert a.shape[0] == b.shape[0], "results must align on queries"
    jac = []
    for ra, rb in zip(a, b):
        sa, sb = set(ra.tolist()), set(rb.tolist())
        union = len(sa | sb)
        jac.append(len(sa & sb) / union if union else 0.0)
    return float(np.mean(jac))
