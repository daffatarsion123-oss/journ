"""Similarity / distance kernels for retrieval scoring.

Ranking note: the RBF kernel ``exp(-gamma * d^2)`` is monotonically decreasing
in Euclidean distance ``d``, so top-k by RBF == top-k by Euclidean. We therefore
retrieve neighbors with a Euclidean index and convert distances to RBF *weights*
for scoring. Cosine uses an inner-product index on L2-normalized vectors.
"""
from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between rows of ``a`` and rows of ``b``."""
    return l2_normalize(a) @ l2_normalize(b).T


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix."""
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    d2 = aa + bb - 2.0 * (a @ b.T)
    return np.sqrt(np.maximum(d2, 0.0))


def rbf_kernel_from_distance(dist: np.ndarray, gamma: float) -> np.ndarray:
    """RBF kernel value from a distance: ``exp(-gamma * d^2)`` in [0, 1]."""
    return np.exp(-gamma * np.square(dist))


def similarity_from_neighbors(
    raw: np.ndarray, metric: str, gamma: float = 1.0
) -> np.ndarray:
    """Convert per-neighbor raw index outputs into a ``higher = more similar``
    similarity in a comparable scale across metrics.

    ``raw`` semantics depend on metric:
      * cosine     -> ``raw`` is already cosine similarity (inner product).
      * euclidean  -> ``raw`` is distance; we return ``1 / (1 + d)``.
      * rbf        -> ``raw`` is distance; we return ``exp(-gamma * d^2)``.
    """
    if metric == "cosine":
        return raw
    if metric == "euclidean":
        return 1.0 / (1.0 + raw)
    if metric == "rbf":
        return rbf_kernel_from_distance(raw, gamma)
    raise ValueError(f"Unknown metric {metric!r}")
