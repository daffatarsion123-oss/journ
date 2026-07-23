"""Nearest-neighbor index abstraction: FAISS if available, else sklearn.

Two metrics are supported at the index level:
  * ``"ip"``  inner product on L2-normalized vectors  == cosine retrieval.
  * ``"l2"``  Euclidean distance                      == euclidean / RBF retrieval.

``query`` returns ``(neighbor_indices [Q, k], raw [Q, k])`` where ``raw`` is the
inner product (ip) or the Euclidean distance (l2). Callers turn ``raw`` into a
similarity via :func:`similarity.similarity_from_neighbors`.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from ..utils.backend import has_faiss
from ..utils.logging import get_logger
from .similarity import l2_normalize

log = get_logger(__name__)


class NeighborIndex:
    """Thin wrapper so retrieval code is backend-agnostic."""

    def __init__(self, vectors: np.ndarray, metric: str, backend: str):
        self.metric = metric                       # "ip" | "l2"
        self.backend = backend                     # "faiss" | "sklearn"
        self._normalized = metric == "ip"
        store = l2_normalize(vectors) if self._normalized else vectors
        self._vectors = np.ascontiguousarray(store, dtype=np.float32)
        self._impl = None
        # --- compute-cost instrumentation --- #
        self.build_sec: float = 0.0
        self.query_sec: float = 0.0
        self.n_queries: int = 0
        _t0 = time.perf_counter()
        self._build()
        self.build_sec = time.perf_counter() - _t0

    def _build(self) -> None:
        if self.backend == "faiss":
            import faiss

            d = self._vectors.shape[1]
            if self.metric == "ip":
                index = faiss.IndexFlatIP(d)
            else:
                index = faiss.IndexFlatL2(d)
            index.add(self._vectors)
            self._impl = index
        else:
            from sklearn.neighbors import NearestNeighbors

            metric = "cosine" if self.metric == "ip" else "euclidean"
            nn = NearestNeighbors(metric=metric, algorithm="auto", n_jobs=-1)
            nn.fit(self._vectors)
            self._impl = nn

    @property
    def size(self) -> int:
        return self._vectors.shape[0]

    def query(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = min(k, self.size)
        q = l2_normalize(queries) if self._normalized else queries
        q = np.ascontiguousarray(q, dtype=np.float32)
        _t0 = time.perf_counter()
        if self.backend == "faiss":
            scores, idx = self._impl.search(q, k)
            if self.metric == "l2":
                scores = np.sqrt(np.maximum(scores, 0.0))   # faiss L2 is squared
            out = (idx, scores)
        else:
            # sklearn
            dist, idx = self._impl.kneighbors(q, n_neighbors=k, return_distance=True)
            # sklearn 'cosine' distance == 1 - cosine_sim
            out = (idx, (1.0 - dist) if self.metric == "ip" else dist)
        self.query_sec += time.perf_counter() - _t0
        self.n_queries += int(q.shape[0])
        return out


def build_index(vectors: np.ndarray, metric: str, backend: str = "auto") -> NeighborIndex:
    """Build an index. ``metric`` in {"ip","l2"}; ``backend`` in {auto,faiss,sklearn}."""
    if metric not in {"ip", "l2"}:
        raise ValueError(f"index metric must be ip|l2, got {metric!r}")
    if backend == "auto":
        backend = "faiss" if has_faiss() else "sklearn"
    elif backend == "faiss" and not has_faiss():
        log.warning("FAISS requested but not installed; falling back to sklearn.")
        backend = "sklearn"
    return NeighborIndex(vectors, metric, backend)
