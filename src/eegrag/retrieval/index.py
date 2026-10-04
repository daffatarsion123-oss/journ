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
from ..utils.arrays import is_cuda_array, to_tensor

log = get_logger(__name__)


class TorchNeighborIndex:
    """Exact search using bounded tiles, without constructing the full Q x N matrix."""

    def __init__(self, vectors, metric, device="cuda", query_chunk_size=512,
                 bank_chunk_size=32768):
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU retrieval requested but CUDA is unavailable")
        self.metric, self.backend = metric, "torch"
        self.query_chunk_size, self.bank_chunk_size = query_chunk_size, bank_chunk_size
        if query_chunk_size < 1 or bank_chunk_size < 1 or len(vectors) < 1:
            raise ValueError("Index and tile sizes must be positive")
        started = time.perf_counter()
        self.vectors = to_tensor(vectors, device).to(dtype=torch.float32)
        if metric == "ip":
            self.vectors = self.vectors / self.vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.norms = self.vectors.square().sum(dim=1)
        if device == "cuda":
            torch.cuda.synchronize()
        self.build_sec = time.perf_counter() - started
        self.query_sec, self.n_queries = 0.0, 0

    @property
    def size(self):
        return len(self.vectors)

    def query(self, queries, k):
        import torch
        k = min(k, self.size)
        if k < 1:
            raise ValueError("k must be positive")
        indices, values = [], []
        started = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=False):
            for first in range(0, len(queries), self.query_chunk_size):
                q = to_tensor(queries[first:first + self.query_chunk_size], self.vectors.device).to(dtype=torch.float32)
                if self.metric == "ip":
                    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                best = torch.empty((len(q), 0), device=q.device)
                best_idx = torch.empty((len(q), 0), device=q.device, dtype=torch.long)
                qnorm = q.square().sum(dim=1, keepdim=True)
                for start in range(0, self.size, self.bank_chunk_size):
                    bank = self.vectors[start:start + self.bank_chunk_size]
                    score = q @ bank.T
                    if self.metric == "l2":
                        score = -(qnorm + self.norms[start:start + len(bank)] - 2 * score).clamp_min(0)
                    local, loc_idx = score.topk(min(k, len(bank)), dim=1)
                    combined = torch.cat((best, local), dim=1)
                    combined_idx = torch.cat((best_idx, loc_idx + start), dim=1)
                    best, positions = combined.topk(min(k, combined.shape[1]), dim=1)
                    best_idx = combined_idx.gather(1, positions)
                if self.metric == "l2":
                    best = (-best).clamp_min(0).sqrt()
                indices.append(best_idx.cpu().numpy())
                values.append(best.cpu().numpy())
        self.query_sec += time.perf_counter() - started
        self.n_queries += len(queries)
        if not indices:
            return np.empty((0, k), dtype=np.int64), np.empty((0, k), dtype=np.float32)
        return np.concatenate(indices), np.concatenate(values)


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


def build_index(vectors: np.ndarray, metric: str, backend: str = "auto",
                query_chunk_size=512, bank_chunk_size=32768) -> NeighborIndex:
    """Build an index. ``metric`` in {"ip","l2"}; ``backend`` in {auto,faiss,sklearn}."""
    if metric not in {"ip", "l2"}:
        raise ValueError(f"index metric must be ip|l2, got {metric!r}")
    if backend == "torch":
        return TorchNeighborIndex(vectors, metric, query_chunk_size=query_chunk_size,
                                  bank_chunk_size=bank_chunk_size)
    if is_cuda_array(vectors):
        raise ValueError("GPU resident embeddings require retrieval.backend=torch")
    if backend not in {"auto", "faiss", "sklearn"}:
        raise ValueError(f"Unknown retrieval backend {backend!r}")
    if backend == "auto":
        backend = "faiss" if has_faiss() else "sklearn"
    elif backend == "faiss" and not has_faiss():
        log.warning("FAISS requested but not installed; falling back to sklearn.")
        backend = "sklearn"
    return NeighborIndex(vectors, metric, backend)
