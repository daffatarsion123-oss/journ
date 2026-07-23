"""Leakage-guarded retrieval memory bank.

THE central anti-leakage object. A :class:`MemoryBank` is constructed from
TRAINING embeddings only and refuses (via assertion) to admit any window whose
subject equals the held-out test subject. Every retrieval returns neighbor
labels AND neighbor subject ids so the cross-subject analyses (RQ2) can run.

The bank can optionally subsample the majority class (keeping ALL seizure
windows) to bound cost on millions of windows -- the seizure minority is never
dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..utils.logging import get_logger
from .index import build_index, NeighborIndex
from .similarity import similarity_from_neighbors

log = get_logger(__name__)

# similarity name -> index metric
_METRIC_TO_INDEX = {"cosine": "ip", "euclidean": "l2", "rbf": "l2"}


@dataclass
class RetrievalResult:
    """Neighbors for a batch of queries (all arrays shape ``[Q, k]``)."""

    neighbor_idx: np.ndarray       # row into the bank
    similarity: np.ndarray         # higher == more similar (metric-specific)
    raw: np.ndarray                # raw index output (ip score or l2 distance)
    neighbor_labels: np.ndarray    # bank label of each neighbor {0,1}
    neighbor_subjects: np.ndarray  # bank subject id (object) of each neighbor
    metric: str
    k: int


class MemoryBank:
    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray,
        *,
        global_idx: Optional[np.ndarray] = None,
        test_subject: Optional[str] = None,
    ):
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.labels = np.asarray(labels).astype(np.int64)
        self.subjects = np.asarray(subjects, dtype=object)
        # global bundle indices (for the leakage assertion against a fold)
        self.global_idx = (
            np.asarray(global_idx) if global_idx is not None
            else np.arange(len(self.labels))
        )
        self.test_subject = test_subject
        self._indices: dict = {}            # metric name -> NeighborIndex
        self._assert_no_test_subject()

    # --- leakage guard ---------------------------------------------------- #
    def _assert_no_test_subject(self) -> None:
        if self.test_subject is None:
            return
        present = self.test_subject in set(self.subjects.tolist())
        assert not present, (
            f"[memory-bank leakage] test subject {self.test_subject!r} found in "
            f"the memory bank ({int((self.subjects == self.test_subject).sum())} "
            f"windows). The bank MUST contain training subjects only."
        )

    @property
    def size(self) -> int:
        return self.embeddings.shape[0]

    @property
    def n_pos(self) -> int:
        return int(self.labels.sum())

    @property
    def n_subjects(self) -> int:
        return len(set(self.subjects.tolist()))

    # --- compute-cost accessors (summed across all built indices) --------- #
    @property
    def index_build_sec(self) -> float:
        return float(sum(ix.build_sec for ix in self._indices.values()))

    @property
    def query_sec(self) -> float:
        return float(sum(ix.query_sec for ix in self._indices.values()))

    # --- index management ------------------------------------------------- #
    def _get_index(self, metric: str, backend: str = "auto") -> NeighborIndex:
        idx_metric = _METRIC_TO_INDEX[metric]
        if idx_metric not in self._indices:
            log.info(
                "Building %s index over %d bank vectors (metric=%s)",
                idx_metric, self.size, metric,
            )
            self._indices[idx_metric] = build_index(
                self.embeddings, idx_metric, backend
            )
        return self._indices[idx_metric]

    def retrieve(
        self, queries: np.ndarray, k: int, metric: str = "cosine",
        gamma: float = 1.0, backend: str = "auto",
    ) -> RetrievalResult:
        if metric not in _METRIC_TO_INDEX:
            raise ValueError(f"Unknown metric {metric!r}")
        index = self._get_index(metric, backend)
        nidx, raw = index.query(queries, k)
        sim = similarity_from_neighbors(raw, metric, gamma)
        return RetrievalResult(
            neighbor_idx=nidx,
            similarity=sim,
            raw=raw,
            neighbor_labels=self.labels[nidx],
            neighbor_subjects=self.subjects[nidx],
            metric=metric,
            k=k,
        )


def _subsample_majority(
    labels: np.ndarray,
    subjects: np.ndarray,
    max_per_class: Optional[int],
    seed: int,
) -> np.ndarray:
    """Return row indices keeping ALL seizures and capping the majority class."""
    if not max_per_class:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        rows = np.flatnonzero(labels == c)
        if c == 1:                       # seizure: keep all
            keep.append(rows)
        elif len(rows) > max_per_class:
            keep.append(rng.choice(rows, size=max_per_class, replace=False))
        else:
            keep.append(rows)
    out = np.sort(np.concatenate(keep))
    return out


def build_memory_bank(
    embeddings: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    *,
    global_idx: Optional[np.ndarray] = None,
    test_subject: Optional[str] = None,
    max_per_class: Optional[int] = None,
    seed: int = 0,
) -> MemoryBank:
    """Build a memory bank from TRAINING data, with optional majority capping."""
    sel = _subsample_majority(np.asarray(labels), np.asarray(subjects),
                              max_per_class, seed)
    gi = None if global_idx is None else np.asarray(global_idx)[sel]
    bank = MemoryBank(
        embeddings=np.asarray(embeddings)[sel],
        labels=np.asarray(labels)[sel],
        subjects=np.asarray(subjects)[sel],
        global_idx=gi,
        test_subject=test_subject,
    )
    log.info(
        "Memory bank: %d windows | %d seizure | %d subjects (test subject excluded=%s)",
        bank.size, bank.n_pos, bank.n_subjects, test_subject,
    )
    return bank
