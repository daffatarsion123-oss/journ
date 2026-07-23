"""Retrieval: leakage-guarded memory bank, index, similarities, decision heads."""
from .similarity import (
    cosine_similarity,
    euclidean_distance,
    rbf_kernel_from_distance,
    similarity_from_neighbors,
)
from .index import NeighborIndex, build_index
from .memory_bank import MemoryBank, build_memory_bank, RetrievalResult
from .heads import (
    knn_seizure_score,
    hybrid_score,
    neighbor_analysis,
    neighbor_overlap,
    NeighborStats,
)

__all__ = [
    "cosine_similarity",
    "euclidean_distance",
    "rbf_kernel_from_distance",
    "similarity_from_neighbors",
    "NeighborIndex",
    "build_index",
    "MemoryBank",
    "build_memory_bank",
    "RetrievalResult",
    "knn_seizure_score",
    "hybrid_score",
    "neighbor_analysis",
    "neighbor_overlap",
    "NeighborStats",
]
