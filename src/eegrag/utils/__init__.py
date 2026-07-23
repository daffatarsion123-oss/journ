"""Utilities: reproducibility, backend detection, IO, logging."""
from .seed import seed_everything, worker_init_fn
from .backend import (
    BackendInfo,
    detect_backend,
    resolve_torch_device,
    has_faiss,
    has_cuml,
    has_thundersvm,
    has_xgboost,
    has_quantum_backend,
)
from .io import (
    ensure_dir,
    save_json,
    load_json,
    save_npz,
    load_npz,
    save_dataframe,
    git_revision,
)
from .logging import get_logger
from .profiling import (
    ResourceTracker,
    ComputeLog,
    count_parameters,
    estimate_flops,
    peak_vram_gb,
    reset_peak_vram,
)

__all__ = [
    "seed_everything",
    "worker_init_fn",
    "BackendInfo",
    "detect_backend",
    "resolve_torch_device",
    "has_faiss",
    "has_cuml",
    "has_thundersvm",
    "has_xgboost",
    "has_quantum_backend",
    "ensure_dir",
    "save_json",
    "load_json",
    "save_npz",
    "load_npz",
    "save_dataframe",
    "git_revision",
    "get_logger",
    "ResourceTracker",
    "ComputeLog",
    "count_parameters",
    "estimate_flops",
    "peak_vram_gb",
    "reset_peak_vram",
]
