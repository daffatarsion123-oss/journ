"""Runtime backend capability detection.

The project runs primarily on an AMD MI300X under ROCm, so we must NOT assume
CUDA. PyTorch's ROCm build still reports ``torch.cuda.is_available() == True``
and ``device='cuda'`` (HIP masquerades as CUDA), but ``torch.version.hip`` is
set. We surface both facts so callers can choose the right knobs (e.g. XGBoost
``device='cuda'`` only works on NVIDIA, not ROCm).

Everything here degrades gracefully: a missing optional dependency yields
``False``/``None`` rather than an import error.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Optional


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def has_faiss() -> bool:
    return _installed("faiss") or _installed("faiss_gpu") or _installed("faiss_cpu")


def has_cuml() -> bool:
    return _installed("cuml")


def has_thundersvm() -> bool:
    return _installed("thundersvm")


def has_xgboost() -> bool:
    return _installed("xgboost")


def has_quantum_backend() -> Optional[str]:
    """Return the name of an available quantum simulator backend, else None."""
    if _installed("pennylane"):
        return "pennylane"
    if _installed("qiskit"):
        return "qiskit"
    return None


@dataclass
class BackendInfo:
    torch_available: bool = False
    torch_device: str = "cpu"        # "cuda" (incl. ROCm/HIP) or "cpu"
    is_rocm: bool = False            # True when torch.version.hip is set
    is_cuda: bool = False            # True for genuine NVIDIA CUDA
    n_accelerators: int = 0
    device_name: str = ""
    cuml: bool = False
    faiss: bool = False
    thundersvm: bool = False
    xgboost: bool = False
    quantum: Optional[str] = None

    def summary(self) -> str:
        accel = "none"
        if self.is_rocm:
            accel = f"ROCm/HIP ({self.device_name})"
        elif self.is_cuda:
            accel = f"CUDA ({self.device_name})"
        return (
            f"torch={self.torch_available} accel={accel} "
            f"n={self.n_accelerators} | cuml={self.cuml} faiss={self.faiss} "
            f"thundersvm={self.thundersvm} xgboost={self.xgboost} "
            f"quantum={self.quantum}"
        )

    def supports_xgb_gpu(self) -> bool:
        """XGBoost GPU histogram training. Native ``device='cuda'`` is NVIDIA-only;
        on ROCm there is currently no upstream GPU path, so we report False and
        let the caller fall back to CPU ``hist``."""
        return self.xgboost and self.is_cuda


def detect_backend() -> BackendInfo:
    info = BackendInfo(
        cuml=has_cuml(),
        faiss=has_faiss(),
        thundersvm=has_thundersvm(),
        xgboost=has_xgboost(),
        quantum=has_quantum_backend(),
    )
    try:
        import torch

        info.torch_available = True
        if torch.cuda.is_available():
            info.torch_device = "cuda"
            info.n_accelerators = torch.cuda.device_count()
            try:
                info.device_name = torch.cuda.get_device_name(0)
            except Exception:
                info.device_name = "unknown"
            hip = getattr(torch.version, "hip", None)
            if hip:
                info.is_rocm = True
            else:
                info.is_cuda = True
        else:
            info.torch_device = "cpu"
    except ImportError:
        info.torch_available = False
    return info


def resolve_torch_device(requested: str = "auto") -> str:
    """Map a config ``device`` string to a concrete torch device string."""
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"device must be auto|cuda|cpu, got {requested!r}")
    if requested == "cpu":
        return "cpu"
    info = detect_backend()
    if requested == "cuda":
        if info.torch_device != "cuda":
            raise RuntimeError("device='cuda' requested but no accelerator found")
        return "cuda"
    # auto
    return info.torch_device
