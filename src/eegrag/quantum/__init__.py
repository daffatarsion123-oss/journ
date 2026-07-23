"""Exploratory quantum-simulated kernel retrieval (optional, simulator-only)."""
from .kernel import (
    QuantumKernel,
    build_quantum_kernel,
    quantum_retrieval,
    quantum_backend_available,
)

__all__ = [
    "QuantumKernel",
    "build_quantum_kernel",
    "quantum_retrieval",
    "quantum_backend_available",
]
