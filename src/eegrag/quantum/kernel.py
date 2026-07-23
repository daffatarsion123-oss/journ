"""Exploratory quantum-simulated kernel similarity for retrieval.

Pipeline (per :class:`QuantumConfig`):

    embedding -> (already PCA/projected to ``projection_dim`` <= n_qubits)
              -> angle-encoded quantum feature map (ZZ / Z / Pauli)
              -> fidelity kernel  K(x, y) = |<phi(x)|phi(y)>|^2
              -> retrieval ranking by kernel value (higher == more similar)

Design constraints honored:
  * **Simulator only.** No real-hardware calls.
  * **Optional + modular.** If neither PennyLane nor Qiskit is importable,
    ``quantum_backend_available()`` is False and :func:`build_quantum_kernel`
    returns a CLASSICAL RBF fallback so the rest of the pipeline still runs.
  * **Cost-bounded.** The fidelity kernel is O(N_bank * N_query) circuit
    evaluations; callers must compress the bank (``max_bank_per_class``) and cap
    queries (``max_query``). The quantum head is never on the critical path of
    the full LOSO sweep.

This is intentionally small and self-contained -- a place to *probe* whether a
quantum feature map yields complementary neighbor structure (RQ: quantum vs
cosine/RBF neighbor overlap), not a production retrieval engine.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..config import QuantumConfig
from ..utils.backend import has_quantum_backend
from ..utils.logging import get_logger

log = get_logger(__name__)


def quantum_backend_available() -> Optional[str]:
    return has_quantum_backend()


class QuantumKernel:
    """Fidelity kernel over an angle-encoded feature map.

    ``is_quantum`` is False when running the classical RBF fallback, so callers
    can label results honestly in the paper ("quantum (simulated)" vs
    "classical-fallback").
    """

    def __init__(self, cfg: QuantumConfig, backend: str):
        self.cfg = cfg
        self.backend = backend                  # "pennylane" | "qiskit" | "rbf"
        self.is_quantum = backend in {"pennylane", "qiskit"}
        self._qnode = None
        if self.backend == "pennylane":
            self._init_pennylane()
        elif self.backend == "qiskit":
            self._init_qiskit()

    # --- backends --------------------------------------------------------- #
    def _init_pennylane(self) -> None:
        import pennylane as qml

        n = self.cfg.n_qubits
        dev = qml.device("default.qubit", wires=n)

        def feature_map(x):
            for i in range(n):
                qml.Hadamard(wires=i)
            for _ in range(self.cfg.reps):
                for i in range(min(len(x), n)):
                    qml.RZ(x[i], wires=i)
                if self.cfg.feature_map == "zz":
                    for i in range(n - 1):
                        qml.CNOT(wires=[i, i + 1])
                        qml.RZ(x[i % len(x)] * x[(i + 1) % len(x)], wires=i + 1)
                        qml.CNOT(wires=[i, i + 1])

        @qml.qnode(dev)
        def kernel_circuit(x1, x2):
            feature_map(x1)
            qml.adjoint(feature_map)(x2)
            return qml.probs(wires=range(n))

        self._qnode = kernel_circuit

    def _init_qiskit(self) -> None:
        # Lazy import; we use statevector overlap as the fidelity kernel.
        from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
        from qiskit.quantum_info import Statevector

        n = self.cfg.projection_dim
        if self.cfg.feature_map == "z":
            self._fmap = ZFeatureMap(feature_dimension=n, reps=self.cfg.reps)
        else:
            self._fmap = ZZFeatureMap(feature_dimension=n, reps=self.cfg.reps)
        self._Statevector = Statevector

    # --- kernel evaluation ------------------------------------------------ #
    def _pair_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        if self.backend == "pennylane":
            probs = self._qnode(x1, x2)
            return float(probs[0])              # P(all-zeros) == fidelity
        if self.backend == "qiskit":
            sv1 = self._Statevector(self._fmap.assign_parameters(x1))
            sv2 = self._Statevector(self._fmap.assign_parameters(x2))
            return float(np.abs(sv1.inner(sv2)) ** 2)
        raise RuntimeError("pair kernel only valid for quantum backends")

    def kernel_matrix(self, queries: np.ndarray, bank: np.ndarray) -> np.ndarray:
        """[Qq, Qb] fidelity kernel between scaled query and bank vectors."""
        queries = self._scale_angles(queries)
        bank = self._scale_angles(bank)
        if self.backend == "rbf":
            from ..retrieval.similarity import euclidean_distance, rbf_kernel_from_distance
            d = euclidean_distance(queries, bank)
            return rbf_kernel_from_distance(d, gamma=1.0)
        q, b = queries.shape[0], bank.shape[0]
        K = np.empty((q, b), dtype=np.float64)
        for i in range(q):
            for j in range(b):
                K[i, j] = self._pair_kernel(queries[i], bank[j])
        return K

    def _scale_angles(self, x: np.ndarray) -> np.ndarray:
        """Map features into a sensible angle range [-pi, pi] per dimension."""
        x = np.asarray(x, dtype=np.float64)[:, : self.cfg.n_qubits]
        # robust per-batch scaling; angle encoding is periodic so we tanh-squash
        return np.pi * np.tanh(x)


def build_quantum_kernel(cfg: QuantumConfig) -> QuantumKernel:
    """Return a QuantumKernel, falling back to classical RBF if unavailable."""
    requested = cfg.backend
    available = quantum_backend_available()
    if not cfg.enabled:
        log.info("Quantum module disabled by config; using RBF fallback kernel.")
        return QuantumKernel(cfg, "rbf")
    if requested == "none":
        return QuantumKernel(cfg, "rbf")
    if requested == "auto":
        backend = available or "rbf"
    else:
        backend = requested if available == requested else (available or "rbf")
    if backend == "rbf":
        log.warning(
            "No quantum backend available (pennylane/qiskit). Falling back to "
            "classical RBF kernel; results are NOT quantum-simulated."
        )
    else:
        log.info("Quantum kernel backend: %s (simulator)", backend)
    return QuantumKernel(cfg, backend)


def quantum_retrieval(
    kernel: QuantumKernel,
    query_embeddings: np.ndarray,
    bank_embeddings: np.ndarray,
    bank_labels: np.ndarray,
    bank_subjects: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rank bank entries by quantum (or fallback) kernel value.

    Returns ``(neighbor_idx, similarity, neighbor_labels, neighbor_subjects)``
    each shaped ``[Q, k]`` -- the same layout as a classical RetrievalResult so
    the neighbor-analysis code is reusable.
    """
    K = kernel.kernel_matrix(query_embeddings, bank_embeddings)  # [Q, B]
    k = min(k, K.shape[1])
    # top-k by descending kernel value
    nidx = np.argpartition(-K, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(K.shape[0])[:, None]
    order = np.argsort(-K[rows, nidx], axis=1)
    nidx = nidx[rows, order]
    sim = K[rows, nidx]
    return nidx, sim, np.asarray(bank_labels)[nidx], np.asarray(bank_subjects)[nidx]
