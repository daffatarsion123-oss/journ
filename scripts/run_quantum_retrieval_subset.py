#!/usr/bin/env python
"""Exploratory quantum-kernel retrieval on a compressed, test-subject-free bank.

Simulator-only; falls back to a classical RBF kernel (clearly labeled) if no
quantum backend (pennylane / qiskit) is installed.

Example:
    python scripts/run_quantum_retrieval_subset.py --config configs/quantum_subset.yaml \
        --data-dir /path/to/outputs \
        --embeddings-dir outputs/contrastive_supcon/embeddings \
        --set quantum.enabled=true
"""
from _bootstrap import base_parser, load


def main() -> None:
    parser = base_parser(__doc__)
    args = parser.parse_args()
    cfg = load(args)
    from eegrag.experiments import run_quantum_retrieval_subset
    out = run_quantum_retrieval_subset(cfg, embeddings_dir=args.embeddings_dir)
    print(f"Quantum subset retrieval done: {out['n_results']} entries -> "
          f"{out['output_dir']}")


if __name__ == "__main__":
    main()
