#!/usr/bin/env python
"""Re-sweep retrieval heads / similarity / k over previously-saved embeddings.

Requires that ``run_loso_contrastive.py`` was run first (it writes per-fold
embeddings to ``outputs/<name>/embeddings/``).

Example:
    python scripts/run_retrieval_eval.py --config configs/retrieval.yaml \
        --data-dir /path/to/outputs \
        --embeddings-dir outputs/contrastive_supcon/embeddings
"""
from _bootstrap import base_parser, load


def main() -> None:
    parser = base_parser(__doc__)
    args = parser.parse_args()
    cfg = load(args)
    if not args.embeddings_dir:
        parser.error("--embeddings-dir is required (point to saved fold embeddings).")
    from eegrag.experiments import run_retrieval_eval
    out = run_retrieval_eval(cfg, embeddings_dir=args.embeddings_dir)
    print(f"Retrieval eval done: {out['n_results']} fold-seed results.")


if __name__ == "__main__":
    main()
