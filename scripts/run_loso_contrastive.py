#!/usr/bin/env python
"""Run the LOSO contrastive + retrieval experiment.

Examples
--------
Smoke test on two subjects, 2 epochs, CPU:
    python scripts/run_loso_contrastive.py --config configs/contrastive.yaml \
        --data-dir /path/to/outputs \
        --set folds_whitelist='[chb01,chb02]' --set training.epochs=2 \
        --set training.device=cpu --set seeds='[0]'

Full run on the MI300X (ROCm):
    python scripts/run_loso_contrastive.py --config configs/contrastive_rocm.yaml \
        --data-dir /path/to/outputs

A100 training-only phase (BF16 Tensor Cores; copy NPZs to MI300X after):
    python scripts/run_loso_contrastive.py --config configs/contrastive_a100.yaml \
        --data-dir /path/to/outputs --train-only
"""
from _bootstrap import base_parser, load


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load(args)
    from eegrag.experiments import run_loso_contrastive
    out = run_loso_contrastive(cfg, train_only=args.train_only)
    print("\n=== Summary (Mean +/- Std across folds x seeds) ===")
    for head, metrics in out["summary"].items():
        ap = metrics.get("auc_pr", {})
        se = metrics.get("sensitivity", {})
        print(f"[{head}] AUC-PR={ap.get('mean'):.4f}+/-{ap.get('std'):.4f} "
              f"Sens={se.get('mean'):.4f}+/-{se.get('std'):.4f}")


if __name__ == "__main__":
    main()
