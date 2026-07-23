#!/usr/bin/env python
"""Run the LOSO classical baselines (LR / RBF-SVM / RandomForest / XGBoost).

Example:
    python scripts/run_loso_classical.py --config configs/classical.yaml \
        --data-dir /path/to/outputs
"""
from _bootstrap import base_parser, load


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load(args)
    from eegrag.experiments import run_loso_classical
    out = run_loso_classical(cfg)
    print("\n=== Classical baselines (Mean +/- Std) ===")
    for model, metrics in out["summary"].items():
        ap = metrics.get("auc_pr", {})
        se = metrics.get("sensitivity", {})
        f1 = metrics.get("f1_seizure", {})
        print(f"[{model}] AUC-PR={ap.get('mean'):.4f}+/-{ap.get('std'):.4f} "
              f"Sens={se.get('mean'):.4f}+/-{se.get('std'):.4f} "
              f"F1_seizure={f1.get('mean'):.4f}+/-{f1.get('std'):.4f}")


if __name__ == "__main__":
    main()
