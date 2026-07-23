#!/usr/bin/env python
"""Run the significance-testing protocol over per-fold metric tables.

Inputs are one or more ``per_fold_metrics.csv`` files (written by the runners),
each with columns including a model id and the metric of interest. Models are
matched across the SAME folds (and seeds) so the tests are paired.

Protocol:
  * >2 models  -> Friedman omnibus, then Nemenyi post-hoc (avg ranks + CD).
  * exactly 2  -> Wilcoxon signed-rank + effect size r + 95% bootstrap CI.

Example:
    python scripts/analyze_statistics.py \
        --csv outputs/classical/stats/per_fold_metrics.csv \
        --csv outputs/contrastive_supcon/stats/per_fold_metrics.csv \
        --metric auc_pr --model-col model \
        --out outputs/stats/auc_pr_significance.json
"""
import argparse
import json
import os

from _bootstrap import _ensure_src_on_path  # noqa: F401


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", action="append", required=True,
                   help="per_fold_metrics.csv (repeatable to combine experiments).")
    p.add_argument("--metric", default="auc_pr", help="Metric column to test.")
    p.add_argument("--model-col", default="model", help="Column identifying models.")
    p.add_argument("--block-cols", default="test_subject,seed",
                   help="Comma-separated columns identifying matched blocks.")
    p.add_argument("--alternative", default="two-sided")
    p.add_argument("--out", default="outputs/stats/significance.json")
    args = p.parse_args()

    import numpy as np
    import pandas as pd
    from eegrag.stats import (
        friedman_test, nemenyi_posthoc, paired_comparison,
    )

    frames = [pd.read_csv(c) for c in args.csv]
    df = pd.concat(frames, ignore_index=True)
    block_cols = [c.strip() for c in args.block_cols.split(",") if c.strip()]
    for col in block_cols + [args.model_col, args.metric]:
        if col not in df.columns:
            raise SystemExit(f"Column {col!r} not found in inputs. Have: "
                             f"{list(df.columns)}")

    # pivot to blocks x models on the chosen metric (mean if duplicate rows)
    pivot = df.pivot_table(index=block_cols, columns=args.model_col,
                           values=args.metric, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")   # keep only fully-matched blocks
    if pivot.shape[0] == 0:
        raise SystemExit("No fully-matched blocks across models; cannot pair.")
    models = list(pivot.columns)
    per_model = {m: pivot[m].to_numpy() for m in models}

    report = {"metric": args.metric, "n_blocks": int(pivot.shape[0]),
              "models": models}

    if len(models) >= 3:
        fr = friedman_test(per_model)
        report["friedman"] = fr.to_dict()
        if fr.p_value < 0.05:
            report["nemenyi"] = nemenyi_posthoc(per_model).to_dict()
            report["note"] = "Friedman significant -> Nemenyi reported."
        else:
            report["note"] = ("Friedman not significant (p>=0.05); Nemenyi "
                              "post-hoc omitted.")
    elif len(models) == 2:
        a, b = models
        report["pairwise"] = paired_comparison(
            a, per_model[a], b, per_model[b], alternative=args.alternative
        )
    else:
        report["note"] = "Only one model present; nothing to compare."

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Significance report -> {args.out}")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("metric", "n_blocks", "models", "note")}, indent=2))


if __name__ == "__main__":
    main()
