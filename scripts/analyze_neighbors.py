#!/usr/bin/env python
"""Aggregate the retrieval neighbor study (RQ2) across folds.

Reads the per-fold metric JSONs written by the contrastive / retrieval runners
(``outputs/<name>/metrics/*.json``), then aggregates:
  * seizure-neighbor ratio vs k,
  * same-subject vs cross-subject neighbor fractions,
  * subject diversity among neighbors,
  * neighbor overlap between cosine / euclidean / rbf,
  * seizure-prediction support (are positive predictions backed by seizure
    neighbors?).

Writes a tidy CSV + a JSON summary to ``outputs/<name>/stats/``.

Example:
    python scripts/analyze_neighbors.py --metrics-dir outputs/contrastive_supcon/metrics
"""
import argparse
import glob
import json
import os

from _bootstrap import _ensure_src_on_path  # noqa: F401  (side effect: sys.path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-dir", required=True,
                   help="Directory of per-fold metric JSONs.")
    p.add_argument("--out-dir", default=None,
                   help="Where to write aggregates (default: sibling 'stats').")
    args = p.parse_args()

    import numpy as np
    import pandas as pd

    files = sorted(glob.glob(os.path.join(args.metrics_dir, "*.json")))
    if not files:
        raise SystemExit(f"No JSON metric files in {args.metrics_dir}")

    rows = []
    overlap_rows = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        subj, seed = d.get("test_subject"), d.get("seed")
        for key, stats in (d.get("neighbor_stats") or {}).items():
            metric, _, k = key.partition("@")
            rows.append({"test_subject": subj, "seed": seed, "metric": metric,
                         "k": int(k), **stats})
        for pair, jac in (d.get("overlaps") or {}).items():
            overlap_rows.append({"test_subject": subj, "seed": seed,
                                 "pair": pair, "jaccard": jac})

    df = pd.DataFrame(rows)
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.metrics_dir), "stats")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "neighbor_analysis_long.csv"), index=False)

    # aggregate by metric x k
    agg_keys = [
        "mean_seizure_neighbor_ratio", "mean_same_subject_fraction",
        "mean_cross_subject_fraction", "mean_subject_diversity",
        "seizure_pred_seizure_support", "nonseizure_pred_seizure_support",
    ]
    grouped = (
        df.groupby(["metric", "k"])[agg_keys]
        .agg(["mean", "std"]).reset_index()
    )
    grouped.columns = ["_".join(c).rstrip("_") for c in grouped.columns]
    grouped.to_csv(os.path.join(out_dir, "neighbor_analysis_by_metric_k.csv"),
                   index=False)

    summary = {"by_metric_k": grouped.to_dict(orient="records")}
    if overlap_rows:
        odf = pd.DataFrame(overlap_rows)
        ov = odf.groupby("pair")["jaccard"].agg(["mean", "std", "count"]).reset_index()
        ov.to_csv(os.path.join(out_dir, "neighbor_overlap.csv"), index=False)
        summary["overlap"] = ov.to_dict(orient="records")

    with open(os.path.join(out_dir, "neighbor_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Neighbor analysis written to {out_dir}")
    print("\nCross-subject fraction (mean) by metric@k:")
    for r in summary["by_metric_k"]:
        print(f"  {r['metric']}@{r['k']}: "
              f"cross={r.get('mean_cross_subject_fraction_mean'):.3f} "
              f"seizure_ratio={r.get('mean_seizure_neighbor_ratio_mean'):.3f}")


if __name__ == "__main__":
    main()
