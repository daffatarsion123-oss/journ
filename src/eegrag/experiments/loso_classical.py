"""LOSO classical-baseline runner (LR / RBF-SVM / RandomForest / XGBoost).

Reproduces the conference-paper baselines under the SAME strict LOSO protocol
and the SAME imbalance-aware metrics, so the journal can compare classical vs
contrastive+retrieval fairly. Hyperparameters (esp. SVM C / gamma / class_weight)
are searched with subject-grouped CV on TRAINING SUBJECTS ONLY (see
``models.classical.fit_classical_with_search``).
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import ExperimentConfig, dump_config
from ..data import assemble_dataset, make_loso_folds
from ..evaluation import compute_metrics, tune_threshold
from ..evaluation.metrics import METRIC_KEYS
from ..models import fit_classical_with_search
from ..preprocessing import prepare_fold
from ..stats import summarize_folds
from ..utils.backend import detect_backend
from ..utils.io import ensure_dir, git_revision, save_dataframe, save_json, save_npz
from ..utils.logging import get_logger
from ..utils.profiling import ComputeLog, ResourceTracker
from ..utils.seed import seed_everything

log = get_logger(__name__)


def run_loso_classical(cfg: ExperimentConfig) -> Dict:
    out_dir = ensure_dir(os.path.join(cfg.output_dir, cfg.name))
    for sub in ("metrics", "predictions", "stats"):
        ensure_dir(os.path.join(out_dir, sub))
    dump_config(cfg, os.path.join(out_dir, "config.resolved.yaml"))

    info = detect_backend()
    log.info("Backend: %s", info.summary())
    git = git_revision()
    device = "cuda" if (info.is_cuda or info.is_rocm) else "cpu"
    clog = ComputeLog(os.path.join(cfg.output_dir, "compute"))

    bundle = assemble_dataset(cfg.data)
    folds = make_loso_folds(bundle, folds_whitelist=cfg.folds_whitelist)
    log.info("Built %d LOSO folds (git=%s).", len(folds), git)

    rows: List[Dict] = []
    for fold in folds:
        for seed in cfg.seeds:
            seed_everything(seed)
            fa = prepare_fold(bundle, fold, cfg.preprocessing)
            total_hours = (len(fa.y_test) * cfg.data.step_size_sec) / 3600.0

            for model_name in cfg.classical.models:
                log.info("Fold test=%s seed=%d | model=%s",
                         fold.test_subject, seed, model_name)
                try:
                    with ResourceTracker(track_vram=True) as rt_fit:
                        model, search = fit_classical_with_search(
                            model_name, fa.x_train, fa.y_train, fa.subjects_train,
                            cfg.classical, info=info, seed=seed,
                        )
                except Exception as exc:  # keep the sweep alive on one bad model
                    log.exception("Model %s failed on fold %s: %s",
                                  model_name, fold.test_subject, exc)
                    continue

                clog.add_resource(
                    stage="classical_fit", method=model_name, fold=fold.fold_index,
                    test_subject=fold.test_subject, seed=seed, tracker=rt_fit,
                    device=f"{device}:{model.backend}", git=git,
                )

                # threshold tuned on TRAIN scores (resubstitution is train-only)
                train_scores = model.predict_proba(fa.x_train)
                thr = tune_threshold(fa.y_train, train_scores, criterion="f1")
                test_scores = model.predict_proba(fa.x_test)
                metrics = compute_metrics(
                    fa.y_test, test_scores, threshold=thr,
                    step_size_sec=cfg.data.step_size_sec, total_hours=total_hours,
                )

                tag = f"{model_name}_{fold.test_subject}_seed{seed}"
                save_npz(
                    os.path.join(out_dir, "predictions", f"{tag}.npz"),
                    y_true=fa.y_test, score=test_scores,
                    test_subjects=bundle.subjects[fold.test_idx],
                )
                save_json(
                    {"model": model_name, "test_subject": fold.test_subject,
                     "seed": seed, "metrics": metrics, "threshold": thr,
                     "search": search},
                    os.path.join(out_dir, "metrics", f"{tag}.json"),
                )

                row = {"test_subject": fold.test_subject, "seed": seed,
                       "model": model_name, "backend": model.backend, "threshold": thr}
                row.update({k: metrics.get(k) for k in METRIC_KEYS})
                rows.append(row)

    # per-model total fit time (printed for the "per-fold and total" requirement)
    fit_totals = clog.totals_by("runtime", group="method", value="time_sec")
    clog.flush()
    for m, secs in sorted(fit_totals.items()):
        log.info("Total fit time | model=%s: %.1f s", m, secs)

    long_df = pd.DataFrame(rows)
    save_dataframe(long_df, os.path.join(out_dir, "stats", "per_fold_metrics.csv"))

    summary: Dict[str, Dict] = {}
    for model_name in sorted(long_df["model"].unique()) if len(long_df) else []:
        sub = long_df[long_df["model"] == model_name].to_dict(orient="records")
        summary[model_name] = summarize_folds(sub, keys=METRIC_KEYS)
    save_json(summary, os.path.join(out_dir, "stats", "summary_mean_std.json"))

    log.info("Classical LOSO complete. Output: %s", out_dir)
    return {"summary": summary, "n_rows": len(rows), "output_dir": out_dir,
            "fit_totals_sec": fit_totals}
