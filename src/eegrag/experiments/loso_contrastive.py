"""LOSO contrastive + retrieval runner (the main experiment).

Per (fold, seed):
  1. prepare_fold -> leakage-safe transformed train/test arrays.
  2. train contrastive encoder on training subjects only.
  3. extract embeddings for train (bank) and test (queries).
  4. train a linear probe on training embeddings (for linear/hybrid heads).
  5. evaluate retrieval heads + neighbor study on the held-out subject.
  6. persist embeddings, predictions, neighbors, metrics, thresholds.

Aggregates per-head metrics as Mean +/- Std across folds and seeds, and dumps a
tidy long-format table for the stats stage.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import ExperimentConfig, dump_config
from ..data import assemble_dataset, make_loso_folds
from ..evaluation.metrics import METRIC_KEYS
from ..preprocessing import prepare_fold
from ..stats import summarize_folds
from ..utils.backend import detect_backend, resolve_torch_device
from ..utils.io import ensure_dir, git_revision, save_dataframe, save_json
from ..utils.logging import get_logger
from ..utils.profiling import ComputeLog, ResourceTracker
from ..utils.seed import seed_everything
from .retrieval_eval import evaluate_retrieval_for_fold
from ..utils.arrays import to_host, validate_resident_config
import hashlib

log = get_logger(__name__)


def run_loso_contrastive(cfg: ExperimentConfig, train_only: bool = False) -> Dict:
    """Run LOSO contrastive experiment.

    Args:
        train_only: If True, skip retrieval evaluation and only train encoders +
            extract embeddings.  Use this on the A100 to maximise use of Tensor
            Cores / cuML; then copy the embeddings NPZs to the MI300X and run
            ``run_retrieval_eval.py`` there.
    """
    from . import _engine  # lazy: only this runner needs torch
    validate_resident_config(cfg)

    if train_only and not cfg.training.save_embeddings:
        raise ValueError("--train-only requires training.save_embeddings=true")

    out_dir = ensure_dir(os.path.join(cfg.output_dir, cfg.name))
    for sub in ("metrics", "predictions", "embeddings", "retrieval", "stats"):
        ensure_dir(os.path.join(out_dir, sub))
    dump_config(cfg, os.path.join(out_dir, "config.resolved.yaml"))

    git = git_revision()
    device = resolve_torch_device(cfg.training.device)
    method = cfg.loss.name
    clog = ComputeLog(os.path.join(cfg.output_dir, "compute"))

    bundle = assemble_dataset(cfg.data)
    folds = make_loso_folds(bundle, folds_whitelist=cfg.folds_whitelist)
    log.info("Built %d LOSO folds (git=%s).", len(folds), git)

    rows: List[Dict] = []
    compute_stats: Optional[Dict] = None
    for fold in folds:
        fa = prepare_fold(bundle, fold, cfg.preprocessing)
        data_signature = None
        if bundle.source_signature:
            data_signature = hashlib.sha256(bundle.source_signature.encode()
                + fold.train_idx.tobytes() + fold.test_idx.tobytes()).hexdigest()
        for seed in cfg.seeds:
            seed_everything(seed, deterministic_torch=cfg.training.deterministic)
            log.info("=== Fold %d | test=%s | seed=%d ===",
                     fold.fold_index, fold.test_subject, seed)

            # --- encoder training (timed, peak RAM + VRAM) ----------------- #
            with ResourceTracker(track_vram=True) as rt_train:
                model = _engine.train_contrastive_encoder(
                    fa.x_train, fa.y_train, bundle.layout, cfg, seed=seed,
                    checkpoint_path=os.path.join(out_dir, "checkpoints",
                                                 f"{fold.test_subject}_seed{seed}.pt"),
                    data_signature=data_signature,
                )
            save_json(model.training_profile, os.path.join(
                out_dir, "metrics", f"{fold.test_subject}_seed{seed}_training.json"))
            # param count + FLOPs (constant across folds; compute once)
            if compute_stats is None:
                compute_stats = _engine.model_compute_stats(
                    model, fa.x_train, bundle.layout, cfg
                )
                log.info("Encoder: %s params, %.3g FLOPs/sample",
                         f"{compute_stats['n_params']:,}", compute_stats["flops"])
            clog.add_resource(
                stage="train_encoder", method=method, fold=fold.fold_index,
                test_subject=fold.test_subject, seed=seed, tracker=rt_train,
                device=device, n_params=compute_stats["n_params"],
                n_trainable=compute_stats["n_trainable"], flops=compute_stats["flops"],
                git=git,
            )

            # --- embedding generation (timed) ------------------------------ #
            with ResourceTracker(track_vram=True) as rt_emb_tr:
                train_emb = _engine.extract_embeddings(
                    model, fa.x_train, bundle.layout, cfg,
                    normalize=cfg.retrieval.normalize_embeddings,
                )
            with ResourceTracker(track_vram=True) as rt_emb_te:
                test_emb = _engine.extract_embeddings(
                    model, fa.x_test, bundle.layout, cfg,
                    normalize=cfg.retrieval.normalize_embeddings,
                )
            clog.add_resource(
                stage="embed", method=method, fold=fold.fold_index,
                test_subject=fold.test_subject, seed=seed, tracker=rt_emb_tr,
                device=device, git=git,
            )

            # save embeddings (so retrieval can be re-swept without retraining)
            if cfg.training.save_embeddings:
                np.savez_compressed(
                    os.path.join(out_dir, "embeddings", f"{fold.test_subject}_seed{seed}.npz"),
                    train_emb=to_host(train_emb), test_emb=to_host(test_emb),
                    train_labels=fa.y_train, test_labels=fa.y_test,
                    train_subjects=bundle.subjects[fold.train_idx],
                    test_subjects=bundle.subjects[fold.test_idx],
                )

            if train_only:
                log.info("--train-only: skipping retrieval eval for fold %s seed %d",
                         fold.test_subject, seed)
                continue

            linear_clf = _engine.train_linear_head(train_emb, fa.y_train, cfg, seed=seed)

            res = evaluate_retrieval_for_fold(
                bundle=bundle, fold=fold,
                train_emb=train_emb, test_emb=test_emb,
                cfg=cfg, seed=seed, linear_clf=linear_clf, save_dir=out_dir,
                embed_train_sec=rt_emb_tr.elapsed_sec,
                embed_test_sec=rt_emb_te.elapsed_sec,
            )
            clog.add_retrieval_cost(
                method=method, fold=fold.fold_index,
                test_subject=fold.test_subject, seed=seed, **res.cost,
            )

            for head, metrics in res.metrics.items():
                row = {"test_subject": fold.test_subject, "seed": seed,
                       "head": head, "model": f"contrastive_{cfg.loss.name}_{head}"}
                row.update({k: metrics.get(k) for k in METRIC_KEYS})
                rows.append(row)
            del model, train_emb, test_emb, linear_clf
        del fa

    # Total training time across folds (printed for the "per-fold and total"
    # requirement); per-fold rows already live in runtime.csv.
    train_total_sec = sum(
        r["time_sec"] for r in clog._buf["runtime"] if r.get("stage") == "train_encoder"
    )
    clog.flush()
    log.info("Total encoder training time (method=%s): %.1f s across %d fold-seeds.",
             method, train_total_sec, len(folds) * len(cfg.seeds))

    # --- aggregate -------------------------------------------------------- #
    if train_only:
        log.info("Contrastive LOSO (train-only) complete. Embeddings in %s; "
                 "run run_retrieval_eval.py to score them.", out_dir)
        return {"summary": {}, "n_rows": 0, "output_dir": out_dir,
                "train_total_sec": train_total_sec, "compute_stats": compute_stats}

    long_df = pd.DataFrame(rows)
    save_dataframe(long_df, os.path.join(out_dir, "stats", "per_fold_metrics.csv"))

    summary: Dict[str, Dict] = {}
    for head in sorted(long_df["head"].unique()):
        sub = long_df[long_df["head"] == head]
        per_fold = sub.to_dict(orient="records")
        summary[head] = summarize_folds(per_fold, keys=METRIC_KEYS)
    save_json(summary, os.path.join(out_dir, "stats", "summary_mean_std.json"))

    log.info("Contrastive LOSO complete. Summary written to %s", out_dir)
    return {"summary": summary, "n_rows": len(rows), "output_dir": out_dir,
            "train_total_sec": train_total_sec, "compute_stats": compute_stats}
