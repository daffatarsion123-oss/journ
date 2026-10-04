"""Retrieval evaluation: heads x similarities x k, with the neighbor study.

The core function :func:`evaluate_retrieval_for_fold` is reused by the
contrastive runner (inline) and by the standalone re-analysis script
(``run_retrieval_eval`` over saved embeddings). Everything routes through the
leakage-guarded :class:`MemoryBank`, and threshold tuning uses TRAIN-ONLY scores
(leave-self-out for the k-NN head; the linear probe's own train predictions for
the linear head).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

from ..config import ExperimentConfig
from ..data.loading import EEGFeatureBundle
from ..data.loso import LOSOFold, assert_no_subject_leakage
from ..evaluation import compute_metrics, tune_threshold
from ..retrieval import (
    build_memory_bank,
    knn_seizure_score,
    hybrid_score,
    neighbor_analysis,
    neighbor_overlap,
)
from ..utils.io import save_json, save_npz, ensure_dir
from ..utils.logging import get_logger
from ..retrieval.similarity import similarity_from_neighbors

log = get_logger(__name__)


@dataclass
class FoldRetrievalResult:
    test_subject: str
    seed: int
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)  # head -> metrics
    neighbor_stats: Dict[str, Dict] = field(default_factory=dict)       # "metric@k" -> stats
    overlaps: Dict[str, float] = field(default_factory=dict)            # pair -> jaccard
    thresholds: Dict[str, float] = field(default_factory=dict)
    cost: Dict[str, float] = field(default_factory=dict)               # retrieval-cost record


def _slice_result(result, k, metric=None, gamma=1.0):
    k = min(k, result.neighbor_idx.shape[1])
    metric = metric or result.metric
    raw = result.raw[:, :k]
    return replace(result, neighbor_idx=result.neighbor_idx[:, :k], raw=raw,
                   similarity=similarity_from_neighbors(raw, metric, gamma),
                   neighbor_labels=result.neighbor_labels[:, :k],
                   neighbor_subjects=result.neighbor_subjects[:, :k], metric=metric, k=k)


def _knn_scores_leave_self_out(bank, queries, k, metric, gamma, backend="auto"):
    """k-NN seizure score for queries that ARE in the bank (drop the self hit)."""
    if bank.size < 2:
        raise ValueError("Leave-self-out calibration requires at least two bank entries")
    k = min(k, bank.size - 1)
    res = bank.retrieve(queries, k + 1, metric=metric, gamma=gamma, backend=backend)
    keep = res.neighbor_idx != np.arange(len(queries))[:, None]
    positions = np.argsort(~keep, axis=1, kind="stable")[:, :k]
    for name in ("neighbor_idx", "similarity", "raw", "neighbor_labels", "neighbor_subjects"):
        setattr(res, name, np.take_along_axis(getattr(res, name), positions, axis=1))
    res.k = k
    return knn_seizure_score(res, weighted=True)


def evaluate_retrieval_for_fold(
    *,
    bundle: EEGFeatureBundle,
    fold: LOSOFold,
    train_emb: np.ndarray,
    test_emb: np.ndarray,
    cfg: ExperimentConfig,
    seed: int,
    linear_clf=None,
    save_dir: Optional[str] = None,
    embed_train_sec: float = float("nan"),
    embed_test_sec: float = float("nan"),
) -> FoldRetrievalResult:
    """Score linear / knn / hybrid heads on the held-out subject + neighbor study.

    ``embed_train_sec`` / ``embed_test_sec`` are the encoder embedding-generation
    times (passed by the contrastive runner; NaN for the standalone re-analysis
    that loads embeddings from disk). They are recorded into ``result.cost``.
    """
    rcfg = cfg.retrieval
    train_labels = bundle.labels[fold.train_idx].astype(int)
    train_subjects = bundle.subjects[fold.train_idx]
    test_labels = bundle.labels[fold.test_idx].astype(int)
    test_subjects = bundle.subjects[fold.test_idx]

    # --- build memory bank (TRAIN ONLY, test subject guarded) ------------- #
    bank = build_memory_bank(
        train_emb, train_labels, train_subjects,
        global_idx=fold.train_idx, test_subject=fold.test_subject,
        max_per_class=rcfg.max_bank_per_class, seed=rcfg.seed,
        query_chunk_size=rcfg.query_chunk_size, bank_chunk_size=rcfg.bank_chunk_size,
    )
    assert_no_subject_leakage(
        bundle, fold, bank_idx=bank.global_idx, context="retrieval-bank"
    )

    result = FoldRetrievalResult(test_subject=fold.test_subject, seed=seed)
    total_hours = (len(test_labels) * cfg.data.step_size_sec) / 3600.0

    # --- decision-k retrieval for the test subject ------------------------ #
    # Capture index-build + query cost for the decision path only (the neighbor
    # sweep below is analysis overhead, not counted as retrieval cost).
    largest = max(rcfg.topk_values)
    decision_neighbors = bank.retrieve(
        test_emb, largest, metric=rcfg.similarity, gamma=rcfg.rbf_gamma,
        backend=rcfg.backend,
    )
    test_res = _slice_result(decision_neighbors, rcfg.decision_k, gamma=rcfg.rbf_gamma)
    result.cost = {
        "embed_train_sec": round(embed_train_sec, 4),
        "embed_test_sec": round(embed_test_sec, 4),
        "index_build_sec": round(bank.index_build_sec, 6),
        "query_sec": round(bank.query_sec, 6),
        "bank_size": bank.size,
        "n_query": int(len(test_emb)),
        "backend": rcfg.backend,
    }
    knn_test = knn_seizure_score(test_res, weighted=True)

    # All TRAIN-side scoring is computed on the BANK rows so the vectors align
    # with ``bank.labels`` even when the majority class was subsampled. The bank
    # is a strict subset of the training partition, so this stays train-only.
    bank_labels = bank.labels

    # linear-head probabilities (if a probe was trained)
    if linear_clf is not None:
        lin_bank = linear_clf.predict_proba(bank.embeddings)[:, 1]
        lin_test = linear_clf.predict_proba(test_emb)[:, 1]
    else:
        lin_bank = lin_test = None

    # --- thresholds tuned on TRAIN ONLY ----------------------------------- #
    # knn head: leave-self-out k-NN scores on the bank itself
    knn_train = _knn_scores_leave_self_out(
        bank, bank.embeddings, rcfg.decision_k, rcfg.similarity, rcfg.rbf_gamma, rcfg.backend
    )
    t_knn = tune_threshold(bank_labels, knn_train, criterion="f1")
    result.thresholds["knn"] = t_knn
    result.metrics["knn"] = compute_metrics(
        test_labels, knn_test, threshold=t_knn,
        step_size_sec=cfg.data.step_size_sec, total_hours=total_hours,
    )

    if lin_test is not None:
        t_lin = tune_threshold(bank_labels, lin_bank, criterion="f1")
        result.thresholds["linear"] = t_lin
        result.metrics["linear"] = compute_metrics(
            test_labels, lin_test, threshold=t_lin,
            step_size_sec=cfg.data.step_size_sec, total_hours=total_hours,
        )
        # hybrid
        hyb_train = hybrid_score(lin_bank, knn_train, rcfg.hybrid_alpha)
        hyb_test = hybrid_score(lin_test, knn_test, rcfg.hybrid_alpha)
        t_hyb = tune_threshold(bank_labels, hyb_train, criterion="f1")
        result.thresholds["hybrid"] = t_hyb
        result.metrics["hybrid"] = compute_metrics(
            test_labels, hyb_test, threshold=t_hyb,
            step_size_sec=cfg.data.step_size_sec, total_hours=total_hours,
        )

    # --- neighbor study: sweep similarities x k --------------------------- #
    preds_for_support = (knn_test >= t_knn).astype(int)
    per_metric_results = {}
    cache = {"ip" if rcfg.similarity == "cosine" else "l2": decision_neighbors}
    for metric in ("cosine", "euclidean", "rbf"):
        key = "ip" if metric == "cosine" else "l2"
        if key not in cache:
            cache[key] = bank.retrieve(test_emb, largest, metric=metric,
                                      gamma=rcfg.rbf_gamma, backend=rcfg.backend)
        for k in rcfg.topk_values:
            res_k = _slice_result(cache[key], k, metric=metric, gamma=rcfg.rbf_gamma)
            stats = neighbor_analysis(res_k, test_subjects, predictions=preds_for_support)
            result.neighbor_stats[f"{metric}@{k}"] = stats.to_dict()
            if k == rcfg.decision_k:
                per_metric_results[metric] = res_k

    # neighbor overlap between metrics at decision_k
    metrics_avail = list(per_metric_results)
    for i in range(len(metrics_avail)):
        for j in range(i + 1, len(metrics_avail)):
            a, b = metrics_avail[i], metrics_avail[j]
            result.overlaps[f"{a}_vs_{b}"] = neighbor_overlap(
                per_metric_results[a], per_metric_results[b]
            )

    # --- persist per-fold artifacts --------------------------------------- #
    if save_dir:
        ensure_dir(save_dir)
        tag = f"{fold.test_subject}_seed{seed}"
        save_npz(
            os.path.join(save_dir, "predictions", f"{tag}.npz"),
            y_true=test_labels,
            knn_score=knn_test,
            linear_prob=lin_test if lin_test is not None else np.array([]),
            test_subjects=test_subjects,
            window_start=bundle.window_start[fold.test_idx],
            window_end=bundle.window_end[fold.test_idx],
        )
        save_npz(
            os.path.join(save_dir, "retrieval", f"{tag}_neighbors.npz"),
            neighbor_idx=test_res.neighbor_idx,
            neighbor_labels=test_res.neighbor_labels,
            neighbor_subjects=test_res.neighbor_subjects,
            similarity=test_res.similarity,
        )
        save_json(
            {
                "test_subject": fold.test_subject,
                "seed": seed,
                "metrics": result.metrics,
                "thresholds": result.thresholds,
                "neighbor_stats": result.neighbor_stats,
                "overlaps": result.overlaps,
                "bank_size": bank.size,
                "bank_n_pos": bank.n_pos,
                "bank_n_subjects": bank.n_subjects,
                "bank_majority_cap": rcfg.max_bank_per_class,
                "calibration": "training_bank_leave_self_out",
            },
            os.path.join(save_dir, "metrics", f"{tag}.json"),
        )
    return result


def run_retrieval_eval(cfg: ExperimentConfig, embeddings_dir: str) -> Dict:
    """Re-run retrieval evaluation over previously-saved fold embeddings.

    Expects ``embeddings_dir`` to contain per-fold ``<subject>_seed<seed>.npz``
    files with ``train_emb, test_emb`` plus the bundle (rebuilt from configs).
    This lets you sweep heads / k / similarity without retraining encoders.
    """
    from ..data import assemble_dataset, make_loso_folds
    from ..utils.io import load_npz
    from ..utils.profiling import ComputeLog

    bundle = assemble_dataset(cfg.data)
    folds = make_loso_folds(bundle, folds_whitelist=cfg.folds_whitelist)
    out_dir = os.path.join(cfg.output_dir, cfg.name)
    clog = ComputeLog(os.path.join(cfg.output_dir, "compute"))
    all_results: List[FoldRetrievalResult] = []

    for fold in folds:
        for seed in cfg.seeds:
            tag = f"{fold.test_subject}_seed{seed}"
            emb_path = os.path.join(embeddings_dir, f"{tag}.npz")
            if not os.path.exists(emb_path):
                log.warning("Missing embeddings for %s; skipping.", tag)
                continue
            z = load_npz(emb_path)
            from ._engine import train_linear_head
            linear_clf = train_linear_head(z["train_emb"], bundle.labels[fold.train_idx], cfg, seed)
            res = evaluate_retrieval_for_fold(
                bundle=bundle, fold=fold,
                train_emb=z["train_emb"], test_emb=z["test_emb"],
                cfg=cfg, seed=seed, linear_clf=linear_clf, save_dir=out_dir,
            )
            # embeddings loaded from disk -> embed times are NaN (recorded as such)
            clog.add_retrieval_cost(
                method=cfg.name, fold=fold.fold_index,
                test_subject=fold.test_subject, seed=seed, **res.cost,
            )
            all_results.append(res)
    clog.flush()
    log.info("Retrieval eval complete: %d fold-seed results.", len(all_results))
    return {"n_results": len(all_results)}
