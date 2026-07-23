#!/usr/bin/env python
"""End-to-end smoke test on a tiny synthetic CHB-MIT-shaped dataset (CPU).

Fabricates a handful of subjects' feature parquets with the exact column schema
the real extractor produces (channel x feature-type names, windowStartSec /
windowEndSec / label), then exercises:

    assemble -> LOSO split -> leakage assertions -> preprocessing ->
    contrastive encoder (1 epoch) -> embeddings -> memory bank ->
    retrieval heads -> metrics -> neighbor study -> stats.

Run directly (``python tests/test_smoke.py``) or via pytest. No GPU, no real
data, no network. Designed to fail loudly if any leakage guard or shape contract
breaks.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

# make the package importable without installation
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3",
    "C3-P3", "P3-O1", "FP2-F4", "F4-C4",
]
FEATS = ["Mean", "Std", "Variance", "Entropy",
         "DeltaPower", "ThetaPower", "AlphaPower", "BetaPower"]


def _make_recording(rng, n_windows, pos_rate, start_offset=0.0):
    cols = {
        "windowStartSec": start_offset + np.arange(n_windows, dtype=float),
        "windowEndSec": start_offset + np.arange(n_windows, dtype=float) + 2.0,
    }
    labels = (rng.random(n_windows) < pos_rate).astype(int)
    cols["label"] = labels
    for ch in CHANNELS:
        for ft in FEATS:
            base = rng.normal(0, 1, n_windows)
            # make seizure windows separable so the pipeline has signal
            base = base + labels * rng.normal(1.5, 0.3)
            cols[f"{ch}{ft}"] = base.astype(np.float32)
    return pd.DataFrame(cols)


def build_synthetic_dataset(tmp, n_subjects=4, recs=2, n_windows=300, seed=0):
    rng = np.random.default_rng(seed)
    for s in range(1, n_subjects + 1):
        for r in range(1, recs + 1):
            # vary seizure prevalence per subject so cross-subject matters
            pos = 0.08 + 0.04 * (s % 3)
            df = _make_recording(rng, n_windows, pos, start_offset=(r - 1) * n_windows)
            df.to_parquet(os.path.join(tmp, f"chb{s:02d}_{r:02d}_features.parquet"),
                          index=False)
    return tmp


def run_smoke():
    from eegrag.config import load_config
    from eegrag.data import assemble_dataset, make_loso_folds, assert_no_subject_leakage
    from eegrag.preprocessing import prepare_fold
    from eegrag.experiments import _engine
    from eegrag.experiments.retrieval_eval import evaluate_retrieval_for_fold
    from eegrag.stats import friedman_test, paired_comparison
    from eegrag.quantum import build_quantum_kernel, quantum_retrieval

    tmp = tempfile.mkdtemp(prefix="eegrag_smoke_")
    build_synthetic_dataset(tmp)
    out = tempfile.mkdtemp(prefix="eegrag_out_")

    cfg = load_config(overrides=[
        f"data.features_dir={tmp}",
        "name=smoke",
        f"output_dir={out}",
        "seeds=[0]",
        "encoder.arch=cnn1d",
        "encoder.embedding_dim=32",
        "encoder.projection_dim=16",
        "training.epochs=1",
        "training.batch_size=64",
        "training.num_workers=0",
        "training.device=cpu",
        "training.amp=false",
        "retrieval.decision_k=5",
        "retrieval.topk_values=[1,5]",
        "quantum.enabled=true",
        "quantum.projection_dim=4",
        "quantum.n_qubits=4",
        "quantum.max_bank_per_class=32",
        "quantum.max_query=16",
    ])

    bundle = assemble_dataset(cfg.data)
    assert bundle.layout.n_channels == len(CHANNELS), bundle.layout.describe()
    assert bundle.layout.n_feat_per_channel == len(FEATS)
    assert bundle.layout.is_regular

    folds = make_loso_folds(bundle)
    assert len(folds) == 4
    fold = folds[0]
    assert_no_subject_leakage(bundle, fold, context="smoke")

    fa = prepare_fold(bundle, fold, cfg.preprocessing)
    assert fa.x_train.shape[1] == fa.x_test.shape[1]
    # leakage: the test subject must not appear in the training rows
    train_subs = set(bundle.subjects[fold.train_idx].tolist())
    assert fold.test_subject not in train_subs

    model = _engine.train_contrastive_encoder(fa.x_train, fa.y_train, bundle.layout,
                                               cfg, seed=0)
    train_emb = _engine.extract_embeddings(model, fa.x_train, bundle.layout, cfg)
    test_emb = _engine.extract_embeddings(model, fa.x_test, bundle.layout, cfg)
    assert train_emb.shape[0] == len(fa.y_train)
    assert test_emb.shape[0] == len(fa.y_test)

    linear_clf = _engine.train_linear_head(train_emb, fa.y_train, cfg)
    res = evaluate_retrieval_for_fold(
        bundle=bundle, fold=fold, train_emb=train_emb, test_emb=test_emb,
        cfg=cfg, seed=0, linear_clf=linear_clf, save_dir=os.path.join(out, "smoke"),
    )
    for head in ("knn", "linear", "hybrid"):
        assert head in res.metrics, f"missing head {head}"
        m = res.metrics[head]
        assert "auc_pr" in m and "sensitivity" in m and "fp_per_hour" in m
    # cross-subject fraction should be ~1.0 under LOSO (no same-subject neighbors)
    stat = res.neighbor_stats["cosine@5"]
    assert stat["mean_cross_subject_fraction"] >= 0.99, stat

    # quantum (RBF fallback here) retrieval over a compressed bank
    kernel = build_quantum_kernel(cfg.quantum)
    from eegrag.retrieval import build_memory_bank
    bank = build_memory_bank(
        train_emb[:, :4], bundle.labels[fold.train_idx],
        bundle.subjects[fold.train_idx], global_idx=fold.train_idx,
        test_subject=fold.test_subject, max_per_class=32, seed=0,
    )
    nidx, sim, nlab, nsub = quantum_retrieval(
        kernel, test_emb[:16, :4], bank.embeddings, bank.labels, bank.subjects, 5
    )
    assert nidx.shape == (16, 5)

    # stats: fabricate matched per-fold metric vectors for 3 models
    rng = np.random.default_rng(1)
    per_model = {
        "logreg": rng.normal(0.3, 0.05, 8),
        "supcon": rng.normal(0.45, 0.05, 8),
        "imbalance_supcon": rng.normal(0.55, 0.05, 8),
    }
    fr = friedman_test(per_model)
    assert fr.n_models == 3 and np.isfinite(fr.p_value)
    pc = paired_comparison("a", per_model["supcon"], "b", per_model["imbalance_supcon"])
    assert "wilcoxon" in pc and "ci_low" in pc and "ci_high" in pc

    print("\nSMOKE TEST PASSED")
    print(f"  knn  AUC-PR={res.metrics['knn']['auc_pr']:.3f} "
          f"sens={res.metrics['knn']['sensitivity']:.3f} "
          f"fp/h={res.metrics['knn']['fp_per_hour']:.1f}")
    print(f"  cross-subject neighbor fraction (cosine@5)="
          f"{stat['mean_cross_subject_fraction']:.3f}")
    print(f"  quantum kernel backend={kernel.backend} (is_quantum={kernel.is_quantum})")
    print(f"  Friedman p={fr.p_value:.4f}  avg_ranks={fr.avg_ranks}")


def test_smoke():
    run_smoke()


if __name__ == "__main__":
    run_smoke()
