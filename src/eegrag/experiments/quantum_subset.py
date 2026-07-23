"""Exploratory quantum-kernel retrieval on a compressed memory bank.

Runs simulator-only and on a *capped* bank/query set (see QuantumConfig), so the
fidelity-kernel cost stays bounded. For each fold it:
  1. loads (or rebuilds) embeddings,
  2. PCA/projects them to ``quantum.projection_dim`` (fit on TRAIN ONLY),
  3. builds a compressed, test-subject-free bank,
  4. retrieves neighbors with the quantum (or RBF-fallback) kernel,
  5. compares quantum vs cosine vs RBF neighbor overlap and seizure support.

If no quantum backend is installed, it degrades to a classical RBF kernel and
labels the output accordingly (so nothing in the journal is mislabeled).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from ..config import ExperimentConfig, dump_config
from ..data import assemble_dataset, make_loso_folds
from ..data.loso import assert_no_subject_leakage
from ..quantum import build_quantum_kernel, quantum_retrieval
from ..retrieval import build_memory_bank, knn_seizure_score, neighbor_overlap
from ..retrieval.heads import neighbor_analysis
from ..retrieval.memory_bank import RetrievalResult
from ..utils.io import ensure_dir, load_npz, save_json
from ..utils.logging import get_logger
from ..utils.profiling import ComputeLog, ResourceTracker
from ..utils.seed import seed_everything

log = get_logger(__name__)


def _fit_projection(train_emb: np.ndarray, dim: int, seed: int):
    from sklearn.decomposition import PCA

    dim = min(dim, train_emb.shape[1])
    pca = PCA(n_components=dim, random_state=seed)
    pca.fit(train_emb)
    return pca


def _result_from_arrays(nidx, sim, labels, subs, metric, k) -> RetrievalResult:
    return RetrievalResult(
        neighbor_idx=nidx, similarity=sim, raw=sim,
        neighbor_labels=labels, neighbor_subjects=subs, metric=metric, k=k,
    )


def run_quantum_retrieval_subset(
    cfg: ExperimentConfig, embeddings_dir: Optional[str] = None
) -> Dict:
    out_dir = ensure_dir(os.path.join(cfg.output_dir, cfg.name))
    ensure_dir(os.path.join(out_dir, "retrieval"))
    dump_config(cfg, os.path.join(out_dir, "config.resolved.yaml"))

    kernel = build_quantum_kernel(cfg.quantum)
    log.info("Quantum kernel backend=%s (is_quantum=%s)",
             kernel.backend, kernel.is_quantum)

    clog = ComputeLog(os.path.join(cfg.output_dir, "compute"))

    bundle = assemble_dataset(cfg.data)
    folds = make_loso_folds(bundle, folds_whitelist=cfg.folds_whitelist)
    qcfg = cfg.quantum

    results: List[Dict] = []
    for fold in folds:
        for seed in cfg.seeds:
            seed_everything(seed)
            tag = f"{fold.test_subject}_seed{seed}"
            emb_path = (
                os.path.join(embeddings_dir, f"{tag}.npz") if embeddings_dir else None
            )
            if not emb_path or not os.path.exists(emb_path):
                log.warning("No embeddings for %s; run the contrastive runner "
                            "first or pass --embeddings_dir. Skipping.", tag)
                continue

            z = load_npz(emb_path)
            train_emb, test_emb = z["train_emb"], z["test_emb"]
            train_labels = bundle.labels[fold.train_idx].astype(int)
            train_subjects = bundle.subjects[fold.train_idx]
            test_labels = bundle.labels[fold.test_idx].astype(int)
            test_subjects = bundle.subjects[fold.test_idx]

            # project to a few dims (fit on TRAIN ONLY)
            pca = _fit_projection(train_emb, qcfg.projection_dim, seed)
            train_proj = pca.transform(train_emb)
            test_proj = pca.transform(test_emb)

            # compressed, test-subject-free bank
            bank = build_memory_bank(
                train_proj, train_labels, train_subjects,
                global_idx=fold.train_idx, test_subject=fold.test_subject,
                max_per_class=qcfg.max_bank_per_class, seed=seed,
            )
            assert_no_subject_leakage(
                bundle, fold, bank_idx=bank.global_idx, context="quantum-bank"
            )

            # cap queries for the expensive kernel
            q = min(qcfg.max_query, len(test_proj))
            rng = np.random.default_rng(seed)
            q_sel = np.sort(rng.choice(len(test_proj), size=q, replace=False)) \
                if len(test_proj) > q else np.arange(len(test_proj))
            qy = test_labels[q_sel]
            qsubs = test_subjects[q_sel]

            k = cfg.retrieval.decision_k
            # quantum (or fallback) retrieval -- the O(N_bank * N_query) kernel
            # is the cost we care about; time it (+ peak RAM).
            with ResourceTracker(track_vram=False) as rt_q:
                nidx, sim, nlab, nsub = quantum_retrieval(
                    kernel, test_proj[q_sel], bank.embeddings, bank.labels,
                    bank.subjects, k,
                )
            clog.add_quantum_cost(
                kernel_backend=kernel.backend, is_quantum=kernel.is_quantum,
                method=f"quantum_{kernel.backend}", fold=fold.fold_index,
                test_subject=fold.test_subject, seed=seed,
                n_qubits=qcfg.n_qubits, projection_dim=qcfg.projection_dim,
                bank_size=int(bank.size), n_query=int(q),
                kernel_sec=round(rt_q.elapsed_sec, 6),
                total_sec=round(rt_q.elapsed_sec, 6),
                peak_ram_gb=round(rt_q.peak_ram_gb, 4),
            )
            q_res = _result_from_arrays(nidx, sim, nlab, nsub,
                                        "quantum" if kernel.is_quantum else "rbf_fallback", k)
            q_score = knn_seizure_score(q_res, weighted=True)
            q_stats = neighbor_analysis(q_res, qsubs,
                                        predictions=(q_score >= 0.5).astype(int))

            # classical cosine + rbf on the SAME projected bank/queries
            cos_res = bank.retrieve(test_proj[q_sel], k, metric="cosine")
            rbf_res = bank.retrieve(test_proj[q_sel], k, metric="rbf",
                                    gamma=cfg.retrieval.rbf_gamma)

            overlaps = {
                "quantum_vs_cosine": neighbor_overlap(q_res, cos_res),
                "quantum_vs_rbf": neighbor_overlap(q_res, rbf_res),
                "cosine_vs_rbf": neighbor_overlap(cos_res, rbf_res),
            }

            entry = {
                "test_subject": fold.test_subject, "seed": seed,
                "kernel_backend": kernel.backend, "is_quantum": kernel.is_quantum,
                "n_qubits": qcfg.n_qubits, "projection_dim": qcfg.projection_dim,
                "n_query": int(q), "bank_size": bank.size, "k": k,
                "quantum_neighbor_stats": q_stats.to_dict(),
                "overlaps": overlaps,
            }
            results.append(entry)
            save_json(entry, os.path.join(out_dir, "retrieval", f"{tag}_quantum.json"))
            log.info("Quantum fold=%s seed=%d overlaps=%s",
                     fold.test_subject, seed, overlaps)

    clog.flush()
    save_json({"results": results, "n": len(results)},
              os.path.join(out_dir, "quantum_summary.json"))
    log.info("Quantum subset retrieval complete: %d fold-seed entries.", len(results))
    return {"n_results": len(results), "output_dir": out_dir}
