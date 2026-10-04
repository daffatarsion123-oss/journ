"""Time the actual first LOSO training fold; extrapolate without running a full sweep."""
from _bootstrap import base_parser, load
import copy
import json
import math
import time
import tempfile


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--losses", type=int, default=1)
    parser.add_argument("--hourly-rate", type=float)
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--end-to-end", action="store_true",
                        help="Also time full first-fold heads, retrieval, metrics and output writes")
    args = parser.parse_args()
    if args.steps < 2 or args.losses < 1:
        parser.error("--steps must be >=2 and --losses >=1")
    cfg = load(args)
    from eegrag.data import assemble_dataset, make_loso_folds
    from eegrag.preprocessing import prepare_fold
    from eegrag.experiments import _engine
    from eegrag.utils.seed import seed_everything
    from eegrag.utils.io import save_json, ensure_dir
    from eegrag.utils.arrays import to_host, validate_resident_config
    import torch
    validate_resident_config(cfg)
    seed_everything(cfg.seeds[0], deterministic_torch=cfg.training.deterministic)
    started = time.perf_counter()
    bundle = assemble_dataset(cfg.data)
    folds = make_loso_folds(bundle, folds_whitelist=cfg.folds_whitelist)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    fa = prepare_fold(bundle, folds[0], cfg.preprocessing)
    preprocessing_seconds = time.perf_counter() - started
    trial = copy.deepcopy(cfg)
    trial.training.tensor_batches = True
    trial.training.steps_per_epoch = args.steps
    trial.training.epochs = 2
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = _engine.train_contrastive_encoder(fa.x_train, fa.y_train, bundle.layout,
                                             trial, cfg.seeds[0])
    encoder_setup_seconds = time.perf_counter() - started - sum(model.training_profile["epoch_seconds"])
    seconds_per_step = model.training_profile["epoch_seconds"][-1] / args.steps
    total_steps = sum(cfg.training.steps_per_epoch or max(1, len(f.train_idx) // cfg.training.batch_size)
                      for f in folds) * cfg.training.epochs * len(cfg.seeds) * args.losses
    training_hours = seconds_per_step * total_steps / 3600
    report = {
        "benchmark_only": True, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "n_windows": bundle.n_windows, "n_features": bundle.features.shape[1],
        "n_patients": len(bundle.subject_order), "n_test_folds": len(folds),
        "seeds": len(cfg.seeds), "losses": args.losses, "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size, "configured_steps_per_epoch": cfg.training.steps_per_epoch,
        "preprocessing_seconds_first_fold": preprocessing_seconds,
        "load_seconds": load_seconds, "encoder_setup_seconds_first_fold": encoder_setup_seconds,
        "seconds_per_step": seconds_per_step, "total_training_steps": total_steps,
        "training_hours": training_hours, "training_hours_with_30_percent_margin": training_hours * 1.3,
        "peak_torch_vram_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "gpu_resident": cfg.training.gpu_resident,
        "limitations": "First-fold extrapolation. Excludes other-fold preprocessing, embeddings, retrieval, linear probes, IO and classical/quantum models.",
    }
    if args.hourly_rate is not None:
        report["training_cost_with_margin"] = training_hours * 1.3 * args.hourly_rate
    if args.retrieval or args.end_to_end:
        from eegrag.retrieval import build_memory_bank
        started = time.perf_counter()
        train_emb = _engine.extract_embeddings(model, fa.x_train, bundle.layout, trial)
        test_emb = _engine.extract_embeddings(model, fa.x_test, bundle.layout, trial)
        report["embedding_seconds_first_fold"] = time.perf_counter() - started
        if args.end_to_end:
            from eegrag.experiments.retrieval_eval import evaluate_retrieval_for_fold
            started = time.perf_counter()
            probe = _engine.train_linear_head(train_emb, fa.y_train, cfg, cfg.seeds[0])
            with tempfile.TemporaryDirectory(prefix="eegrag_benchmark_") as temporary:
                if cfg.training.save_embeddings:
                    import numpy as np
                    np.savez_compressed(temporary + "/embeddings.npz", train_emb=to_host(train_emb),
                                        test_emb=to_host(test_emb))
                evaluate_retrieval_for_fold(bundle=bundle, fold=folds[0],
                    train_emb=train_emb, test_emb=test_emb, cfg=cfg, seed=cfg.seeds[0],
                    linear_clf=probe, save_dir=temporary)
            report["evaluation_seconds_first_fold"] = time.perf_counter() - started
            report["post_training_seconds_first_fold"] = (
                report["embedding_seconds_first_fold"] + report["evaluation_seconds_first_fold"])
            report["limitations"] = "First-fold measurement, scaled across patients. Includes full test scoring, heads and writes; excludes classical baselines and quantum. Timing-only warmup model."
        bank = build_memory_bank(train_emb, fa.y_train, fa.subjects_train,
                                 test_subject=folds[0].test_subject,
                                 max_per_class=cfg.retrieval.max_bank_per_class,
                                 query_chunk_size=cfg.retrieval.query_chunk_size,
                                 bank_chunk_size=cfg.retrieval.bank_chunk_size)
        query = test_emb[:min(1024, len(test_emb))]
        bank.retrieve(query[:32], max(cfg.retrieval.topk_values), backend=cfg.retrieval.backend)
        before = bank.query_sec
        bank.retrieve(query, max(cfg.retrieval.topk_values), backend=cfg.retrieval.backend)
        seconds_per_query = (bank.query_sec - before) / len(query)
        report["retrieval_bank_size"] = bank.size
        report["retrieval_seconds_per_query"] = seconds_per_query
        report["retrieval_hours_first_fold_estimate"] = seconds_per_query * (bank.size + 2 * len(test_emb)) / 3600
        report["retrieval_note"] = "Approximation for calibration + two metric searches; excludes neighbor summaries and linear probe. Warmup encoders are for timing only."
    ensure_dir(cfg.output_dir)
    if cfg.training.gpu_resident:
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        report["cupy_pool_used_gib"] = cp.get_default_memory_pool().used_bytes() / 2**30
        report["cupy_pool_reserved_gib"] = cp.get_default_memory_pool().total_bytes() / 2**30
        report["device_used_gib_at_report"] = (total - free) / 2**30
    save_json(report, cfg.output_dir + "/runpod_benchmark.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
