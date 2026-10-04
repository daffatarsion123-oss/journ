"""Shared training / embedding engine for contrastive LOSO runs (torch).

Kept separate from the runners so torch is imported lazily -- the classical and
stats pipelines run without a torch install.
"""
from __future__ import annotations

from typing import Optional, Tuple
from functools import partial
import hashlib
import json
import os
import random
import time

import numpy as np

from ..config import ExperimentConfig
from ..data.feature_layout import FeatureLayout
from ..datasets import (
    ContrastiveViewDataset,
    EEGWindowDataset,
    build_augmenter,
    make_balanced_sampler,
)
from ..losses import build_loss
from ..models import ContrastiveModel
from ..utils.backend import resolve_torch_device
from ..utils.logging import get_logger
from ..utils.seed import worker_init_fn
from ..utils.arrays import is_cuda_array, to_tensor

log = get_logger(__name__)


def _make_optimizer(model, cfg):
    import torch

    if cfg.training.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
    if cfg.training.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.training.lr,
                                weight_decay=cfg.training.weight_decay)
    if cfg.training.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.training.lr,
                               momentum=0.9, weight_decay=cfg.training.weight_decay)
    raise ValueError(f"Unknown optimizer {cfg.training.optimizer!r}")


def _make_scheduler(optimizer, cfg):
    import torch

    if cfg.training.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.training.epochs)
        )
    if cfg.training.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return None


def train_contrastive_encoder(
    x_train: np.ndarray,
    y_train: np.ndarray,
    layout: FeatureLayout,
    cfg: ExperimentConfig,
    seed: int = 0,
    checkpoint_path: Optional[str] = None,
    data_signature: Optional[str] = None,
) -> ContrastiveModel:
    """Train the encoder + projection head with the configured contrastive loss.

    Receives ALREADY-transformed training features (leakage-safe). The test
    subject is not present here by construction (caller passes train rows only).
    """
    import torch
    from torch.utils.data import DataLoader

    device = resolve_torch_device(cfg.training.device)
    log.info("Training contrastive encoder on device=%s seed=%d amp=%s dtype=%s",
             device, seed, cfg.training.amp, cfg.training.amp_dtype)

    if cfg.training.tensor_batches:
        from ..datasets.tensor_batches import TensorBatches
        loader = TensorBatches(x_train, y_train, layout, cfg, device)
    else:
        augmenter = build_augmenter(cfg.augmentation, layout)
        dataset = ContrastiveViewDataset(
            x_train, y_train, np.zeros(len(y_train), dtype=object), layout,
            sequence_len=cfg.encoder.sequence_len,
            augmenter=augmenter, n_views=cfg.augmentation.n_views,
        )

        if cfg.training.balanced_sampler:
            sampler = make_balanced_sampler(
                dataset.start_labels, minority_class=cfg.loss.minority_class,
                pos_boost=max(1.0, cfg.training.min_pos_per_batch),
            )
            shuffle = False
        else:
            sampler, shuffle = None, True

        if cfg.training.steps_per_epoch is not None:
            raise ValueError("steps_per_epoch requires tensor_batches=true")
        loader = DataLoader(
            dataset, batch_size=cfg.training.batch_size, sampler=sampler,
            shuffle=shuffle, num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory and device == "cuda",
            drop_last=True, worker_init_fn=partial(worker_init_fn, base_seed=seed),
            persistent_workers=cfg.training.num_workers > 0,
        )
    if len(loader) == 0:
        raise ValueError("Training dataset is smaller than one batch")
    torch.backends.cuda.matmul.allow_tf32 = cfg.training.allow_tf32
    torch.backends.cudnn.allow_tf32 = cfg.training.allow_tf32

    model = ContrastiveModel(cfg.encoder, x_train.shape[1], layout).to(device)
    criterion = build_loss(cfg.loss)
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)
    use_amp = cfg.training.amp and device == "cuda"
    amp_dtype = getattr(torch, cfg.training.amp_dtype) if use_amp else torch.float32
    # BF16 has wide dynamic range (no underflow), so GradScaler is unnecessary.
    # FP16 needs it to avoid gradient underflow.
    use_scaler = use_amp and cfg.training.amp_dtype == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_epoch = 0
    epoch_times = []
    signature = None
    if checkpoint_path:
        digest = hashlib.sha256()
        if is_cuda_array(x_train):
            if not data_signature:
                raise ValueError("GPU checkpoints require a source/fold data signature")
            digest.update(data_signature.encode())
        else:
            digest.update(memoryview(np.ascontiguousarray(x_train)).cast("B"))
        digest.update(memoryview(np.ascontiguousarray(y_train)).cast("B"))
        digest.update(json.dumps(cfg.to_dict(), sort_keys=True).encode())
        digest.update(str(seed).encode())
        signature = digest.hexdigest()
    if checkpoint_path and os.path.exists(checkpoint_path):
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Checkpoint data/config changed; choose a new experiment name")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        start_epoch = saved["epoch"]
        epoch_times = saved["epoch_times"]
        log.info("Resuming %s at epoch %d", checkpoint_path, start_epoch)
    model.train()
    for epoch in range(start_epoch, cfg.training.epochs):
        if device == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        running = torch.zeros((), device=device)
        n_batches = 0
        for views, labels, _ in loader:
            # views: [B, V, ...] -> stack views into the batch for projection
            b, v = views.shape[0], views.shape[1]
            flat = views.reshape(b * v, *views.shape[2:]).to(device, non_blocking=True)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                proj = model.project(flat)                  # [B*V, P]
                proj = proj.reshape(b, v, -1)
                loss = criterion(proj, labels)
            scaler.scale(loss).backward()
            if cfg.training.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += loss.detach()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        if device == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - epoch_start)
        if n_batches and (epoch % max(1, cfg.training.log_every // 10) == 0
                          or epoch == cfg.training.epochs - 1):
            log.info("epoch %d/%d  loss=%.4f", epoch + 1, cfg.training.epochs,
                     float(running.cpu()) / max(n_batches, 1))
        if checkpoint_path and ((epoch + 1) % cfg.training.checkpoint_every == 0
                                or epoch + 1 == cfg.training.epochs):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                "signature": signature, "epoch": epoch + 1,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict(), "epoch_times": epoch_times,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            }, checkpoint_path + ".tmp")
            os.replace(checkpoint_path + ".tmp", checkpoint_path)
    model.training_profile = {"epoch_seconds": epoch_times, "steps_per_epoch": len(loader),
                              "n_train": len(y_train), "batch_size": cfg.training.batch_size}
    return model


def model_compute_stats(
    model: ContrastiveModel,
    x_sample: np.ndarray,
    layout: FeatureLayout,
    cfg: ExperimentConfig,
) -> dict:
    """Parameter count + per-sample forward FLOPs for the trained encoder.

    Builds a single dataset-shaped example so the FLOP estimator sees exactly the
    tensor shape the encoder is trained/evaluated on.
    """
    import torch
    from ..utils.profiling import count_parameters, estimate_flops

    device = next(model.parameters()).device
    if is_cuda_array(x_sample):
        xb = to_tensor(x_sample[:1], device)
    else:
        ds = EEGWindowDataset(
            np.asarray(x_sample[:1]), np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=object), layout, sequence_len=cfg.encoder.sequence_len,
        )
        xb, _, _ = ds[0]
        xb = xb.unsqueeze(0).to(device)
    n_params, n_trainable = count_parameters(model)
    try:
        flops = estimate_flops(model, xb)
    except Exception as exc:  # pragma: no cover - estimator is best-effort
        log.warning("FLOP estimation failed (%s); reporting NaN.", exc)
        flops = float("nan")
    return {"n_params": n_params, "n_trainable": n_trainable, "flops": flops}


def extract_embeddings(
    model: ContrastiveModel,
    x: np.ndarray,
    layout: FeatureLayout,
    cfg: ExperimentConfig,
    batch_size: int = 4096,
    normalize: bool = True,
) -> np.ndarray:
    """Extract L2-normalizable encoder embeddings for a feature matrix."""
    import torch
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    if cfg.training.tensor_batches:
        model.eval()
        resident = cfg.training.gpu_resident
        if resident and (device.type != "cuda" or not is_cuda_array(x)):
            raise ValueError("Resident embeddings require CuPy features on CUDA")
        if resident:
            out = torch.empty((len(x), model.embedding_dim), dtype=torch.float32, device=device)
            source = to_tensor(x, device)
        else:
            out = np.empty((len(x), model.embedding_dim), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                xb = source[start:start + batch_size] if resident else to_tensor(x[start:start + batch_size], device)
                with torch.amp.autocast("cuda", enabled=cfg.training.amp and device.type == "cuda",
                                        dtype=getattr(torch, cfg.training.amp_dtype)):
                    emb = model.embed(xb, normalize=normalize)
                out[start:start + len(xb)] = emb.float() if resident else emb.float().cpu().numpy()
        if resident:
            import cupy as cp
            return cp.from_dlpack(out)
        return out
    dataset = EEGWindowDataset(
        x, np.zeros(len(x), dtype=np.int64), np.zeros(len(x), dtype=object),
        layout, sequence_len=cfg.encoder.sequence_len,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=cfg.training.num_workers)
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _, _ in loader:
            xb = xb.to(device)
            emb = model.embed(xb, normalize=normalize)
            out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def train_linear_head(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cfg: ExperimentConfig,
    seed: int = 0,
    epochs: int = 100,
) -> Tuple[np.ndarray, float]:
    """Train a class-weighted linear probe on (frozen) embeddings.

    Returns ``(weight_bias_packed, )`` is overkill; instead we return a callable
    via closure is awkward to serialize, so we return the fitted sklearn-style
    coefficients applied through a small helper. For simplicity and zero extra
    deps we fit a logistic regression on the embeddings (CPU, fast).
    """
    if cfg.retrieval.linear_probe_max_negatives is not None:
        from ..retrieval.memory_bank import _subsample_majority
        selected = _subsample_majority(labels, np.zeros(len(labels)),
                                      cfg.retrieval.linear_probe_max_negatives, seed)
        embeddings, labels = embeddings[selected], labels[selected]
    if is_cuda_array(embeddings):
        from cuml.linear_model import LogisticRegression as CuLogisticRegression
        import cupy as cp
        clf = CuLogisticRegression(max_iter=2000, class_weight="balanced",
                                   C=1.0, output_type="numpy")
        clf.fit(embeddings, cp.asarray(labels))
        return clf
    if cfg.classical.prefer_gpu and cfg.training.device != "cpu":
        try:
            from cuml.linear_model import LogisticRegression as CuLogisticRegression
        except ImportError:
            log.info("cuML unavailable; fitting linear probe on CPU")
        else:
            clf = CuLogisticRegression(max_iter=2000, class_weight="balanced",
                                       C=1.0, output_type="numpy")
            clf.fit(embeddings, labels)
            return clf
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", n_jobs=-1, C=1.0,
    )
    clf.fit(embeddings, labels)
    return clf
