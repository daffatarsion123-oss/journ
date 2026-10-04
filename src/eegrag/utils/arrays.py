"""Explicit host boundaries and zero-copy CuPy / torch interoperability."""
import numpy as np


def is_cuda_array(x):
    return hasattr(x, "__cuda_array_interface__")


def array_module(x):
    if is_cuda_array(x):
        import cupy as cp
        return cp
    return np


def to_host(x):
    if is_cuda_array(x):
        import cupy as cp
        return cp.asnumpy(x)
    return np.asarray(x)


def to_tensor(x, device):
    import torch
    if is_cuda_array(x):
        tensor = torch.from_dlpack(x)
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        if tensor.device != target:
            raise ValueError("GPU resident array and torch device must match")
        return tensor
    return torch.as_tensor(np.ascontiguousarray(x), device=device)


def synchronize(x):
    if is_cuda_array(x):
        array_module(x).cuda.get_current_stream().synchronize()


def validate_resident_config(cfg):
    if not cfg.training.gpu_resident:
        if cfg.data.storage_backend == "cupy":
            raise ValueError("CuPy feature storage requires training.gpu_resident=true")
        return
    if (cfg.training.device != "cuda" or not cfg.training.tensor_batches
            or cfg.data.storage_backend != "cupy" or cfg.preprocessing.backend != "cupy"
            or cfg.retrieval.backend != "torch" or cfg.encoder.sequence_len != 1):
        raise ValueError("GPU resident path requires CUDA tensor batches, CuPy storage/preprocessing, torch retrieval and single windows")
    if cfg.preprocessing.use_pca or cfg.preprocessing.feature_selection_k:
        raise ValueError("Resident preprocessing does not support PCA/feature selection")
    try:
        import cupy
        from cuml.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("GPU resident path requires CuPy and cuML; install requirements-runpod-rapids.txt") from exc
