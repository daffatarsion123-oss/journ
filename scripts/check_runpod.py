"""Execute GPU kernels before starting a paid training run."""
import argparse
import json
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rapids", action="store_true")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use the RunPod CUDA PyTorch image")
    cuda = tuple(int(x) for x in torch.version.cuda.split(".")[:2])
    if cuda < (12, 8):
        raise RuntimeError("Blackwell requires a CUDA 12.8+ PyTorch build")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable")
    x = torch.randn(32, 8, 23, device="cuda", requires_grad=True)
    conv = torch.nn.Conv1d(8, 64, 3, padding=1).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = conv(x).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    import cupy as cp
    if cp.cuda.runtime.runtimeGetVersion() < 12080:
        raise RuntimeError("CuPy selected a pre-12.8 CUDA runtime")
    if cp.cuda.nvrtc.getVersion() < (12, 8):
        raise RuntimeError("CuPy selected a pre-12.8 NVRTC compiler")
    a = cp.arange(4096, dtype=cp.float32).reshape(64, 64)
    value = (cp.sin(a) + a @ a.T).sum()
    cp.fft.rfft(a)
    cp.cuda.Stream.null.synchronize()
    if not bool(cp.isfinite(value)):
        raise RuntimeError("CuPy kernel smoke test returned a nonfinite value")
    shared = torch.from_dlpack(a)
    if shared.data_ptr() != a.data.ptr:
        raise RuntimeError("CuPy -> PyTorch DLPack did not share device memory")
    returned = cp.from_dlpack(shared)
    if returned.data.ptr != a.data.ptr:
        raise RuntimeError("PyTorch -> CuPy DLPack did not share device memory")
    from _bootstrap import _ensure_src_on_path
    _ensure_src_on_path()
    from eegrag.config import PreprocessingConfig
    from eegrag.data.gpu_scaling import CuPyFeaturePreprocessor
    test = cp.array([[1, cp.nan], [3, 4], [5, 6]], dtype=cp.float32)
    preprocessor = CuPyFeaturePreprocessor(PreprocessingConfig(backend="cupy"))
    transformed = preprocessor.fit_transform(test)
    held_out = preprocessor.transform(cp.array([[cp.inf, cp.nan]], dtype=cp.float32))
    if not bool(cp.isfinite(transformed).all()) or not bool(cp.allclose(held_out, 0)):
        raise RuntimeError("CuPy train-only imputation/scaling smoke failed")
    report = {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "torch_arches": torch.cuda.get_arch_list(), "cupy": cp.__version__,
        "cupy_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "nvrtc": cp.cuda.nvrtc.getVersion(), "gpu_smoke": "passed",
        "dlpack_zero_copy": "passed", "cupy_preprocessing": "passed",
    }
    if args.rapids:
        from cuml.linear_model import LogisticRegression
        features = cp.random.RandomState(0).normal(size=(128, 8)).astype(cp.float32)
        labels = (features[:, 0] > 0).astype(cp.int32)
        model = LogisticRegression(class_weight="balanced", output_type="numpy").fit(features, labels)
        model.predict_proba(features)
        cp.cuda.Stream.null.synchronize()
        report["rapids_smoke"] = "passed"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
