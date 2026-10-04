import copy
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eegrag.config import load_config
from eegrag.data.gpu_scaling import CuPyFeaturePreprocessor
from eegrag.utils.arrays import validate_resident_config


@pytest.mark.parametrize("scaler", ["standard", "robust", "none"])
def test_resident_preprocessing_math_matches_cpu_train_only(scaler, monkeypatch):
    # Check the algorithm locally; CUDA execution has a separate integration test.
    from eegrag.data.scaling import FeaturePreprocessor
    monkeypatch.setitem(sys.modules, "cupy", np)
    cfg = load_config().preprocessing
    cfg.scaler = scaler
    cfg.backend = "cupy"
    train = np.array([[1, np.nan, 7, np.nan], [3, 4, 7, np.nan],
                      [5, 6, 7, np.nan], [8, np.inf, 7, np.nan]], np.float32)
    test = np.array([[100, 100, 7, 999], [np.inf, np.nan, 7, np.nan]], np.float32)
    original = train.copy()
    with pytest.warns(RuntimeWarning, match="All-NaN"):
        resident = CuPyFeaturePreprocessor(cfg)
        actual_train = resident.fit_transform(train)
    cpu = FeaturePreprocessor(cfg).fit(train)
    np.testing.assert_allclose(actual_train, cpu.transform(train), atol=2e-6)
    np.testing.assert_allclose(resident.transform(test), cpu.transform(test), atol=2e-6)
    np.testing.assert_array_equal(train, original)
    np.testing.assert_array_equal(resident._impute_values, [4, 5, 7, 0])


def test_resident_profiles_are_consistent_and_reject_host_fallback():
    for profile in ("contrastive_runpod", "contrastive_runpod_pilot", "contrastive_runpod_budget"):
        path = os.path.join(os.path.dirname(__file__), "..", "configs", profile + ".yaml")
        cfg = load_config(path)
        assert cfg.data.storage_backend == cfg.preprocessing.backend == "cupy"
        assert cfg.training.gpu_resident and cfg.training.tensor_batches
        assert cfg.retrieval.backend == "torch"
        cfg.retrieval.backend = "sklearn"
        with pytest.raises(ValueError, match="GPU resident path requires"):
            validate_resident_config(cfg)


def make_data(path):
    for subject, values in (("chb01", [1, 2, 3, 4]),
                            ("chb02", [10, 20, 30, 40]),
                            ("chb21", [5, 6, 7, 8])):
        pd.DataFrame({"windowStartSec": np.arange(4), "windowEndSec": np.arange(4) + 2,
                      "label": [0, 1, 0, 0], "AMean": values, "AStd": [1, 2, 3, 4],
                      "BMean": values, "BStd": [np.nan, 1, 2, 3]}).to_parquet(
            path / f"{subject}_01_features.parquet", index=False)


def test_streamed_loader_matches_host_order_and_source_signature(tmp_path, monkeypatch):
    from eegrag.data import loading
    make_data(tmp_path)
    cfg = load_config().data
    cfg.features_dir = str(tmp_path)
    expected = loading.assemble_dataset(cfg)
    cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(memGetInfo=lambda: (96 * 2**30, 96 * 2**30)),
                                 get_current_stream=lambda: types.SimpleNamespace(synchronize=lambda: None))
    fake = types.SimpleNamespace(empty=np.empty, float32=np.float32, cuda=cuda,
                                 asarray=lambda x, blocking: np.asarray(x))
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setattr(loading, "load_subject_frames", lambda *args: pytest.fail("Bulk host loading forbidden"))
    cfg.storage_backend = "cupy"
    actual = loading.assemble_dataset(cfg)
    np.testing.assert_array_equal(actual.features, expected.features)
    np.testing.assert_array_equal(actual.subjects, expected.subjects)
    assert actual.subject_order == ["chb01", "chb02"]
    assert actual.source_signature == loading.assemble_dataset(cfg).source_signature
    path = tmp_path / "chb02_01_features.parquet"
    changed = pd.read_parquet(path)
    changed.loc[0, "AMean"] = 123
    changed.to_parquet(path, index=False)
    assert actual.source_signature != loading.assemble_dataset(cfg).source_signature


def test_resource_tracker_sampler_can_join():
    from eegrag.utils.profiling import ResourceTracker
    with ResourceTracker() as tracker:
        sum(range(100))
    assert tracker.elapsed_sec >= 0


def test_inline_budget_evaluation_without_embedding_archives(tmp_path):
    from eegrag.experiments.loso_contrastive import run_loso_contrastive
    data = tmp_path / "data"
    data.mkdir()
    make_data(data)
    cfg = load_config(overrides=["seeds=[0]", "folds_whitelist=[chb01]",
        "training.save_embeddings=false", "training.tensor_batches=true", "training.device=cpu",
        "training.amp=false", "training.epochs=1", "training.steps_per_epoch=1",
        "training.batch_size=8", "encoder.arch=mlp", "encoder.hidden_dims=[8]",
        "retrieval.backend=sklearn", "retrieval.topk_values=[1,2]", "retrieval.decision_k=2"])
    cfg.data.features_dir = str(data)
    cfg.output_dir = str(tmp_path / "results")
    torch.set_num_threads(1)
    result = run_loso_contrastive(cfg)
    assert set(result["summary"]) == {"knn", "linear", "hybrid"}
    assert result["n_rows"] == 3
    assert not list((tmp_path / "results" / cfg.name / "embeddings").glob("*.npz"))
    assert (tmp_path / "results" / cfg.name / "predictions" / "chb01_seed0.npz").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable on local host")
def test_resident_cuda_fold_zero_copy_training_probe_and_retrieval(tmp_path, monkeypatch):
    cp = pytest.importorskip("cupy")
    pytest.importorskip("cuml")
    from eegrag.data import assemble_dataset, make_loso_folds
    from eegrag.preprocessing import prepare_fold
    from eegrag.datasets.tensor_batches import TensorBatches
    from eegrag.experiments import _engine
    from eegrag.experiments.retrieval_eval import evaluate_retrieval_for_fold
    from eegrag.retrieval import build_memory_bank

    make_data(tmp_path)
    cfg = load_config(overrides=["data.storage_backend=cupy", "preprocessing.backend=cupy",
        "training.gpu_resident=true", "training.tensor_batches=true", "training.device=cuda",
        "training.amp_dtype=bfloat16", "training.epochs=1", "training.steps_per_epoch=2",
        "training.batch_size=8", "encoder.arch=mlp", "encoder.hidden_dims=[8]",
        "retrieval.backend=torch", "retrieval.topk_values=[1,2]", "retrieval.decision_k=2"])
    cfg.data.features_dir = str(tmp_path)
    validate_resident_config(cfg)
    bundle = assemble_dataset(cfg.data)
    fold = make_loso_folds(bundle)[0]
    fa = prepare_fold(bundle, fold, cfg.preprocessing)
    assert isinstance(bundle.features, cp.ndarray)
    assert isinstance(fa.x_train, cp.ndarray)
    batches = TensorBatches(fa.x_train, fa.y_train, bundle.layout, cfg, "cuda")
    assert batches.x.data_ptr() == fa.x_train.data.ptr
    back = cp.from_dlpack(batches.x)
    assert back.data.ptr == fa.x_train.data.ptr
    original_asnumpy = cp.asnumpy
    def forbid_matrix_download(x, *args, **kwargs):
        if x.ndim == 2:
            pytest.fail("Feature/embedding matrix must not be downloaded for computation")
        return original_asnumpy(x, *args, **kwargs)
    monkeypatch.setattr(cp, "asnumpy", forbid_matrix_download)
    checkpoint = str(tmp_path / "encoder.pt")
    model = _engine.train_contrastive_encoder(fa.x_train, fa.y_train, bundle.layout, cfg,
        checkpoint_path=checkpoint, data_signature=bundle.source_signature)
    assert _engine.model_compute_stats(model, fa.x_train, bundle.layout, cfg)["n_params"] > 0
    train = _engine.extract_embeddings(model, fa.x_train, bundle.layout, cfg)
    test = _engine.extract_embeddings(model, fa.x_test, bundle.layout, cfg)
    assert isinstance(train, cp.ndarray) and bool(cp.isfinite(train).all())
    bank = build_memory_bank(train, fa.y_train, fa.subjects_train, test_subject=fold.test_subject)
    assert isinstance(bank.embeddings, cp.ndarray)
    probe = _engine.train_linear_head(train, fa.y_train, cfg)
    result = evaluate_retrieval_for_fold(bundle=bundle, fold=fold, train_emb=train,
        test_emb=test, cfg=cfg, seed=0, linear_clf=probe, save_dir=str(tmp_path / "results"))
    assert set(result.metrics) == {"knn", "linear", "hybrid"}
    changed = copy.deepcopy(cfg)
    changed.loss.name = "supcon"
    with pytest.raises(ValueError, match="Checkpoint data/config changed"):
        _engine.train_contrastive_encoder(fa.x_train, fa.y_train, bundle.layout, changed,
            checkpoint_path=checkpoint, data_signature=bundle.source_signature)
