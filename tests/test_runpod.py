import copy
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eegrag.config import load_config
from eegrag.data.feature_layout import FeatureLayout
from eegrag.data.loading import load_subject_frames
from eegrag.datasets.tensor_batches import TensorBatches
from eegrag.experiments import _engine
from eegrag.losses.supcon import SupConLoss, SimCLRLoss, ImbalanceAwareSupConLoss
from eegrag.retrieval.index import TorchNeighborIndex
from eegrag.utils.seed import seed_everything


def test_contrastive_views_are_positive_and_other_classes_are_negative():
    features = torch.tensor([[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]]], requires_grad=True)
    loss = SupConLoss()(features, torch.tensor([0, 1]))
    assert loss.item() < 0.001
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert SimCLRLoss()(features).item() < 0.001


def test_focal_loss_matches_single_core_reference_and_has_finite_gradient():
    from eegrag.losses.supcon import _supcon_core
    cfg = load_config().loss
    labels = torch.tensor([0, 0, 1])
    x = torch.nn.functional.normalize(torch.randn(3, 2, 8), dim=-1).requires_grad_()
    objective = ImbalanceAwareSupConLoss(cfg)
    _, per_anchor, valid, mask, log_prob, count = _supcon_core(
        x, labels, cfg.temperature, cfg.base_temperature, return_per_anchor=True)
    p = torch.exp((mask * log_prob).sum(1) / count)
    weight = objective._class_weights(labels) * ((1 - p) ** cfg.focal_gamma).view(2, 3).mean(0)
    expected = (per_anchor * weight.repeat(2))[valid].mean()
    actual = objective(x, labels)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(x.grad).all()


def test_gpu_lr_and_svm_builders_preserve_class_weights(monkeypatch):
    import types
    from eegrag.models.classical import build_classical_model
    from eegrag.utils.backend import BackendInfo
    class Estimator:
        def __init__(self, **kwargs):
            self.options = kwargs
    monkeypatch.setitem(sys.modules, "cuml.linear_model", types.SimpleNamespace(LogisticRegression=Estimator))
    monkeypatch.setitem(sys.modules, "cuml.svm", types.SimpleNamespace(SVC=Estimator))
    cfg = load_config().classical
    info = BackendInfo(cuml=True)
    lr = build_classical_model("logreg", cfg, info)
    svm = build_classical_model("svm_rbf", cfg, info, class_weight=None)
    assert lr.estimator.options["class_weight"] == "balanced"
    assert svm.estimator.options["class_weight"] is None


@pytest.mark.parametrize("metric", ["ip", "l2"])
def test_tiled_retrieval_matches_brute_force_across_bank_boundaries(metric):
    rng = np.random.default_rng(8)
    bank = rng.normal(size=(31, 8)).astype(np.float32)
    queries = rng.normal(size=(7, 8)).astype(np.float32)
    index = TorchNeighborIndex(bank, metric, device="cpu", query_chunk_size=3, bank_chunk_size=6)
    idx, values = index.query(queries, 10)
    if metric == "ip":
        scores = queries / np.linalg.norm(queries, axis=1, keepdims=True) @ (
            bank / np.linalg.norm(bank, axis=1, keepdims=True)).T
        expected_idx = np.argsort(-scores, axis=1)[:, :10]
    else:
        scores = np.linalg.norm(queries[:, None] - bank[None], axis=-1)
        expected_idx = np.argsort(scores, axis=1)[:, :10]
    np.testing.assert_array_equal(idx, expected_idx)
    np.testing.assert_allclose(values, np.take_along_axis(scores, expected_idx, axis=1), atol=1e-6)


def test_balanced_batches_and_group_masks():
    cfg = load_config(overrides=["training.batch_size=256", "training.steps_per_epoch=1",
        "augmentation.gaussian_noise_p=0", "augmentation.amplitude_scale_p=0",
        "augmentation.channel_dropout_p=1", "augmentation.channel_dropout_max=1",
        "augmentation.freq_masking_p=0"])
    layout = FeatureLayout.from_columns(["AMean", "AStd", "BMean", "BStd"])
    x = np.ones((100, 4), dtype=np.float32)
    y = np.array([0] * 99 + [1])
    batches = TensorBatches(x, y, layout, cfg, "cpu")
    views, labels, _ = next(iter(batches))
    assert 0.35 < labels.float().mean() < 0.65
    assert torch.equal(views[..., 0], views[..., 1])
    assert torch.equal(views[..., 2], views[..., 3])
    assert (views.sum(-1) == 2).all()
    np.testing.assert_array_equal(x, np.ones_like(x))


def test_patient_aliases_group_chb01_and_chb21(tmp_path):
    import pandas as pd
    paths = []
    for subject in ("chb01", "chb21"):
        path = tmp_path / f"{subject}_01_features.parquet"
        pd.DataFrame({"label": [0]}).to_parquet(path)
        paths.append(str(path))
    grouped = load_subject_frames(paths, load_config().data)
    assert list(grouped) == ["chb01"]
    assert len(grouped["chb01"]) == 2


def test_explicit_schema_keeps_missing_channels_as_nan_and_invalidates_cache(tmp_path):
    import json
    import pandas as pd
    from eegrag.data import assemble_dataset
    data = tmp_path / "data"
    data.mkdir()
    for subject, features in (("chb01", {"AMean": [1.], "AStd": [2.]}),
                              ("chb02", {"AMean": [3.]})):
        pd.DataFrame({"windowStartSec": [0.], "windowEndSec": [2.],
                      "label": [0], **features}).to_parquet(data / f"{subject}_01_features.parquet")
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(["AMean", "AStd"]))
    cfg = load_config().data
    cfg.features_dir = str(data)
    cfg.cache_dir = str(tmp_path / "cache")
    cfg.feature_schema_path = str(schema)
    with pytest.raises(ValueError, match="missing feature columns"):
        assemble_dataset(cfg)
    cfg.missing_feature_policy = "impute"
    first = assemble_dataset(cfg)
    assert first.features.shape == (2, 2)
    assert np.isnan(first.features[1, 1])
    schema.write_text(json.dumps(["AMean"]))
    second = assemble_dataset(cfg)
    assert second.features.shape == (2, 1)


def test_preprocessing_keeps_training_medians_and_does_not_mutate_inputs():
    from eegrag.data.scaling import FeaturePreprocessor
    cfg = load_config().preprocessing
    x = np.array([[1, np.nan], [3, 4], [5, 6]], dtype=np.float32)
    original = x.copy()
    preprocessor = FeaturePreprocessor(cfg).fit(x)
    result = preprocessor.transform(np.array([[np.inf, np.nan]], dtype=np.float32))
    np.testing.assert_allclose(result, [[0, 0]], atol=1e-6)
    np.testing.assert_array_equal(x, original)


def test_leave_self_out_removes_identity_even_when_duplicates_rank_first():
    from eegrag.experiments.retrieval_eval import _knn_scores_leave_self_out
    from eegrag.retrieval.memory_bank import RetrievalResult
    class Bank:
        size = 3
        def retrieve(self, queries, k, **kwargs):
            indices = np.array([[1, 0, 2], [0, 1, 2], [0, 2, 1]])
            labels = np.array([0, 1, 1])
            return RetrievalResult(indices, np.ones((3, 3)), np.ones((3, 3)),
                                   labels[indices], np.full((3, 3), "train"), "cosine", 3)
    score = _knn_scores_leave_self_out(Bank(), np.ones((3, 2)), 2, "cosine", 1)
    np.testing.assert_allclose(score, [1, 0.5, 0.5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable on local host")
def test_cuda_bf16_training_and_gpu_search_are_finite():
    cfg = load_config(overrides=["training.tensor_batches=true", "training.device=cuda",
        "training.amp_dtype=bfloat16", "training.epochs=1", "training.steps_per_epoch=2",
        "training.batch_size=16", "encoder.arch=mlp", "encoder.hidden_dims=[8]"])
    x = np.random.default_rng(0).normal(size=(64, 4)).astype(np.float32)
    labels = np.array([0, 1] * 32)
    layout = FeatureLayout.from_columns(["AMean", "AStd", "BMean", "BStd"])
    model = _engine.train_contrastive_encoder(x, labels, layout, cfg)
    emb = _engine.extract_embeddings(model, x, layout, cfg)
    assert np.isfinite(emb).all()
    idx, score = TorchNeighborIndex(emb, "ip").query(emb[:5], 3)
    assert idx.shape == (5, 3) and np.isfinite(score).all()


def test_epoch_restart_matches_uninterrupted_training(tmp_path, monkeypatch):
    cfg = load_config(overrides=["training.tensor_batches=true", "training.device=cpu",
        "training.amp=false", "training.num_workers=0", "training.epochs=2",
        "training.steps_per_epoch=2", "training.batch_size=8", "encoder.arch=mlp",
        "encoder.hidden_dims=[8]", "encoder.embedding_dim=4", "encoder.projection_dim=4"])
    rng = np.random.default_rng(2)
    x = rng.normal(size=(32, 4)).astype(np.float32)
    y = np.array([0, 1] * 16)
    layout = FeatureLayout.from_columns(["AMean", "AStd", "BMean", "BStd"])
    torch.set_num_threads(1)
    seed_everything(4)
    expected = _engine.train_contrastive_encoder(x, y, layout, cfg, 4)
    checkpoint = str(tmp_path / "checkpoint.pt")
    replace = os.replace
    def interrupt_after_checkpoint(source, target):
        replace(source, target)
        raise RuntimeError("simulated interruption")
    seed_everything(4)
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupt_after_checkpoint)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            _engine.train_contrastive_encoder(x, y, layout, cfg, 4, checkpoint)
    seed_everything(4)
    resumed = _engine.train_contrastive_encoder(x, y, layout, cfg, 4, checkpoint)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    embedded = _engine.extract_embeddings(resumed, x, layout, cfg, batch_size=7)
    assert embedded.shape == (32, 4)
    assert np.isfinite(embedded).all()
    changed = copy.deepcopy(cfg)
    changed.loss.name = "supcon"
    with pytest.raises(ValueError, match="Checkpoint data/config changed"):
        _engine.train_contrastive_encoder(x, y, layout, changed, 4, checkpoint)
