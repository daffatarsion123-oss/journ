"""Classical baselines with GPU-capable backends and clean CPU fallbacks.

Backend policy (resolved at runtime, never assumes CUDA):

    LogisticRegression : cuML LogisticRegression  -> sklearn
    SVM (RBF)          : cuML SVC(kernel='rbf')    -> ThunderSVM -> sklearn SVC
    RandomForest       : cuML RandomForestClassifier -> sklearn
    XGBoost            : GPU hist (NVIDIA only)     -> CPU hist

Because the primary platform is an AMD MI300X (ROCm), most RAPIDS/cuML and
XGBoost-GPU paths (which are CUDA-only) will NOT be available there; the
fallbacks keep the classical baselines running on CPU with the ~240 GB RAM. The
abstraction stays clean so a CUDA box (or a future ROCm RAPIDS build) can light
up the GPU paths with no call-site changes.

``fit_classical_with_search`` performs a leakage-safe hyperparameter search
using *training-subject* GroupKFold only (groups == subject id), with
average-precision (AUC-PR) as the imbalance-aware selection metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import ClassicalConfig
from ..utils.backend import detect_backend, BackendInfo
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ClassicalModel:
    """Uniform wrapper exposing ``fit`` and ``predict_proba`` (P[class==1])."""

    name: str
    backend: str
    estimator: Any
    needs_dense_float32: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalModel":
        X = self._coerce(X)
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._coerce(X)
        proba = self.estimator.predict_proba(X)
        proba = _to_numpy(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    def _coerce(self, X: np.ndarray) -> np.ndarray:
        if self.needs_dense_float32:
            return np.ascontiguousarray(X, dtype=np.float32)
        return X


def _to_numpy(x: Any) -> np.ndarray:
    # cuML returns cudf/cupy; bring to host.
    for attr in ("to_numpy", "get", "values"):
        if hasattr(x, attr):
            try:
                out = getattr(x, attr)
                return np.asarray(out() if callable(out) else out)
            except Exception:
                continue
    return np.asarray(x)


# --------------------------------------------------------------------------- #
# Builders (per model), each returns (estimator, backend_label, needs_f32)
# --------------------------------------------------------------------------- #
def _build_logreg(cfg: ClassicalConfig, info: BackendInfo, **params):
    if cfg.prefer_gpu and info.cuml:
        try:
            from cuml.linear_model import LogisticRegression as cuLR
            return cuLR(max_iter=1000, class_weight=cfg.class_weight,
                        **_filter(params, {"C"})), "cuml", True
        except Exception as exc:  # pragma: no cover - depends on env
            log.warning("cuML LogisticRegression unavailable (%s); using sklearn", exc)
    from sklearn.linear_model import LogisticRegression
    return (
        LogisticRegression(
            max_iter=2000,
            class_weight=cfg.class_weight,
            n_jobs=-1,
            **_filter(params, {"C"}),
        ),
        "sklearn",
        False,
    )


def _build_svm_rbf(cfg: ClassicalConfig, info: BackendInfo, **params):
    C = params.get("C", 1.0)
    gamma = params.get("gamma", "scale")
    cw = params.get("class_weight", cfg.class_weight)
    if cfg.prefer_gpu and info.cuml:
        try:
            from cuml.svm import SVC as cuSVC
            return cuSVC(kernel="rbf", C=C, gamma=gamma, class_weight=cw,
                         probability=True), "cuml", True
        except Exception as exc:  # pragma: no cover
            log.warning("cuML SVC unavailable (%s); trying ThunderSVM", exc)
    if cfg.prefer_gpu and info.thundersvm:
        try:
            from thundersvm import SVC as tSVC
            return (
                tSVC(kernel="rbf", C=C, gamma=gamma, probability=True),
                "thundersvm",
                True,
            )
        except Exception as exc:  # pragma: no cover
            log.warning("ThunderSVM unavailable (%s); using sklearn SVC", exc)
    from sklearn.svm import SVC
    return (
        SVC(kernel="rbf", C=C, gamma=gamma, class_weight=cw, probability=True),
        "sklearn",
        False,
    )


def _build_random_forest(cfg: ClassicalConfig, info: BackendInfo, **params):
    if cfg.prefer_gpu and info.cuml:
        try:
            from cuml.ensemble import RandomForestClassifier as cuRF
            return (
                cuRF(
                    n_estimators=cfg.rf_n_estimators,
                    max_depth=cfg.rf_max_depth if cfg.rf_max_depth else 16,
                ),
                "cuml",
                True,
            )
        except Exception as exc:  # pragma: no cover
            log.warning("cuML RandomForest unavailable (%s); using sklearn", exc)
    from sklearn.ensemble import RandomForestClassifier
    return (
        RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            class_weight=cfg.class_weight,
            n_jobs=-1,
        ),
        "sklearn",
        False,
    )


def _build_xgboost(cfg: ClassicalConfig, info: BackendInfo, **params):
    if not info.xgboost:
        raise RuntimeError("xgboost not installed; remove it from classical.models")
    import xgboost as xgb

    # GPU histogram is NVIDIA-only. On ROCm/CPU we use tree_method='hist' on CPU.
    use_gpu = cfg.prefer_gpu and info.supports_xgb_gpu()
    device = "cuda" if use_gpu else "cpu"
    backend = "xgboost-gpu" if use_gpu else "xgboost-cpu"
    if cfg.prefer_gpu and info.is_rocm:
        log.info("XGBoost has no upstream ROCm GPU path; using CPU hist on MI300X.")
    return (
        xgb.XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            tree_method="hist",
            device=device,
            eval_metric="aucpr",
            n_jobs=-1,
        ),
        backend,
        False,
    )


_BUILDERS = {
    "logreg": _build_logreg,
    "svm_rbf": _build_svm_rbf,
    "random_forest": _build_random_forest,
    "xgboost": _build_xgboost,
}


def _ap_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-PR: try cuML (stays on GPU, faster) then sklearn."""
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)
    try:
        from cuml.metrics import average_precision_score as cuml_ap
        return float(cuml_ap(y_true, y_score))
    except Exception:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_score))


def _filter(params: Dict[str, Any], keep: set) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if k in keep}


def build_classical_model(
    name: str, cfg: ClassicalConfig, info: Optional[BackendInfo] = None, **params
) -> ClassicalModel:
    info = info or detect_backend()
    if name not in _BUILDERS:
        raise ValueError(f"Unknown classical model {name!r}")
    est, backend, needs_f32 = _BUILDERS[name](cfg, info, **params)
    log.info("Built classical model %s on backend=%s params=%s", name, backend, params)
    return ClassicalModel(name=name, backend=backend, estimator=est,
                          needs_dense_float32=needs_f32)


# --------------------------------------------------------------------------- #
# Hyperparameter search (training subjects only, GroupKFold by subject)
# --------------------------------------------------------------------------- #
def _param_grid(name: str, cfg: ClassicalConfig) -> List[Dict[str, Any]]:
    if name == "svm_rbf":
        grid = []
        for C in cfg.svm_C_grid:
            for g in cfg.svm_gamma_grid:
                for cw in cfg.svm_class_weight_grid:
                    grid.append({"C": C, "gamma": g, "class_weight": cw})
        return grid
    if name == "logreg":
        return [{"C": C} for C in (0.01, 0.1, 1.0, 10.0)]
    # RF / XGB: single configuration (their headline knobs live in cfg).
    return [{}]


def fit_classical_with_search(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    cfg: ClassicalConfig,
    info: Optional[BackendInfo] = None,
    seed: int = 0,
) -> Tuple[ClassicalModel, Dict[str, Any]]:
    """Select hyperparameters via subject-grouped CV on TRAIN ONLY, then refit.

    The CV groups are subject ids, so no subject appears in both inner-train and
    inner-validation -- the HP search itself is LOSO-consistent and leak-free.
    Selection metric: average precision (AUC-PR).
    """
    from sklearn.model_selection import GroupKFold

    info = info or detect_backend()
    grid = _param_grid(name, cfg)
    n_groups = len(np.unique(subjects))
    n_splits = min(cfg.inner_cv_folds, max(2, n_groups))

    best_params: Dict[str, Any] = {}
    best_score = -np.inf
    search_log: List[Dict[str, Any]] = []

    if len(grid) == 1:
        best_params = grid[0]
    else:
        gkf = GroupKFold(n_splits=n_splits)
        for params in grid:
            scores = []
            for tr, va in gkf.split(X, y, groups=subjects):
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                    continue
                model = build_classical_model(name, cfg, info, **params)
                model.fit(X[tr], y[tr])
                p = model.predict_proba(X[va])
                scores.append(_ap_score(y[va], p))
            mean_score = float(np.mean(scores)) if scores else -np.inf
            search_log.append({"params": params, "cv_aucpr": mean_score})
            if mean_score > best_score:
                best_score, best_params = mean_score, params

    final = build_classical_model(name, cfg, info, **best_params).fit(X, y)
    return final, {
        "best_params": best_params,
        "best_cv_aucpr": best_score,
        "search": search_log,
        "n_splits": n_splits,
        "backend": final.backend,
    }
