"""Leakage-safe feature preprocessing (fit on TRAINING subjects only).

Wraps scaler + optional PCA + optional univariate selection behind a single
object whose ``fit`` *must* be called with training data only. ``transform`` is
then applied to train/test alike. Non-finite values (occasional NaN/Inf from the
entropy / band-power extraction) are imputed with training-set column medians.

GPU acceleration:
    cuML StandardScaler / RobustScaler / PCA are tried first when cuML is
    installed (A100 path). If cuML is absent the code falls back to sklearn
    transparently -- no config change needed. The active backend is logged so
    you can confirm which path ran.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

from ..config import PreprocessingConfig
from ..utils.logging import get_logger

log = get_logger(__name__)


def _to_numpy(x: Any) -> np.ndarray:
    """Coerce cuML / cudf / cupy output back to a host numpy array."""
    for attr in ("to_numpy", "get", "values"):
        if hasattr(x, attr):
            try:
                val = getattr(x, attr)
                return np.asarray(val() if callable(val) else val)
            except Exception:
                continue
    return np.asarray(x)


def _build_scaler(name: str) -> Tuple[Any, str]:
    """Return (scaler_object, backend_label). cuML first, sklearn fallback."""
    if name == "none":
        return None, "none"
    if name in ("standard", "robust"):
        try:
            if name == "standard":
                from cuml.preprocessing import StandardScaler
            else:
                from cuml.preprocessing import RobustScaler as StandardScaler
            return StandardScaler(), "cuml"
        except Exception as exc:
            log.debug("cuML scaler unavailable (%s); using sklearn", exc)
        from sklearn.preprocessing import StandardScaler, RobustScaler
        return (StandardScaler() if name == "standard" else RobustScaler()), "sklearn"
    raise ValueError(f"Unknown scaler {name!r}")


def _build_pca(n_components: Any) -> Tuple[Any, str]:
    """Return (pca_object, backend_label). cuML first, sklearn fallback."""
    try:
        from cuml.decomposition import PCA as cuPCA
        return cuPCA(n_components=n_components), "cuml"
    except Exception as exc:
        log.debug("cuML PCA unavailable (%s); using sklearn", exc)
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, svd_solver="full", random_state=0), "sklearn"


@dataclass
class FeaturePreprocessor:
    cfg: PreprocessingConfig
    _scaler: Any = field(default=None, init=False, repr=False)
    _scaler_backend: str = field(default="none", init=False)
    _pca: Any = field(default=None, init=False, repr=False)
    _pca_backend: str = field(default="none", init=False)
    _selector: Any = field(default=None, init=False, repr=False)
    _impute_values: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)
    n_features_in_: int = field(default=0, init=False)
    n_features_out_: int = field(default=0, init=False)

    def _impute(self, x: np.ndarray, fitting: bool) -> np.ndarray:
        """Impute non-finite values with column medians (computed on train only)."""
        if not self.cfg.impute_nonfinite:
            return x
        x = np.array(x, dtype=np.float32, copy=True)
        if fitting:
            self._impute_values = np.empty(x.shape[1], dtype=np.float32)
            for first in range(0, x.shape[1], 32):
                block = x[:, first:first + 32]
                observed = np.isfinite(block).any(axis=0)
                med = np.zeros(block.shape[1], dtype=np.float32)
                if observed.any():
                    available = block[:, observed]
                    safe = np.where(np.isfinite(available), available, np.nan)
                    med[observed] = np.nanmedian(safe, axis=0, overwrite_input=True)
                self._impute_values[first:first + 32] = np.where(np.isfinite(med), med, 0)
        for first in range(0, x.shape[1], 32):
            block = x[:, first:first + 32]
            rows, columns = np.where(~np.isfinite(block))
            block[rows, columns] = self._impute_values[first + columns]
        return x

    def fit(
        self, x_train: np.ndarray, y_train: Optional[np.ndarray] = None
    ) -> "FeaturePreprocessor":
        """Fit ALL stages on training data only. ``y_train`` needed for selection."""
        x = self._impute(x_train, fitting=True)
        self.n_features_in_ = x.shape[1]

        # Use float32 from here on: cuML requires it and it's the downstream dtype.
        x = x.astype(np.float32)

        self._scaler, self._scaler_backend = _build_scaler(self.cfg.scaler)
        if self._scaler is not None:
            self._scaler.fit(x)
            if self.cfg.feature_selection_k or self.cfg.use_pca:
                x = _to_numpy(self._scaler.transform(x)).astype(np.float32)

        if self.cfg.feature_selection_k:
            if y_train is None:
                raise ValueError("feature_selection_k requires y_train")
            from sklearn.feature_selection import SelectKBest, f_classif
            k = min(self.cfg.feature_selection_k, x.shape[1])
            self._selector = SelectKBest(f_classif, k=k)
            x = self._selector.fit_transform(x, y_train)

        if self.cfg.use_pca:
            self._pca, self._pca_backend = _build_pca(self.cfg.pca_components)
            x = _to_numpy(self._pca.fit_transform(x)).astype(np.float32)

        self.n_features_out_ = x.shape[1]
        self._fitted = True
        log.info(
            "Preprocessor fitted on TRAIN ONLY: %d -> %d features "
            "(scaler=%s[%s] select_k=%s pca=%s[%s])",
            self.n_features_in_, self.n_features_out_,
            self.cfg.scaler, self._scaler_backend,
            self.cfg.feature_selection_k,
            self.cfg.use_pca, self._pca_backend,
        )
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeaturePreprocessor.transform called before fit")
        x = self._impute(x, fitting=False)
        x = x.astype(np.float32)
        if self._scaler is not None:
            x = _to_numpy(self._scaler.transform(x)).astype(np.float32)
        if self._selector is not None:
            x = self._selector.transform(x)
        if self._pca is not None:
            x = _to_numpy(self._pca.transform(x)).astype(np.float32)
        return x.astype(np.float32, copy=False)

    def fit_transform(
        self, x_train: np.ndarray, y_train: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.fit(x_train, y_train).transform(x_train)
