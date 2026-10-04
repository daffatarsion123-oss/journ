"""Train-only CuPy imputation and scaling without host feature copies."""


class CuPyFeaturePreprocessor:
    def __init__(self, cfg):
        if cfg.use_pca or cfg.feature_selection_k:
            raise ValueError("Resident CuPy preprocessing currently supports imputation/scaling only")
        if cfg.scaler not in {"standard", "robust", "none"}:
            raise ValueError(f"Unknown scaler {cfg.scaler!r}")
        self.cfg = cfg
        self._fitted = False

    def _impute(self, x, fitting=False):
        import cupy as cp
        x = cp.array(x, dtype=cp.float32, copy=True)
        if not self.cfg.impute_nonfinite:
            return x
        if fitting:
            medians = []
            for first in range(0, x.shape[1], 32):
                block = x[:, first:first + 32]
                safe = cp.where(cp.isfinite(block), block, cp.nan)
                median = cp.nanmedian(safe, axis=0)
                medians.append(cp.where(cp.isfinite(median), median, 0))
            self._impute_values = cp.concatenate(medians)
        for first in range(0, len(x), 65536):
            block = x[first:first + 65536]
            block[:] = cp.where(cp.isfinite(block), block, self._impute_values)
        return x

    def fit_transform(self, x, y=None):
        import cupy as cp
        x = self._impute(x, fitting=True)
        self.n_features_in_ = self.n_features_out_ = x.shape[1]
        if self.cfg.scaler == "standard":
            self.center = x.mean(axis=0, dtype=cp.float64)
            self.scale = cp.sqrt(x.var(axis=0, dtype=cp.float64))
        elif self.cfg.scaler == "robust":
            quantiles = []
            for first in range(0, x.shape[1], 32):
                quantiles.append(cp.percentile(x[:, first:first + 32], [25, 50, 75], axis=0))
            q = cp.concatenate(quantiles, axis=1)
            self.center, self.scale = q[1], q[2] - q[0]
        else:
            self.center = cp.zeros(x.shape[1], dtype=cp.float32)
            self.scale = cp.ones(x.shape[1], dtype=cp.float32)
        # Match sklearn's near-constant StandardScaler detection.
        if self.cfg.scaler == "standard":
            eps = cp.finfo(cp.float64).eps
            variance = self.scale ** 2
            constant = variance <= len(x) * eps * variance + (len(x) * self.center * eps) ** 2
        else:
            constant = self.scale < 10 * cp.finfo(self.scale.dtype).eps
        self.scale = cp.where(constant, 1, self.scale)
        self._fitted = True
        return self._scale(x)

    def _scale(self, x):
        if self.cfg.scaler != "none":
            x -= self.center
            x /= self.scale
        return x

    def fit(self, x, y=None):
        self.fit_transform(x, y)
        return self

    def transform(self, x):
        if not self._fitted:
            raise RuntimeError("CuPyFeaturePreprocessor.transform called before fit")
        return self._scale(self._impute(x))
