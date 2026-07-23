"""Physiology-preserving EEG-feature augmentations.

These operate on the *feature* representation (per-window channel x feature-type
vectors), using a :class:`FeatureLayout` so that augmentations respect EEG
topology:

  * **Gaussian noise** -- additive jitter in (standardized) feature units.
  * **Amplitude scaling** -- mild global gain change, U(1-d, 1+d).
  * **Channel dropout / electrode masking** -- zero ALL features of a few
    randomly chosen electrodes (== "random masking of several electrodes").
  * **Frequency masking** -- zero a whole named band-power group (Delta/Theta/
    Alpha/Beta) across channels. This is a *band* mask, NOT a learned/arbitrary
    cutoff-frequency shift, so it stays physiologically interpretable.
  * **Temporal masking** -- only when samples are window *sequences*
    (sequence_len > 1): zero a contiguous span of timesteps.

EXPLICITLY EXCLUDED (would break topology): image-style random spatial flips,
arbitrary cropping, and arbitrary cutoff-frequency shifting.

After standardization the feature mean is ~0, so "zeroing" a masked group is
equivalent to mean-imputation -- a principled neutral value.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import AugmentationConfig
from ..data.feature_layout import FeatureLayout


class FeatureAugmenter:
    """Stochastic augmentation chain over feature vectors / sequences.

    Input ``x`` shape: ``[n_features]`` (single window) or
    ``[seq_len, n_features]`` (window sequence). Output has the same shape.
    """

    def __init__(self, cfg: AugmentationConfig, layout: FeatureLayout):
        self.cfg = cfg
        self.layout = layout
        # Precompute per-channel index arrays and band groups once.
        self._channel_index_arrays = layout.channel_indices_array()
        self._band_groups = list(layout.band_feature_indices().items())

    # --- individual ops --------------------------------------------------- #
    def _gaussian_noise(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.cfg.gaussian_noise_p:
            return x
        return x + rng.normal(0.0, self.cfg.gaussian_noise_std, size=x.shape).astype(
            x.dtype
        )

    def _amplitude_scale(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.cfg.amplitude_scale_p:
            return x
        d = self.cfg.amplitude_scale_delta
        gain = rng.uniform(1.0 - d, 1.0 + d)
        return (x * gain).astype(x.dtype)

    def _channel_dropout(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.cfg.channel_dropout_p or not self._channel_index_arrays:
            return x
        n_chan = len(self._channel_index_arrays)
        max_drop = min(self.cfg.channel_dropout_max, max(n_chan - 1, 1))
        if max_drop < 1:
            return x
        n_drop = rng.integers(1, max_drop + 1)
        chosen = rng.choice(n_chan, size=int(n_drop), replace=False)
        x = x.copy()
        for c in chosen:
            x[..., self._channel_index_arrays[c]] = 0.0
        return x

    def _freq_masking(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.cfg.freq_masking_p or not self._band_groups:
            return x
        max_bands = min(self.cfg.freq_masking_bands_max, len(self._band_groups))
        if max_bands < 1:
            return x
        n_bands = rng.integers(1, max_bands + 1)
        chosen = rng.choice(len(self._band_groups), size=int(n_bands), replace=False)
        x = x.copy()
        for b in chosen:
            _, idx = self._band_groups[b]
            x[..., idx] = 0.0
        return x

    def _temporal_masking(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Only meaningful for sequence inputs [seq_len, n_features].
        if x.ndim < 2 or x.shape[0] <= 1:
            return x
        if rng.random() >= self.cfg.temporal_mask_p:
            return x
        seq_len = x.shape[0]
        max_span = max(1, int(round(self.cfg.temporal_mask_max_frac * seq_len)))
        span = int(rng.integers(1, max_span + 1))
        start = int(rng.integers(0, seq_len - span + 1))
        x = x.copy()
        x[start:start + span] = 0.0
        return x

    # --- public ----------------------------------------------------------- #
    def augment(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply the full chain once, returning one augmented view."""
        if not self.cfg.enabled:
            return x
        x = np.asarray(x)
        x = self._gaussian_noise(x, rng)
        x = self._amplitude_scale(x, rng)
        x = self._channel_dropout(x, rng)
        x = self._freq_masking(x, rng)
        x = self._temporal_masking(x, rng)
        return x

    def views(
        self, x: np.ndarray, rng: np.random.Generator, n_views: Optional[int] = None
    ) -> List[np.ndarray]:
        """Return ``n_views`` independently augmented views of ``x``."""
        k = n_views or self.cfg.n_views
        return [self.augment(x, rng) for _ in range(k)]


def build_augmenter(cfg: AugmentationConfig, layout: FeatureLayout) -> FeatureAugmenter:
    return FeatureAugmenter(cfg, layout)
