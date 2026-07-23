"""Parse the CHB-MIT feature column schema into a topology-aware layout.

The previous-paper extractor emits, per 2 s window, one column per
``<channel><FeatureType>`` pair, e.g. ``FP1-F7Mean``, ``FP1-F7DeltaPower`` ...
with 23 bipolar channels x 8 feature types = 184 features.

``FeatureLayout`` recovers the (channel, feature_type) grouping from the column
names so that:

  * the encoder can reshape the flat vector into ``[n_channels, n_feat_per_chan]``
    and convolve along the electrode axis (respecting montage topology);
  * augmentations can drop *whole electrodes* (channel dropout) or mask *whole
    band groups* (frequency masking) instead of arbitrary spatial flips.

The parser is schema-driven (it reads the actual columns), so it adapts if the
feature set changes -- it does not hardcode the 23x8 layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Known feature-type suffixes, longest-first so "DeltaPower" matches before any
# hypothetical "Delta". Band-power suffixes are flagged as frequency groups.
_FEATURE_SUFFIXES: Tuple[str, ...] = (
    "DeltaPower",
    "ThetaPower",
    "AlphaPower",
    "BetaPower",
    "GammaPower",   # tolerated if a future extractor adds it
    "Variance",
    "Entropy",
    "Mean",
    "Std",
)
_BAND_SUFFIXES = {"DeltaPower", "ThetaPower", "AlphaPower", "BetaPower", "GammaPower"}


@dataclass
class FeatureLayout:
    """Topology metadata derived from the ordered feature-column names."""

    feature_columns: List[str]
    # channel -> list of column indices (into the feature matrix) for that channel
    channel_to_indices: Dict[str, List[int]] = field(default_factory=dict)
    # feature_type -> list of column indices for that feature type across channels
    ftype_to_indices: Dict[str, List[int]] = field(default_factory=dict)
    channels: List[str] = field(default_factory=list)
    feature_types: List[str] = field(default_factory=list)
    # column index -> (channel, feature_type)
    column_meta: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_feat_per_channel(self) -> int:
        return len(self.feature_types)

    @property
    def is_regular(self) -> bool:
        """True iff every channel exposes the same set of feature types, so the
        flat vector can be reshaped to a dense [n_channels, n_feat_per_chan]."""
        return self.n_channels * self.n_feat_per_channel == self.n_features

    @classmethod
    def from_columns(cls, columns: Sequence[str]) -> "FeatureLayout":
        feature_columns = list(columns)
        channel_to_indices: Dict[str, List[int]] = {}
        ftype_to_indices: Dict[str, List[int]] = {}
        channels: List[str] = []
        feature_types: List[str] = []
        column_meta: List[Tuple[str, str]] = []

        for idx, col in enumerate(feature_columns):
            channel, ftype = cls._split_column(col)
            column_meta.append((channel, ftype))
            if channel not in channel_to_indices:
                channel_to_indices[channel] = []
                channels.append(channel)
            channel_to_indices[channel].append(idx)
            if ftype not in ftype_to_indices:
                ftype_to_indices[ftype] = []
                feature_types.append(ftype)
            ftype_to_indices[ftype].append(idx)

        return cls(
            feature_columns=feature_columns,
            channel_to_indices=channel_to_indices,
            ftype_to_indices=ftype_to_indices,
            channels=channels,
            feature_types=feature_types,
            column_meta=column_meta,
        )

    @staticmethod
    def _split_column(col: str) -> Tuple[str, str]:
        for suffix in _FEATURE_SUFFIXES:
            if col.endswith(suffix) and len(col) > len(suffix):
                return col[: -len(suffix)], suffix
        # Unknown layout -> treat the whole name as its own "channel" with a
        # generic feature type so nothing crashes; topology ops become no-ops.
        return col, "Value"

    def band_feature_indices(self) -> Dict[str, List[int]]:
        """Column indices grouped by band-power feature type only."""
        return {
            ft: idx
            for ft, idx in self.ftype_to_indices.items()
            if ft in _BAND_SUFFIXES
        }

    def channel_indices_array(self) -> List[np.ndarray]:
        """Per-channel index arrays, ordered like ``self.channels``."""
        return [np.asarray(self.channel_to_indices[c], dtype=np.int64)
                for c in self.channels]

    def reshape_to_channels(self, x: np.ndarray) -> np.ndarray:
        """[..., n_features] -> [..., n_channels, n_feat_per_channel].

        Only valid when ``is_regular``. The per-channel column order is assumed
        consistent across channels (true for the CHB-MIT extractor).
        """
        if not self.is_regular:
            raise ValueError(
                "Irregular feature layout cannot be reshaped to a dense grid."
            )
        lead = x.shape[:-1]
        order = np.concatenate(self.channel_indices_array())
        x = x[..., order]
        return x.reshape(*lead, self.n_channels, self.n_feat_per_channel)

    def describe(self) -> str:
        return (
            f"FeatureLayout(n_features={self.n_features}, "
            f"n_channels={self.n_channels}, "
            f"n_feat_per_channel={self.n_feat_per_channel}, "
            f"regular={self.is_regular})"
        )
