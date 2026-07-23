"""PyTorch datasets for supervised and contrastive training.

Two datasets:
  * :class:`EEGWindowDataset` -- returns ``(x, y, subject_code)`` for plain
    supervised use and for embedding extraction (no augmentation).
  * :class:`ContrastiveViewDataset` -- returns ``n_views`` augmented views per
    anchor for SupCon-style training, plus the label and subject code.

Both accept *already-transformed* feature arrays (output of the leakage-safe
preprocessor) so the dataset never touches raw subjects it should not see.

Optional ``sequence_len`` stacks consecutive windows *within the same
recording* into a [seq_len, n_features] sample, enabling temporal masking and
TCN/Transformer encoders. Sequences never cross recording boundaries.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, WeightedRandomSampler
    _TORCH = True
except ImportError:  # allow importing the package without torch installed
    Dataset = object  # type: ignore
    _TORCH = False

from ..data.feature_layout import FeatureLayout
from .augment import FeatureAugmenter


def _build_sequence_index(
    n: int,
    recordings: Optional[np.ndarray],
    seq_len: int,
) -> np.ndarray:
    """Return start indices of valid length-``seq_len`` windows that stay inside
    a single recording. For ``seq_len == 1`` this is just ``arange(n)``."""
    if seq_len <= 1:
        return np.arange(n, dtype=np.int64)
    starts: List[int] = []
    if recordings is None:
        for s in range(0, n - seq_len + 1):
            starts.append(s)
        return np.asarray(starts, dtype=np.int64)
    # group by contiguous identical recording id (rows are time-ordered)
    rec = recordings
    i = 0
    while i < n:
        j = i
        while j < n and rec[j] == rec[i]:
            j += 1
        # window [i, j) is one recording
        for s in range(i, j - seq_len + 1):
            starts.append(s)
        i = j
    return np.asarray(starts, dtype=np.int64)


class EEGWindowDataset(Dataset):
    """Plain (x, y, subject) dataset. No augmentation; used for embeddings."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        subjects: Sequence[str],
        layout: FeatureLayout,
        *,
        recordings: Optional[np.ndarray] = None,
        sequence_len: int = 1,
        as_channel_grid: bool = False,
    ):
        if not _TORCH:
            raise ImportError("PyTorch is required for EEGWindowDataset")
        self.features = np.ascontiguousarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.subjects = np.asarray(subjects, dtype=object)
        self.layout = layout
        self.sequence_len = max(1, sequence_len)
        self.as_channel_grid = as_channel_grid
        self.recordings = recordings
        self._starts = _build_sequence_index(
            len(self.labels), recordings, self.sequence_len
        )
        # stable integer codes for subjects (for tensorizable subject ids)
        uniq = list(dict.fromkeys(self.subjects.tolist()))
        self._subj_to_code = {s: i for i, s in enumerate(uniq)}
        self.subject_codes = np.asarray(
            [self._subj_to_code[s] for s in self.subjects], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self._starts)

    def _gather(self, start: int) -> np.ndarray:
        if self.sequence_len == 1:
            x = self.features[start]
        else:
            x = self.features[start:start + self.sequence_len]
        return x

    def _label_for(self, start: int) -> int:
        if self.sequence_len == 1:
            return int(self.labels[start])
        # sequence label = positive if any window in the span is a seizure
        return int(self.labels[start:start + self.sequence_len].max())

    def _format(self, x: np.ndarray):
        if self.as_channel_grid and self.layout.is_regular:
            x = self.layout.reshape_to_channels(x)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

    def __getitem__(self, i: int):
        start = int(self._starts[i])
        x = self._gather(start)
        y = self._label_for(start)
        subj = int(self.subject_codes[start])
        return self._format(x), y, subj

    @property
    def start_labels(self) -> np.ndarray:
        """Labels aligned to ``__len__`` (per sequence start) -- for samplers."""
        if self.sequence_len == 1:
            return self.labels[self._starts]
        return np.asarray(
            [self._label_for(int(s)) for s in self._starts], dtype=np.int64
        )


class ContrastiveViewDataset(EEGWindowDataset):
    """Returns ``n_views`` augmented views per anchor for contrastive training.

    ``__getitem__`` -> (views_tensor [n_views, ...], label, subject_code).
    """

    def __init__(self, *args, augmenter: FeatureAugmenter, n_views: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmenter = augmenter
        self.n_views = n_views

    def __getitem__(self, i: int):
        start = int(self._starts[i])
        x = self._gather(start)
        y = self._label_for(start)
        subj = int(self.subject_codes[start])
        # Per-item RNG keyed on the index keeps augmentation reproducible per
        # epoch when combined with worker_init_fn / torch seeding.
        rng = np.random.default_rng(abs(hash((i, int(self.labels[start])))) % (2**32))
        views = self.augmenter.views(x, rng, self.n_views)
        view_tensors = torch.stack([self._format(v) for v in views], dim=0)
        return view_tensors, y, subj


def make_balanced_sampler(
    labels: np.ndarray, minority_class: int = 1, pos_boost: float = 1.0
):
    """WeightedRandomSampler that up-samples the minority (seizure) class.

    ``pos_boost`` multiplies the inverse-frequency weight of the minority class
    so seizure anchors reliably appear in contrastive batches.
    """
    if not _TORCH:
        raise ImportError("PyTorch is required for make_balanced_sampler")
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    freq = {int(c): int(n) for c, n in zip(classes, counts)}
    inv = {c: 1.0 / max(n, 1) for c, n in freq.items()}
    if minority_class in inv:
        inv[minority_class] *= pos_boost
    weights = np.asarray([inv[int(y)] for y in labels], dtype=np.float64)
    weights = weights / weights.sum()
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )
