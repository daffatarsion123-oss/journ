"""Assemble leakage-safe per-fold arrays.

``prepare_fold`` is the single entry point used by every runner. It:
  1. slices the bundle into train/test by the fold's row indices,
  2. asserts the test subject does not appear in the training partition,
  3. fits the preprocessor on TRAIN ONLY, then transforms train and test,
  4. returns a tidy :class:`FoldArrays` container.

By funneling all runners through here, the leakage guard is enforced in exactly
one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import PreprocessingConfig
from ..data.loading import EEGFeatureBundle
from ..data.loso import LOSOFold, assert_no_subject_leakage
from ..data.scaling import FeaturePreprocessor


@dataclass
class FoldArrays:
    """Transformed, leakage-safe arrays for one LOSO fold."""

    fold: LOSOFold
    x_train: np.ndarray            # [n_train, F'] transformed features
    y_train: np.ndarray            # [n_train]
    subjects_train: np.ndarray     # [n_train] subject id per row
    x_test: np.ndarray             # [n_test, F'] transformed features
    y_test: np.ndarray             # [n_test]
    window_start_test: np.ndarray  # [n_test] seconds (for FP/hour)
    window_end_test: np.ndarray
    preprocessor: FeaturePreprocessor
    # Mapping from local train-row position -> global bundle row index, so the
    # retrieval stage can subsample the bank yet still pass global indices to the
    # leakage assertion.
    train_global_idx: np.ndarray
    test_global_idx: np.ndarray

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]


def fit_fold_preprocessor(
    bundle: EEGFeatureBundle, fold: LOSOFold, cfg: PreprocessingConfig
) -> FeaturePreprocessor:
    """Fit a preprocessor on the fold's TRAINING rows only (with leakage guard)."""
    assert_no_subject_leakage(bundle, fold, context="preprocessor-fit")
    x_train = bundle.features[fold.train_idx]
    y_train = bundle.labels[fold.train_idx]
    return FeaturePreprocessor(cfg).fit(x_train, y_train)


def prepare_fold(
    bundle: EEGFeatureBundle,
    fold: LOSOFold,
    cfg: PreprocessingConfig,
    preprocessor: Optional[FeaturePreprocessor] = None,
) -> FoldArrays:
    """Produce transformed train/test arrays for a fold (leakage-safe)."""
    assert_no_subject_leakage(bundle, fold, context="prepare-fold")

    if preprocessor is None:
        preprocessor = fit_fold_preprocessor(bundle, fold, cfg)

    x_train = preprocessor.transform(bundle.features[fold.train_idx])
    x_test = preprocessor.transform(bundle.features[fold.test_idx])

    return FoldArrays(
        fold=fold,
        x_train=x_train,
        y_train=bundle.labels[fold.train_idx].astype(np.int64),
        subjects_train=bundle.subjects[fold.train_idx],
        x_test=x_test,
        y_test=bundle.labels[fold.test_idx].astype(np.int64),
        window_start_test=bundle.window_start[fold.test_idx],
        window_end_test=bundle.window_end[fold.test_idx],
        preprocessor=preprocessor,
        train_global_idx=fold.train_idx.copy(),
        test_global_idx=fold.test_idx.copy(),
    )
