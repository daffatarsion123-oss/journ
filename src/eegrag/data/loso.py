"""Leave-One-Subject-Out (LOSO) split with hard anti-leakage assertions.

The single most important methodological constraint of this project:

    >>> Test-subject windows/embeddings must NEVER appear in training,
    >>> in the preprocessing fit, OR in the retrieval memory bank.

This module produces index-based folds and provides assertions that other
stages call to prove (at runtime) that the constraint holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .loading import EEGFeatureBundle


@dataclass
class LOSOFold:
    """One LOSO fold: exactly one held-out test subject."""

    fold_index: int
    test_subject: str
    train_subjects: List[str]
    train_idx: np.ndarray      # row indices into the bundle (training subjects)
    test_idx: np.ndarray       # row indices into the bundle (test subject)

    def __post_init__(self) -> None:
        assert self.test_subject not in self.train_subjects, (
            f"test subject {self.test_subject} leaked into train_subjects"
        )
        # No row index may appear in both partitions.
        overlap = np.intersect1d(self.train_idx, self.test_idx)
        assert overlap.size == 0, (
            f"fold {self.fold_index}: {overlap.size} row indices in BOTH "
            f"train and test partitions"
        )

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def n_test(self) -> int:
        return int(self.test_idx.size)


def make_loso_folds(
    bundle: EEGFeatureBundle,
    subjects: Optional[Sequence[str]] = None,
    folds_whitelist: Optional[Sequence[str]] = None,
) -> List[LOSOFold]:
    """Build one fold per subject (held out as test).

    Parameters
    ----------
    subjects:
        Restrict the universe of subjects considered (default: all in bundle).
    folds_whitelist:
        Only *generate* folds for these test subjects (training still uses all
        other subjects in ``subjects``). Useful for smoke tests / partial runs.
    """
    all_subjects = list(subjects) if subjects is not None else list(bundle.subject_order)
    all_subjects = [s for s in all_subjects if s in set(bundle.subjects.tolist())]
    if len(all_subjects) < 2:
        raise ValueError(
            f"LOSO needs >= 2 subjects, found {len(all_subjects)}: {all_subjects}"
        )

    test_subjects = (
        [s for s in all_subjects if s in set(folds_whitelist)]
        if folds_whitelist
        else all_subjects
    )
    if not test_subjects:
        raise ValueError(
            f"folds_whitelist {list(folds_whitelist)} matched no subjects."
        )

    # Precompute subject -> row indices once.
    subject_to_rows = {
        s: np.flatnonzero(bundle.subjects == s) for s in all_subjects
    }

    folds: List[LOSOFold] = []
    for i, test_subject in enumerate(test_subjects):
        train_subjects = [s for s in all_subjects if s != test_subject]
        train_idx = np.concatenate([subject_to_rows[s] for s in train_subjects])
        train_idx.sort()
        test_idx = subject_to_rows[test_subject]
        folds.append(
            LOSOFold(
                fold_index=i,
                test_subject=test_subject,
                train_subjects=train_subjects,
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )
    return folds


def assert_no_subject_leakage(
    bundle: EEGFeatureBundle,
    fold: LOSOFold,
    *,
    bank_idx: Optional[np.ndarray] = None,
    extra_train_idx: Optional[np.ndarray] = None,
    context: str = "",
) -> None:
    """Runtime guard: the test subject must not appear anywhere in training.

    Call this from every stage that builds something from training data
    (preprocessing fit, encoder training set, memory bank, threshold tuning).

    Parameters
    ----------
    bank_idx:
        Row indices selected into the retrieval memory bank, if any.
    extra_train_idx:
        Any additional indices treated as "training" (e.g. a subsampled set).
    """
    prefix = f"[leakage:{context or fold.test_subject}] "
    test_subject = fold.test_subject

    # 1) The declared train indices must not contain the test subject.
    train_subjects = set(bundle.subjects[fold.train_idx].tolist())
    assert test_subject not in train_subjects, (
        prefix + f"test subject {test_subject} present in fold.train_idx"
    )

    # 2) Train/test index sets disjoint.
    overlap = np.intersect1d(fold.train_idx, fold.test_idx)
    assert overlap.size == 0, prefix + f"{overlap.size} indices shared train/test"

    # 3) Any explicitly provided training-derived index set must exclude the
    #    test subject entirely.
    for name, idx in (("bank_idx", bank_idx), ("extra_train_idx", extra_train_idx)):
        if idx is None:
            continue
        idx = np.asarray(idx)
        if idx.size == 0:
            continue
        subs = set(bundle.subjects[idx].tolist())
        assert test_subject not in subs, (
            prefix + f"test subject {test_subject} leaked into {name}"
        )
        # And every such index must be a training-partition index.
        not_in_train = np.setdiff1d(idx, fold.train_idx)
        assert not_in_train.size == 0, (
            prefix
            + f"{not_in_train.size} {name} indices are not in the training partition"
        )
