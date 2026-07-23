#!/usr/bin/env python
"""Negative tests proving the anti-leakage guards actually fire.

These are the most important tests in the repo: they assert that the protocol
*rejects* a test subject leaking into the memory bank or the training index.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

from eegrag.data.loso import LOSOFold, assert_no_subject_leakage  # noqa: E402
from eegrag.retrieval import build_memory_bank  # noqa: E402


class _Bundle:
    """Minimal bundle stub exposing only ``subjects``."""

    subjects = np.array(["chb01"] * 5 + ["chb02"] * 5, dtype=object)


def _fold():
    return LOSOFold(
        fold_index=0, test_subject="chb02", train_subjects=["chb01"],
        train_idx=np.array([0, 1, 2, 3, 4]), test_idx=np.array([5, 6, 7, 8, 9]),
    )


def test_memory_bank_rejects_test_subject():
    emb = np.random.rand(10, 8).astype("float32")
    subs = _Bundle.subjects
    labs = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    with pytest.raises(AssertionError):
        build_memory_bank(emb, labs, subs, test_subject="chb02")


def test_memory_bank_accepts_clean_bank():
    emb = np.random.rand(5, 8).astype("float32")
    subs = np.array(["chb01"] * 5, dtype=object)
    labs = np.array([0, 1, 0, 1, 0])
    bank = build_memory_bank(emb, labs, subs, test_subject="chb02")
    assert bank.size == 5


def test_assert_no_subject_leakage_fires_on_test_subject_in_bank():
    with pytest.raises(AssertionError):
        assert_no_subject_leakage(
            _Bundle(), _fold(), bank_idx=np.array([0, 1, 9]), context="neg"
        )


def test_assert_no_subject_leakage_fires_on_nontrain_index():
    # index 9 is a test-partition row -> must be rejected as not-in-train
    with pytest.raises(AssertionError):
        assert_no_subject_leakage(
            _Bundle(), _fold(), extra_train_idx=np.array([9]), context="neg"
        )


def test_assert_no_subject_leakage_accepts_clean():
    assert_no_subject_leakage(
        _Bundle(), _fold(), bank_idx=np.array([0, 1, 2]), context="ok"
    )


def test_loso_fold_rejects_overlapping_partitions():
    with pytest.raises(AssertionError):
        LOSOFold(0, "chb02", ["chb01"], np.array([0, 1, 5]), np.array([5, 6]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
