"""Data layer: parquet loading, subject parsing, FeatureLayout, LOSO splits."""
from .feature_layout import FeatureLayout
from .loading import (
    subject_id_from_filename,
    discover_recordings,
    load_subject_frames,
    SubjectData,
    assemble_dataset,
    EEGFeatureBundle,
)
from .loso import (
    LOSOFold,
    make_loso_folds,
    assert_no_subject_leakage,
)
from .scaling import FeaturePreprocessor

__all__ = [
    "FeatureLayout",
    "subject_id_from_filename",
    "discover_recordings",
    "load_subject_frames",
    "SubjectData",
    "assemble_dataset",
    "EEGFeatureBundle",
    "LOSOFold",
    "make_loso_folds",
    "assert_no_subject_leakage",
    "FeaturePreprocessor",
]
