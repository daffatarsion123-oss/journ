"""Train-only preprocessing orchestration for a LOSO fold."""
from .pipeline import fit_fold_preprocessor, FoldArrays, prepare_fold

# Re-export the underlying preprocessor for convenience.
from ..data.scaling import FeaturePreprocessor

__all__ = [
    "fit_fold_preprocessor",
    "FoldArrays",
    "prepare_fold",
    "FeaturePreprocessor",
]
