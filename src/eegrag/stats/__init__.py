"""Statistical testing and result aggregation for the journal tables."""
from .tests import (
    friedman_test,
    nemenyi_posthoc,
    wilcoxon_signed_rank,
    bootstrap_ci,
    paired_comparison,
)
from .aggregate import mean_std, summarize_folds, format_mean_std

__all__ = [
    "friedman_test",
    "nemenyi_posthoc",
    "wilcoxon_signed_rank",
    "bootstrap_ci",
    "paired_comparison",
    "mean_std",
    "summarize_folds",
    "format_mean_std",
]
