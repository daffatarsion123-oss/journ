"""Significance tests for model comparison.

Protocol (matches the methodology constraints):
  * >2 models  -> :func:`friedman_test` (omnibus) then :func:`nemenyi_posthoc`.
  * 2 models   -> :func:`wilcoxon_signed_rank` with effect size ``r`` and a
    :func:`bootstrap_ci` on the paired metric difference.

Inputs are per-fold (or per-seed-x-fold) metric vectors, one entry per matched
observation, aligned across models.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Omnibus: Friedman
# --------------------------------------------------------------------------- #
@dataclass
class FriedmanResult:
    statistic: float
    p_value: float
    n_models: int
    n_blocks: int
    avg_ranks: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


def _rank_matrix(scores: np.ndarray) -> np.ndarray:
    """Rank models within each block (row). Higher score -> better -> rank 1."""
    from scipy.stats import rankdata

    ranks = np.empty_like(scores, dtype=np.float64)
    for i in range(scores.shape[0]):
        ranks[i] = rankdata(-scores[i])     # negate so larger value = rank 1
    return ranks


def friedman_test(
    per_model_scores: Dict[str, Sequence[float]],
) -> FriedmanResult:
    """Friedman omnibus test across >=3 models on matched blocks (folds/seeds).

    ``per_model_scores`` maps model name -> equal-length metric vector.
    """
    from scipy.stats import friedmanchisquare

    names = list(per_model_scores)
    if len(names) < 3:
        raise ValueError("Friedman test requires >= 3 models")
    mat = np.asarray([per_model_scores[n] for n in names], dtype=np.float64)  # [M, B]
    lengths = {len(v) for v in per_model_scores.values()}
    if len(lengths) != 1:
        raise ValueError(f"All models need matched blocks; got lengths {lengths}")

    stat, p = friedmanchisquare(*[mat[i] for i in range(mat.shape[0])])
    ranks = _rank_matrix(mat.T)             # blocks x models
    avg_ranks = {n: float(ranks[:, j].mean()) for j, n in enumerate(names)}
    return FriedmanResult(
        statistic=float(stat),
        p_value=float(p),
        n_models=len(names),
        n_blocks=mat.shape[1],
        avg_ranks=avg_ranks,
    )


# --------------------------------------------------------------------------- #
# Post-hoc: Nemenyi
# --------------------------------------------------------------------------- #
@dataclass
class NemenyiResult:
    avg_ranks: Dict[str, float]
    p_values: Dict[str, Dict[str, float]]   # pairwise
    critical_difference: float
    alpha: float

    def to_dict(self) -> Dict:
        return {
            "avg_ranks": self.avg_ranks,
            "p_values": self.p_values,
            "critical_difference": self.critical_difference,
            "alpha": self.alpha,
        }


def nemenyi_posthoc(
    per_model_scores: Dict[str, Sequence[float]], alpha: float = 0.05
) -> NemenyiResult:
    """Nemenyi post-hoc pairwise test (run after a significant Friedman).

    Uses the studentized-range distribution for pairwise p-values and reports
    the critical difference (CD) for a Demsar-style CD diagram.
    """
    from scipy.stats import studentized_range

    names = list(per_model_scores)
    k = len(names)
    mat = np.asarray([per_model_scores[n] for n in names], dtype=np.float64)
    n_blocks = mat.shape[1]
    ranks = _rank_matrix(mat.T)
    avg_ranks = {n: float(ranks[:, j].mean()) for j, n in enumerate(names)}

    se = np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    p_values: Dict[str, Dict[str, float]] = {n: {} for n in names}
    for i in range(k):
        for j in range(k):
            if i == j:
                p_values[names[i]][names[j]] = 1.0
                continue
            diff = abs(avg_ranks[names[i]] - avg_ranks[names[j]])
            q = diff / se
            # studentized range expects q on the sqrt(2)-scaled axis
            p = float(studentized_range.sf(q * np.sqrt(2.0), k, np.inf))
            p_values[names[i]][names[j]] = min(1.0, p)

    q_alpha = float(studentized_range.ppf(1 - alpha, k, np.inf)) / np.sqrt(2.0)
    cd = q_alpha * se
    return NemenyiResult(
        avg_ranks=avg_ranks, p_values=p_values, critical_difference=cd, alpha=alpha
    )


# --------------------------------------------------------------------------- #
# Pairwise: Wilcoxon signed-rank + effect size r + bootstrap CI
# --------------------------------------------------------------------------- #
@dataclass
class WilcoxonResult:
    statistic: float
    p_value: float
    z: float
    effect_size_r: float
    n: int
    n_nonzero: int
    median_diff: float
    alternative: str

    def to_dict(self) -> Dict:
        return asdict(self)


def wilcoxon_signed_rank(
    a: Sequence[float], b: Sequence[float], alternative: str = "two-sided"
) -> WilcoxonResult:
    """Wilcoxon signed-rank test on paired metrics ``a`` vs ``b``.

    Effect size r = |Z| / sqrt(N) where N is the number of non-zero-difference
    pairs (matched-pairs rank-biserial style normal-approximation).
    """
    from scipy.stats import wilcoxon, norm

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired samples must have equal length")
    diff = a - b
    nonzero = diff[diff != 0]
    n_nz = len(nonzero)
    if n_nz == 0:
        return WilcoxonResult(0.0, 1.0, 0.0, 0.0, len(a), 0, 0.0, alternative)

    try:
        res = wilcoxon(a, b, alternative=alternative, method="approx",
                       zero_method="wilcox")
        stat = float(res.statistic)
        p = float(res.pvalue)
        z = float(getattr(res, "zstatistic", np.nan))
    except TypeError:  # older scipy without method=
        stat, p = wilcoxon(a, b, alternative=alternative)
        stat, p = float(stat), float(p)
        z = np.nan

    if not np.isfinite(z):
        # Recover |Z| from the (two-sided-equivalent) p-value as a fallback.
        z = float(norm.isf(min(max(p / 2.0, 1e-300), 0.5)))
        z = z * np.sign(np.median(diff)) if np.median(diff) != 0 else z

    r = abs(z) / np.sqrt(n_nz)
    return WilcoxonResult(
        statistic=stat,
        p_value=p,
        z=z,
        effect_size_r=float(r),
        n=len(a),
        n_nonzero=n_nz,
        median_diff=float(np.median(diff)),
        alternative=alternative,
    )


def bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float] = None,
    *,
    statistic: str = "mean_diff",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI.

    If ``b`` is given, bootstraps the paired difference statistic (default the
    mean difference). If ``b`` is None, bootstraps the mean of ``a``.
    Returns ``(point_estimate, ci_low, ci_high)``.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    if b is not None:
        b = np.asarray(b, dtype=np.float64)
        diff = a - b
        point = float(np.mean(diff))
        n = len(diff)
        boots = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            boots[i] = np.mean(diff[idx])
    else:
        point = float(np.mean(a))
        n = len(a)
        boots = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            boots[i] = np.mean(a[idx])
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 - (1 - ci) / 2)))
    return point, lo, hi


def paired_comparison(
    name_a: str,
    a: Sequence[float],
    name_b: str,
    b: Sequence[float],
    *,
    alternative: str = "two-sided",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict:
    """Full pairwise report: Wilcoxon p, effect size r, bootstrap CI of the diff.

    This is the canonical 'Mean +/- Std + Wilcoxon p + effect size r + 95% CI'
    bundle requested for the paper.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w = wilcoxon_signed_rank(a, b, alternative=alternative)
    point, lo, hi = bootstrap_ci(a, b, n_boot=n_boot, ci=ci, seed=seed)
    return {
        "model_a": name_a,
        "model_b": name_b,
        "mean_a": float(np.mean(a)),
        "std_a": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "mean_b": float(np.mean(b)),
        "std_b": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
        "wilcoxon": w.to_dict(),
        "mean_diff": point,
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": ci,
    }
