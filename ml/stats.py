"""Stage 8: statistics of the evaluation (notes/stage8-protocol.md, p. 2-3).

  wilcoxon       paired two-sided Wilcoxon signed-rank test (scipy, zero_method="wilcox", method="auto")
  cliffs_delta   Cliff's delta d = P(X > Y) - P(X < Y) over all pairs, with Romano's verbal magnitude
  holm           Holm step-down adjusted p-values (within one family)
  bootstrap      commit bootstrap: 10 000 resamples of the commits with replacement, default_rng(20261117),
                 percentile 95 % interval. One index matrix is drawn per call and shared by every statistic,
                 so recall and time of the same resample come from the same commits. Ratios of sums
                 (recall) are computed as the ratio of the resampled sums, never as a mean of ratios.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

BOOT_N = 10_000
BOOT_SEED = 20261117
ROMANO = ((0.147, "negligible"), (0.33, "small"), (0.474, "medium"))  # |d| < bound -> label; else "large"
ROMANO_UK = {"negligible": "знехтуваний", "small": "малий", "medium": "середній", "large": "великий"}


# ---------------------------------------------------------------------- paired test


def wilcoxon(x, y) -> dict:
    """Paired test of x - y. Returns n, zero differences, mean / median difference, statistic and p.
    If every difference is zero the test is undefined: statistic and p are NaN (reported, not raised)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of the same length (paired by commit)")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("NaN in paired samples")
    d = x - y
    out = {
        "n": int(len(d)),
        "n_zero": int((d == 0).sum()),
        "mean_diff": float(d.mean()) if len(d) else float("nan"),
        "median_diff": float(np.median(d)) if len(d) else float("nan"),
    }
    if out["n_zero"] == len(d):
        out.update(statistic=float("nan"), p=float("nan"))
        return out
    res = sps.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="auto")
    out.update(statistic=float(res.statistic), p=float(res.pvalue))
    return out


# ---------------------------------------------------------------------- effect size


def cliffs_delta(x, y) -> float:
    """d = (#{x_i > y_j} - #{x_i < y_j}) / (|x| |y|) over all pairs (x is the method, y the reference)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not len(x) or not len(y):
        return float("nan")
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def romano(d: float) -> str:
    """Verbal magnitude of |d| with Romano et al. thresholds 0.147 / 0.33 / 0.474."""
    if d != d:
        return "n/a"
    a = abs(d)
    for bound, label in ROMANO:
        if a < bound:
            return label
    return "large"


# ---------------------------------------------------------------------- multiple comparisons


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values, same order as the input; NaN p-values stay NaN and are not counted."""
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    idx = np.where(~np.isnan(p))[0]
    m = len(idx)
    if not m:
        return out
    order = idx[np.argsort(p[idx], kind="mergesort")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


# ---------------------------------------------------------------------- commit bootstrap


def bootstrap_counts(n_commits: int, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> np.ndarray:
    """(n_boot, n_commits) multiplicity of every commit in every resample (resampling with replacement).
    A fresh default_rng(seed) per call, so the same call always gives the same resamples."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_commits, size=(n_boot, n_commits))
    counts = np.zeros((n_boot, n_commits), dtype=np.int32)
    np.add.at(counts, (np.repeat(np.arange(n_boot), n_commits), idx.ravel()), 1)
    return counts


def boot_ratio(counts: np.ndarray, num, den) -> np.ndarray:
    """Per resample: mean over seeds of sum(num) / sum(den). num, den: (S, C) or (C,). NaN where sum(den) = 0."""
    num = np.atleast_2d(np.asarray(num, dtype=float))
    den = np.atleast_2d(np.asarray(den, dtype=float))
    n_s = counts @ num.T  # (B, S)
    d_s = counts @ den.T  # (B, S) or (B, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(d_s > 0, n_s / d_s, np.nan)
    return r.mean(1)


def boot_mean(counts: np.ndarray, x, mask=None) -> np.ndarray:
    """Per resample: mean over seeds of the mean of x over the resampled commits (optionally only where mask)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    w = np.ones(x.shape[1]) if mask is None else np.asarray(mask, dtype=float)
    return boot_ratio(counts, x * w, w)


def boot_max(counts: np.ndarray, x) -> np.ndarray:
    """Per resample: mean over seeds of the max of x over the resampled commits."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    present = counts > 0  # (B, C)
    out = np.empty((counts.shape[0], x.shape[0]))
    for s in range(x.shape[0]):
        out[:, s] = np.where(present, x[s][None, :], -np.inf).max(1)
    return out.mean(1)


def percentile_ci(samples, level: float = 0.95) -> dict:
    """Percentile interval of the bootstrap samples; resamples with an undefined value (NaN) are dropped
    and counted (e.g. a resample without first exposures has no recall_first)."""
    s = np.asarray(samples, dtype=float)
    ok = s[~np.isnan(s)]
    a = (1 - level) / 2
    lo, hi = (np.percentile(ok, [100 * a, 100 * (1 - a)]) if len(ok) else (math.nan, math.nan))
    return {"lo": float(lo), "hi": float(hi), "level": level, "n_boot": int(len(s)), "n_undefined": int(len(s) - len(ok))}
