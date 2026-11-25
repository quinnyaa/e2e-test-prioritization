"""Stage 8 (8A): self-checks of ml/stats.py on hand-made examples with known answers.

Usage:  ml/.venv/bin/python ml/check_stats.py   (~1 s)
"""

from __future__ import annotations

import sys

import numpy as np

from stats import (BOOT_N, boot_max, boot_mean, boot_ratio, bootstrap_counts, cliffs_delta, holm, percentile_ci, romano,
                   wilcoxon)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def main() -> int:
    print("Cliff's delta")
    check("complete dominance: d = 1", cliffs_delta([4, 5, 6], [1, 2, 3]) == 1.0)
    check("reverse dominance: d = -1", cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0)
    check("identical samples: d = 0", cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0)
    # x = [1, 2, 3], y = [2]: 1 < 2, 2 = 2, 3 > 2 -> (1 - 1) / 3 = 0
    check("one tie, one win, one loss: d = 0", cliffs_delta([1, 2, 3], [2]) == 0.0)
    # x = [2, 3], y = [1, 2, 4]: pairs > : (2,1),(3,1),(3,2) = 3; < : (2,4),(3,4) = 2 -> 1/6
    d = cliffs_delta([2, 3], [1, 2, 4])
    check("hand example: d = 1/6", abs(d - 1 / 6) < 1e-15, f"{d:.6f}")
    check("Romano thresholds", [romano(v) for v in (0.0, 0.146, 0.147, 0.329, 0.33, 0.473, 0.474, -0.9)]
          == ["negligible", "negligible", "small", "small", "medium", "medium", "large", "large"])

    print("Holm")
    got = holm([0.01, 0.04, 0.03, 0.005])
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    exp = np.array([0.03, 0.06, 0.06, 0.02])
    check("hand example [0.01, 0.04, 0.03, 0.005]", np.allclose(got, exp), f"{np.round(got, 4).tolist()}")
    check("capped at 1", holm([0.6, 0.7]).tolist() == [1.0, 1.0])
    g = holm([0.02, np.nan, 0.01])
    check("NaN stays NaN and is not counted (m = 2)", np.isnan(g[1]) and np.allclose(g[[0, 2]], [0.02, 0.02]), f"{g.tolist()}")

    print("Wilcoxon wrapper")
    x = np.array([0.9, 0.8, 0.7, 0.95, 0.6, 0.85, 0.75, 0.65])
    y = x - np.array([0.1, 0.05, 0.0, 0.2, -0.03, 0.07, 0.02, 0.04])
    w = wilcoxon(x, y)
    from scipy.stats import wilcoxon as sw
    ref = sw(x, y, zero_method="wilcox", alternative="two-sided", method="auto")
    check("matches scipy (zero_method=wilcox, two-sided, auto)", w["p"] == float(ref.pvalue) and w["statistic"] == float(ref.statistic),
          f"p = {w['p']:.4f}")
    check("n, zero differences, mean / median difference",
          w["n"] == 8 and w["n_zero"] == 1 and abs(w["median_diff"] - np.median(x - y)) < 1e-15 and abs(w["mean_diff"] - (x - y).mean()) < 1e-15)
    z = wilcoxon([1, 2], [1, 2])
    check("all differences zero -> p NaN (not an exception)", np.isnan(z["p"]) and z["n_zero"] == 2)

    print("Commit bootstrap")
    c1, c2 = bootstrap_counts(108), bootstrap_counts(108)
    check("reproducible with the seed", np.array_equal(c1, c2) and c1.shape == (BOOT_N, 108))
    check("every resample has 108 commits", bool((c1.sum(1) == 108).all()))
    check("another seed gives other resamples", not np.array_equal(c1, bootstrap_counts(108, seed=1)))
    const = np.full((3, 108), 0.25)
    ci = percentile_ci(boot_mean(c1, const))
    check("degenerate case: constant time -> interval = point", abs(ci["lo"] - 0.25) < 1e-12 and abs(ci["hi"] - 0.25) < 1e-12)
    num, den = np.full(108, 2.0), np.full(108, 4.0)
    ci = percentile_ci(boot_ratio(c1, num, den))
    check("degenerate case: constant ratio -> interval = point", ci["lo"] == ci["hi"] == 0.5)
    # Ratio of sums, not mean of ratios: commit A 1/1, commit B 0/3 -> pooled 1/4 (mean of ratios would be 0.5).
    counts = np.array([[1, 1], [2, 0], [0, 2]])
    r = boot_ratio(counts, [1.0, 0.0], [1.0, 3.0])
    check("ratio of sums per resample", np.allclose(r, [0.25, 1.0, 0.0]), f"{r.tolist()}")
    r = boot_ratio(np.array([[0, 2]]), [1.0, 0.0], [1.0, 0.0])
    check("resample without denominator -> NaN, counted as undefined", np.isnan(r[0]) and percentile_ci(r)["n_undefined"] == 1)
    # Seeds are averaged after the ratio: seed 1 = 1/2, seed 2 = 0/2 -> 0.25
    r = boot_ratio(np.array([[1, 1]]), np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([1.0, 1.0]))
    check("mean over seeds of per-seed ratios", np.allclose(r, [0.25]))
    m = boot_max(np.array([[1, 0, 1], [0, 3, 0]]), [0.1, 0.9, 0.3])
    check("max over the resampled commits", np.allclose(m, [0.3, 0.9]))
    check("mask in boot_mean (mean over red commits only)",
          np.allclose(boot_mean(np.array([[1, 1, 1]]), [0.2, 0.4, 0.9], mask=[1, 1, 0]), [0.3]))

    print(f"{'all checks passed' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
