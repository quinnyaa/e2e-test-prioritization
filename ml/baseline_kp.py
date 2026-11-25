"""Stage 8 (8A): the Kim-Porter history-based baseline (notes/stage8-protocol.md, p. 7.1).

Score of a test at commit c: an exponentially weighted failure history
    h_c = alpha * f_{c-1} + (1 - alpha) * h_{c-1},   h_0 = 0,
where f_{c-1} is the test's `label` (confirmedFail) at the previous commit -- the same quantity as
`hist_fail_last_1` (equal on every commit 1..431, checked in ml/run_stage8.py). Only commits < c enter h_c,
so the score is computed from commit 0 of the history, not from the start of the evaluated range.

alpha in ALPHAS is chosen on val by mean APFD (as the variant of the combined heuristic); on an exact tie
the earlier grid entry wins (as `tune`). The policy settings (top-k and budget B for the criteria
commit / defect / first and targets 0.90 / 0.95) are chosen on val by the same procedure as for every
other heuristic (`policy.policy_summary`). Both are written to kp.json by `run_stage8.py --range val`
and only read on the test part. Like the hist_ features, the score assumes full runs on every past
commit; it is not a dataset feature. baselines.py is not changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALPHAS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def label_matrix(history_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(commits, labels (C, n)) of rows of commits 0..end-1 sorted by (commit_index, test_id)."""
    commits = np.array(sorted(history_rows["commit_index"].unique()))
    if not np.array_equal(commits, np.arange(len(commits))):
        raise ValueError("Kim-Porter needs the history from commit 0 without gaps")
    n = history_rows["test_id"].nunique()
    y = history_rows["label"].to_numpy(dtype=float).reshape(len(commits), n)
    return commits, y


def kp_matrix(labels: np.ndarray, alpha: float) -> np.ndarray:
    """h (C, n): row c uses only rows < c of `labels`."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    h = np.zeros_like(labels, dtype=float)
    for c in range(1, len(labels)):
        h[c] = alpha * labels[c - 1] + (1 - alpha) * h[c - 1]
    return h


def kp_scores(history_rows: pd.DataFrame, rng: tuple[int, int], alpha: float) -> np.ndarray:
    """Scores of the commits in [a, b) as a flat vector in the (commit_index, test_id) order of Dataset.slice.
    `history_rows` = rows of commits 0..b-1 (sorted)."""
    commits, y = label_matrix(history_rows)
    a, b = rng
    if commits[-1] + 1 < b:
        raise ValueError("history rows end before the evaluated range")
    return kp_matrix(y[:b], alpha)[a:b].reshape(-1)
