"""Stage 6: non-ML baselines (notes/decisions.md, p. 2) as score functions.

Every baseline maps dataset rows to a score where higher = run earlier. Lexicographic keys are
turned into one dense rank, so ties stay ties and are broken by the shared random rule of
metrics.py (never by test_id). The random baseline is a constant score: all tests tie, and the
tie-break permutations are exactly random orders.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def lex_score(*keys: np.ndarray) -> np.ndarray:
    """Dense rank of the tuple (k1, k2, ...), higher tuple = higher score; equal tuples tie."""
    arrs = [np.asarray(k, dtype=float) for k in keys]
    stacked = np.column_stack(arrs)
    _, inverse = np.unique(stacked, axis=0, return_inverse=True)  # unique sorts rows lexicographically
    return inverse.reshape(-1).astype(float)


def col(rows: pd.DataFrame, name: str) -> np.ndarray:
    return rows[name].to_numpy(dtype=float)


Score = Callable[[pd.DataFrame], np.ndarray]

BASELINES: dict[str, Score] = {
    "random": lambda r: np.zeros(len(r)),
    "fail_rate": lambda r: col(r, "hist_fail_rate"),
    "recent": lambda r: -col(r, "hist_since_fail"),
    "criticality": lambda r: col(r, "meta_criticality"),
    "shortest": lambda r: -col(r, "meta_duration_mean"),
    "covers": lambda r: lex_score(col(r, "chg_cov_files"), col(r, "chg_mod_files")),
}

# Variants of the combined heuristic; one is chosen on val by mean APFD and then fixed.
COMBINED_VARIANTS: dict[str, Score] = {
    # The variant from notes/stage6-prompt.md: failed last commit, then covers, then recency.
    "comb_last_cov_recent": lambda r: lex_score(
        col(r, "hist_fail_last_1"), col(r, "chg_cov_files"), -col(r, "hist_since_fail")
    ),
    # Diff first: any covered changed file, then failed last commit, then recency.
    "comb_cov_last_recent": lambda r: lex_score(
        (col(r, "chg_cov_files") > 0).astype(float), col(r, "hist_fail_last_1"), -col(r, "hist_since_fail")
    ),
    # As the first, with the module overlap as an extra diff key.
    "comb_last_cov_mod_recent": lambda r: lex_score(
        col(r, "hist_fail_last_1"), col(r, "chg_cov_files"), col(r, "chg_mod_files"), -col(r, "hist_since_fail")
    ),
    # As the first, with the history-learned file -> test association before covers.
    "comb_last_file_cov_recent": lambda r: lex_score(
        col(r, "hist_fail_last_1"), col(r, "hist_file_fail_rate"), col(r, "chg_cov_files"), -col(r, "hist_since_fail")
    ),
}

ORACLE: Score = lambda r: lex_score(col(r, "label"), -col(r, "eval_duration_ms"))  # uses labels
INVERSE_ORACLE: Score = lambda r: lex_score(-col(r, "label"), col(r, "eval_duration_ms"))

LABELS_UK = {
    "random": "випадковий порядок",
    "fail_rate": "історія падінь",
    "recent": "недавні падіння",
    "criticality": "критичність",
    "shortest": "найкоротші першими",
    "covers": "covers ∩ діф",
    "combined": "комбінована евристика",
    "oracle": "оракул (верхня межа)",
    "comb_last_cov_recent": "fail_last_1 → cov_files → since_fail",
    "comb_cov_last_recent": "cov_files>0 → fail_last_1 → since_fail",
    "comb_last_cov_mod_recent": "fail_last_1 → cov_files → mod_files → since_fail",
    "comb_last_file_cov_recent": "fail_last_1 → file_fail_rate → cov_files → since_fail",
}
