"""Stage 6: prioritization and selection metrics shared by baselines, models and stage 8.

Conventions (recorded in notes/decisions.md, p. 7, and notes/models.md):
- A method gives every (commit, test) row a score; a higher score means "run earlier".
- Fault (main definition) = a failed test (label = 1). n = tests per commit, m = failed tests,
  TF_i = 1-based position of the i-th failed test:
    APFD  = 1 - sum(TF_i) / (n m) + 1 / (2n);
    APFDc = sum_i (sum_{j >= TF_i} t_j - t_{TF_i} / 2) / (m sum_j t_j),  t_j = eval_duration_ms.
- Fault (additional definition) = an open defect with at least one positive in the commit
  (positives.csv `causes`, ';'-separated); TF_d = position of the first test it makes fail.
- First exposure of a defect = the (commit, defect) pair at the defect's introduction commit
  (defects.csv `introduced_at`). There is no masking in the histories, so a detectable defect makes
  tests fail already at that commit; later pairs of its life are repeats of an already visible
  defect (and are only visible to history features because every past commit ran the full suite).
- New failure: label = 1 and hist_fail_last_1 = 0; repeated failure: label = 1 and hist_fail_last_1 = 1.
  APFD_new ranks all tests as usual but counts only new failures (commits with >= 1 new failure).
- Commits without failures are excluded from APFD and from the recall denominators; selection
  time shares are averaged over all commits of the range (the tool cannot know which are red).
- Durations: actual `eval_duration_ms` (the pilot convention), including the inflated durations of
  failed tests (expect/action time-outs). The full run is the sequential sum over all tests.
- Ties: one rule for every method. The tie-break key of perm p for commit c is drawn from
  default_rng([seed, c]) (common random numbers: every method sees the same keys), and metrics
  are averaged over perms. An ordering without ties is evaluated once. Never by test_id.
- Selection:
    top-k%        first ceil(k n) tests of the ranking;
    budget B%     tests in rank order while the cumulative *predicted* duration (meta_duration_mean)
                  <= B * predicted full run; stops at the first test that overflows;
    recall@B% time (prioritization view) a commit counts if the actual cumulative time up to and
                  including its first failed test is <= B * actual full run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

TOPK = (0.1, 0.2, 0.3, 0.5)
BUDGETS = (0.1, 0.2, 0.3, 0.5)
N_PERM = 20
N_PERM_RANDOM = 200
TIE_SEED = 20261117


def pct_key(x: float) -> str:
    return str(int(round(100 * x)))


@dataclass
class EvalSet:
    """Matrix view (commits x tests) of a commit range, rows sorted by (commit_index, test_id)."""

    commits: np.ndarray  # (C,)
    tests: np.ndarray  # (n,)
    y: np.ndarray  # (C, n) bool, failed
    new: np.ndarray  # (C, n) bool, new failure
    dur: np.ndarray  # (C, n) actual duration, ms
    pred_dur: np.ndarray  # (C, n) predicted duration (meta_duration_mean), ms
    def_commit: np.ndarray  # (K,) row index into `commits` of each (commit, detected defect)
    def_mask: np.ndarray  # (K, n) bool, tests failing because of that defect
    def_ids: np.ndarray  # (K,)
    def_first: np.ndarray  # (K,) bool, the pair is the defect's introduction commit (first exposure)

    @property
    def n(self) -> int:
        return len(self.tests)

    @property
    def repeat(self) -> np.ndarray:
        return self.y & ~self.new

    @property
    def def_per_commit(self) -> np.ndarray:
        return np.bincount(self.def_commit, minlength=len(self.commits))

    @property
    def first_per_commit(self) -> np.ndarray:
        return np.bincount(self.def_commit, weights=self.def_first.astype(float), minlength=len(self.commits)).astype(int)

    def matrix(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr.reshape(len(self.commits), self.n)


def build_eval_set(rows: pd.DataFrame, positives: pd.DataFrame, introduced_at: dict[str, int]) -> EvalSet:
    """`introduced_at` maps defect_id -> introduction commit (defects.csv), for first-exposure pairs."""
    tests = np.array(sorted(rows["test_id"].unique()))
    commits = np.array(sorted(rows["commit_index"].unique()))
    C, n = len(commits), len(tests)
    if len(rows) != C * n:
        raise ValueError("rows are not a full commit x test grid")
    expected = np.tile(tests, C)
    if not (rows["test_id"].to_numpy() == expected).all() or not (rows["commit_index"].to_numpy() == np.repeat(commits, n)).all():
        raise ValueError("rows must be sorted by (commit_index, test_id)")
    y = rows["label"].to_numpy().reshape(C, n).astype(bool)
    new = y & (rows["hist_fail_last_1"].to_numpy().reshape(C, n) == 0)
    dur = rows["eval_duration_ms"].to_numpy(dtype=float).reshape(C, n)
    pred = rows["meta_duration_mean"].to_numpy(dtype=float).reshape(C, n)

    # Defect-level faults from positives.csv (analysis only).
    pos = positives[positives["commit_index"].isin(commits)]
    c_index = {c: i for i, c in enumerate(commits)}
    t_index = {t: j for j, t in enumerate(tests)}
    groups: dict[tuple[int, str], np.ndarray] = {}
    for c, t, causes in pos[["commit_index", "test_id", "causes"]].itertuples(index=False):
        if not y[c_index[c], t_index[t]]:
            raise AssertionError(f"positives.csv lists ({c}, {t}) but its label is 0")
        for d in str(causes).split(";"):
            key = (c_index[c], d)
            if key not in groups:
                groups[key] = np.zeros(n, dtype=bool)
            groups[key][t_index[t]] = True
    keys = sorted(groups)
    def_commit = np.array([k[0] for k in keys], dtype=int)
    def_ids = np.array([k[1] for k in keys], dtype=object)
    def_mask = np.array([groups[k] for k in keys], dtype=bool).reshape(len(keys), n)
    intro = np.array([int(introduced_at[d]) for d in def_ids], dtype=int)  # KeyError = unknown defect
    if (intro > commits[def_commit]).any():
        raise AssertionError("a defect causes a failure before its introduction commit")
    def_first = intro == commits[def_commit]
    if y.sum() != len(pos):
        raise AssertionError("every positive must have a row in positives.csv")
    return EvalSet(commits, tests, y, new, dur, pred, def_commit, def_mask, def_ids, def_first)


# ---------- ordering ----------


def tie_keys(es: EvalSet, n_perm: int, seed: int = TIE_SEED) -> np.ndarray:
    """(P, C, n) uniform keys; the key of (perm p, commit c) does not depend on P or on the range."""
    out = np.empty((n_perm, len(es.commits), es.n))
    for i, c in enumerate(es.commits):
        out[:, i, :] = np.random.default_rng([seed, int(c)]).random((n_perm, es.n))
    return out


def has_ties(scores: np.ndarray) -> bool:
    s = np.sort(scores, axis=1)
    return bool(np.any(s[:, 1:] == s[:, :-1]))


def order_from(scores: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Indices of tests in run order: descending score, ties by ascending random key."""
    return np.lexsort((keys, -scores), axis=-1)


def positions(order: np.ndarray) -> np.ndarray:
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.arange(1, order.shape[1] + 1)[None, :].repeat(order.shape[0], 0), axis=1)
    return pos


# ---------- per-permutation metrics ----------


def _apfd(pos: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    m = mask.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = 1 - (pos * mask).sum(1) / (n * m) + 1 / (2 * n)
    return np.where(m > 0, v, np.nan)


def _apfdc(order: np.ndarray, dur: np.ndarray, mask: np.ndarray) -> np.ndarray:
    t_ord = np.take_along_axis(dur, order, axis=1)
    suffix_ord = np.cumsum(t_ord[:, ::-1], axis=1)[:, ::-1]  # sum_{j >= r} t_j
    suffix = np.empty_like(suffix_ord)
    np.put_along_axis(suffix, order, suffix_ord, axis=1)
    m = mask.sum(1)
    total = dur.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = ((suffix - dur / 2) * mask).sum(1) / (m * total)
    return np.where(m > 0, v, np.nan)


def _cum_by_test(order: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Inclusive cumulative sum of `values` in run order, mapped back to test columns."""
    cum_ord = np.cumsum(np.take_along_axis(values, order, axis=1), axis=1)
    out = np.empty_like(cum_ord)
    np.put_along_axis(out, order, cum_ord, axis=1)
    return out


def selection_columns(es: EvalSet, sel: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """Per-commit outcome of selecting the tests in `sel` (C, n) bool."""
    total = es.dur.sum(1)
    pred_total = es.pred_dur.sum(1)
    C = len(es.commits)
    if len(es.def_commit):
        caught = (es.def_mask & sel[es.def_commit]).any(1).astype(float)
        def_caught = np.bincount(es.def_commit, weights=caught, minlength=C)
        first_caught = np.bincount(es.def_commit, weights=caught * es.def_first, minlength=C)
    else:
        def_caught = first_caught = np.zeros(C)
    return {
        f"{prefix}_hit": (es.y & sel).any(1).astype(float),
        f"{prefix}_hit_new": (es.new & sel).any(1).astype(float),
        f"{prefix}_hit_repeat": (es.repeat & sel).any(1).astype(float),
        f"{prefix}_caught": (es.y & sel).sum(1).astype(float),
        f"{prefix}_caught_new": (es.new & sel).sum(1).astype(float),
        f"{prefix}_def_caught": def_caught,
        f"{prefix}_first_caught": first_caught,
        f"{prefix}_tests": sel.sum(1) / es.n,
        f"{prefix}_time": (es.dur * sel).sum(1) / total,
        f"{prefix}_pred_time": (es.pred_dur * sel).sum(1) / pred_total,
    }


def perm_metrics(es: EvalSet, order: np.ndarray) -> dict[str, np.ndarray]:
    n = es.n
    pos = positions(order)
    out: dict[str, np.ndarray] = {
        "apfd": _apfd(pos, es.y, n),
        "apfd_new": _apfd(pos, es.new, n),
        "apfd_repeat": _apfd(pos, es.repeat, n),
        "apfdc": _apfdc(order, es.dur, es.y),
        "apfdc_new": _apfdc(order, es.dur, es.new),
    }
    # Defect-level APFD: TF_d = first position of a test failing because of defect d.
    if len(es.def_commit):
        tf = np.where(es.def_mask, pos[es.def_commit], n + 1).min(1)
        k = np.bincount(es.def_commit, minlength=len(es.commits))
        s = np.bincount(es.def_commit, weights=tf, minlength=len(es.commits))
        with np.errstate(invalid="ignore", divide="ignore"):
            out["apfd_defect"] = np.where(k > 0, 1 - s / (n * k) + 1 / (2 * n), np.nan)
    else:
        out["apfd_defect"] = np.full(len(es.commits), np.nan)

    # Prioritization view: actual time until the first failed test has finished.
    cum = _cum_by_test(order, es.dur)
    first_time = np.where(es.y, cum, np.inf).min(1)
    ttd = np.where(es.y.any(1), first_time / es.dur.sum(1), np.nan)
    out["ttd_share"] = ttd
    out["first_pos"] = np.where(es.y.any(1), np.where(es.y, pos, n + 1).min(1), np.nan)
    for b in BUDGETS:
        out[f"rt{pct_key(b)}_hit"] = np.where(np.isnan(ttd), np.nan, (ttd <= b + 1e-12).astype(float))

    # Selection views.
    for k in TOPK:
        sel = pos <= math.ceil(k * n)
        out.update(selection_columns(es, sel, f"top{pct_key(k)}"))
    cum_pred = _cum_by_test(order, es.pred_dur)
    pred_total = es.pred_dur.sum(1, keepdims=True)
    for b in BUDGETS:
        sel = cum_pred <= b * pred_total + 1e-9
        out.update(selection_columns(es, sel, f"bud{pct_key(b)}"))
    return out


# ---------- evaluation of a score matrix ----------


def evaluate(es: EvalSet, scores: np.ndarray, n_perm: int = N_PERM, seed: int = TIE_SEED,
             keys: np.ndarray | None = None) -> pd.DataFrame:
    """Per-commit metrics averaged over tie-break permutations (one pass if there are no ties)."""
    scores = np.asarray(scores, dtype=float).reshape(len(es.commits), es.n)
    if np.isnan(scores).any():
        raise ValueError("NaN scores")
    p = n_perm if has_ties(scores) else 1
    if keys is None or keys.shape[0] < p:
        keys = tie_keys(es, p, seed)
    acc: dict[str, np.ndarray] = {}
    for i in range(p):
        res = perm_metrics(es, order_from(scores, keys[i]))
        for name, v in res.items():
            acc[name] = acc.get(name, 0) + v
    per = pd.DataFrame({name: v / p for name, v in acc.items()})
    per.insert(0, "commit_index", es.commits)
    per.insert(1, "n_fail", es.y.sum(1))
    per.insert(2, "n_new", es.new.sum(1))
    per.insert(3, "n_repeat", es.repeat.sum(1))
    per.insert(4, "n_defects", es.def_per_commit)
    per.insert(5, "n_first", es.first_per_commit)
    per.insert(6, "n_perm", p)
    return per


def summarize(per: pd.DataFrame) -> dict[str, float]:
    """Aggregate per-commit metrics: APFD-type means over red commits, pooled test recall,
    commit recall over red commits (new/repeat: over commits with such failures), defect / first-exposure
    recall pooled over (commit, defect) pairs, time over all commits."""
    red = per["n_fail"] > 0
    has_new = per["n_new"] > 0
    has_rep = per["n_repeat"] > 0
    out: dict[str, float] = {
        "commits": float(len(per)),
        "red_commits": float(red.sum()),
        "defect_pairs": float(per["n_defects"].sum()),
        "first_pairs": float(per["n_first"].sum()),
        "apfd": per.loc[red, "apfd"].mean(),
        "apfd_median": per.loc[red, "apfd"].median(),
        "apfdc": per.loc[red, "apfdc"].mean(),
        "apfd_new": per.loc[has_new, "apfd_new"].mean(),
        "apfdc_new": per.loc[has_new, "apfdc_new"].mean(),
        "apfd_repeat": per.loc[has_rep, "apfd_repeat"].mean(),
        "apfd_defect": per.loc[per["n_defects"] > 0, "apfd_defect"].mean(),
        "ttd_share": per.loc[red, "ttd_share"].mean(),
        "first_pos": per.loc[red, "first_pos"].mean(),
    }
    for b in BUDGETS:
        out[f"recall_time_{pct_key(b)}"] = per.loc[red, f"rt{pct_key(b)}_hit"].mean()
    prefixes = [f"top{pct_key(k)}" for k in TOPK] + [f"bud{pct_key(b)}" for b in BUDGETS]
    for pre in prefixes:
        out.update(summarize_selection(per, pre))
    return out


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def summarize_selection(per: pd.DataFrame, pre: str) -> dict[str, float]:
    """recall_defect / recall_first are pooled over (commit, defect) pairs / first-exposure pairs."""
    red = per["n_fail"] > 0
    has_new = per["n_new"] > 0
    has_rep = per["n_repeat"] > 0
    return {
        f"{pre}_recall_commit": per.loc[red, f"{pre}_hit"].mean(),
        f"{pre}_recall_commit_new": per.loc[has_new, f"{pre}_hit_new"].mean(),
        f"{pre}_recall_commit_repeat": per.loc[has_rep, f"{pre}_hit_repeat"].mean(),
        f"{pre}_recall_test": per[f"{pre}_caught"].sum() / per["n_fail"].sum(),
        f"{pre}_recall_test_new": per[f"{pre}_caught_new"].sum() / max(per["n_new"].sum(), 1),
        f"{pre}_recall_defect": _ratio(per[f"{pre}_def_caught"].sum(), per["n_defects"].sum()),
        f"{pre}_recall_first": _ratio(per[f"{pre}_first_caught"].sum(), per["n_first"].sum()),
        f"{pre}_tests": per[f"{pre}_tests"].mean(),
        f"{pre}_time": per[f"{pre}_time"].mean(),
        f"{pre}_time_max": per[f"{pre}_time"].max(),
        f"{pre}_time_red": per.loc[red, f"{pre}_time"].mean(),
        f"{pre}_pred_time": per[f"{pre}_pred_time"].mean(),
    }


def classification_diagnostics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Pointwise diagnostics only (not selection criteria). p must be probabilities for Brier/F1."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "f1_at_0_5": f1_score(y, (p >= 0.5).astype(int), zero_division=0),
        "mean_p": float(p.mean()),
    }


# ---------- analytic expectations (self-checks) ----------


def random_commit_hit_expectation(n: int, m: int, s: int) -> float:
    """P(at least one of m failed tests among s random picks out of n)."""
    if m == 0:
        return float("nan")
    return 1 - math.comb(n - m, s) / math.comb(n, s)


def random_defect_apfd_expectation(n: int, ks: list[int]) -> float:
    """E[APFD] of a random order when defect i is detected by k_i tests: E[TF_i] = (n+1)/(k_i+1)."""
    tf = sum((n + 1) / (k + 1) for k in ks)
    return 1 - tf / (n * len(ks)) + 1 / (2 * n)
