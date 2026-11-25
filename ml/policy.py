"""Stage 6: test selection policies on top of a ranking.

Policies (notes/models.md, p. 6):
  topk       first s tests of the ranking (s = ceil(k n) for k% of tests);
  budget     tests in rank order while the cumulative *predicted* duration <= B * predicted full run
             (stops at the first test that overflows); actual time is measured with eval_duration_ms;
  threshold  every test with score >= tau (the number of tests depends on how risky the commit looks).
             tau is chosen on a val range as the largest value with recall >= target.

Recall criteria for choosing a setting (CRITERIA): `commit` = share of red commits with a selected
failed test (stage 6 default); `defect` = share of (commit, defect) pairs with a selected test failing
because of that defect; `first` = the same over first-exposure pairs only (introduction commits).

All three are evaluated with the same tie rule as metrics.py (random keys shared by all methods).
Numbers obtained by choosing a setting on a range and reporting on the same range are in-sample.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from metrics import EvalSet, _cum_by_test, has_ties, order_from, positions, selection_columns

TARGETS = (0.90, 0.95)
CRITERIA = {"commit": "recall_commit", "defect": "recall_defect", "first": "recall_first"}
BUDGET_GRID = tuple(np.round(np.arange(0.01, 1.0001, 0.01), 2))


def _orders(es: EvalSet, scores: np.ndarray, keys: np.ndarray) -> list[np.ndarray]:
    scores = es.matrix(scores)
    p = keys.shape[0] if has_ties(scores) else 1
    return [order_from(scores, keys[i]) for i in range(p)]


def _mean_selection(es: EvalSet, sels: list[np.ndarray]) -> dict[str, np.ndarray]:
    acc: dict[str, np.ndarray] = {}
    for sel in sels:
        for k, v in selection_columns(es, sel, "sel").items():
            acc[k] = acc.get(k, 0) + v
    return {k: v / len(sels) for k, v in acc.items()}


def _aggregate(es: EvalSet, cols: dict[str, np.ndarray]) -> dict[str, float]:
    red = es.y.any(1)
    has_new = es.new.any(1)
    has_rep = es.repeat.any(1)
    n_def, n_first = es.def_per_commit.sum(), es.first_per_commit.sum()
    return {
        "recall_commit": float(cols["sel_hit"][red].mean()),
        "recall_defect": float(cols["sel_def_caught"].sum() / n_def) if n_def else float("nan"),
        "recall_first": float(cols["sel_first_caught"].sum() / n_first) if n_first else float("nan"),
        "recall_commit_new": float(cols["sel_hit_new"][has_new].mean()),
        "recall_commit_repeat": float(cols["sel_hit_repeat"][has_rep].mean()),
        "recall_test": float(cols["sel_caught"].sum() / es.y.sum()),
        "recall_test_new": float(cols["sel_caught_new"].sum() / max(es.new.sum(), 1)),
        "tests": float(cols["sel_tests"].mean()),
        "time": float(cols["sel_time"].mean()),
        "time_max": float(cols["sel_time"].max()),
        "time_red": float(cols["sel_time"][red].mean()),
        "time_green": float(cols["sel_time"][~red].mean()) if (~red).any() else float("nan"),
        "pred_time": float(cols["sel_pred_time"].mean()),
        "pred_time_max": float(cols["sel_pred_time"].max()),
        "empty_selections": float((cols["sel_tests"] == 0).mean()),
    }


def topk_curve(es: EvalSet, scores: np.ndarray, keys: np.ndarray) -> pd.DataFrame:
    orders = _orders(es, scores, keys)
    pos = [positions(o) for o in orders]
    rows = []
    for s in range(0, es.n + 1):
        agg = _aggregate(es, _mean_selection(es, [p <= s for p in pos]))
        rows.append({"policy": "topk", "setting": s / es.n, "n_tests": s, **agg})
    return pd.DataFrame(rows)


def budget_curve(es: EvalSet, scores: np.ndarray, keys: np.ndarray, grid=BUDGET_GRID) -> pd.DataFrame:
    orders = _orders(es, scores, keys)
    cums = [_cum_by_test(o, es.pred_dur) for o in orders]
    pred_total = es.pred_dur.sum(1, keepdims=True)
    rows = []
    for b in grid:
        agg = _aggregate(es, _mean_selection(es, [c <= b * pred_total + 1e-9 for c in cums]))
        rows.append({"policy": "budget", "setting": float(b), **agg})
    return pd.DataFrame(rows)


def threshold_stats(es: EvalSet, scores: np.ndarray, tau: float) -> dict[str, float]:
    return _aggregate(es, _mean_selection(es, [es.matrix(scores) >= tau]))


def threshold_for_recall(es: EvalSet, scores: np.ndarray, target: float, criterion: str = "commit") -> float:
    """Largest tau such that the share of units (red commits / defect pairs / first-exposure pairs)
    with a relevant failed test scored >= tau is >= target."""
    s = es.matrix(scores)
    if criterion == "commit":
        q = np.where(es.y, s, -np.inf).max(1)[es.y.any(1)]
    else:
        q = np.where(es.def_mask, s[es.def_commit], -np.inf).max(1)
        if criterion == "first":
            q = q[es.def_first]
    q = np.sort(q)[::-1]
    return float(q[math.ceil(target * len(q) - 1e-9) - 1])


def threshold_curve(es: EvalSet, scores: np.ndarray, n_points: int = 200) -> pd.DataFrame:
    qs = np.unique(np.quantile(scores, np.linspace(0, 1, n_points)))
    rows = [{"policy": "threshold", "setting": float(t), **threshold_stats(es, scores, t)} for t in qs]
    return pd.DataFrame(rows)


def cheapest_for_target(curve: pd.DataFrame, target: float, criterion: str = "commit") -> pd.Series | None:
    """The setting with the lowest mean actual time among those reaching the target recall."""
    ok = curve[curve[CRITERIA[criterion]] >= target - 1e-12]
    if ok.empty:
        return None
    return ok.sort_values(["time", "setting"]).iloc[0]


def policy_summary(es: EvalSet, scores: np.ndarray, keys: np.ndarray, method: str, probabilistic: bool) -> pd.DataFrame:
    """For each criterion, policy and target recall: the cheapest setting reaching it on this range (in-sample)."""
    curves = [topk_curve(es, scores, keys), budget_curve(es, scores, keys)]
    rows = []
    for criterion in CRITERIA:
        for curve in curves:
            for target in TARGETS:
                best = cheapest_for_target(curve, target, criterion)
                if best is not None:
                    rows.append({"method": method, "criterion": criterion, "target": target, **best.to_dict()})
        if probabilistic:
            for target in TARGETS:
                tau = threshold_for_recall(es, scores, target, criterion)
                rows.append({"method": method, "criterion": criterion, "target": target, "policy": "threshold", "setting": tau,
                             **threshold_stats(es, scores, tau)})
    return pd.DataFrame(rows)


def fixed_settings(es: EvalSet, scores: np.ndarray, keys: np.ndarray, method: str) -> pd.DataFrame:
    """top-k = 10/20/30/50 % of tests and budgets 10/20/30/50 % of predicted time."""
    tk = topk_curve(es, scores, keys)
    tk = tk[tk["n_tests"].isin([math.ceil(k * es.n) for k in (0.1, 0.2, 0.3, 0.5)])]
    bu = budget_curve(es, scores, keys, grid=(0.1, 0.2, 0.3, 0.5))
    out = pd.concat([tk, bu], ignore_index=True)
    out.insert(0, "method", method)
    return out
