"""Stage 8 (8A): closed-loop replay (selection feedback) and cold start (notes/stage8-protocol.md, p. 5.4, 5.6).

Infrastructure only: 8A checks it on val (ml/check_replay.py); the experiments on test run in 8C.

Observed history (`ObservedHistoryState`, a subclass of features_online.HistoryState; features_online.py is
not changed). After the tool has run only the selected tests of a commit, `update_partial(diff, run, executed)`
adds the commit to the history as it was *observed*:
  * an unexecuted test contributes 0 to the failure vector (no observed failure), so hist_fail_last_*,
    hist_since_fail, hist_fail_rate, the peer rates, hist_prev_commit_fails and the file <-> test
    associations see only executed failures;
  * the mean duration (meta_duration_mean) is updated only from executed passing tests;
  * the diff itself is always known (file change counts are updated for every commit).
With every test executed, `update_partial` is exactly `HistoryState.update` (checked).

Closed loop (`run_loop`). For every commit c of [start, end): features from the observed history ->
score -> selection by the frozen policy -> the history records the executed tests only. A full run
happens on the commits with (c - start + 1) mod N = 0 (N in {5, 10, inf}; N = 1 = always full, i.e. the
open loop); on those commits every test is executed and recorded, and the selection is still computed
and logged. Each method and seed has its own loop (its own observed history). Ties of a heuristic score
are broken with the permutation-0 key of the commit (as the CLI does): one realised selection per loop,
not an average over the 20 permutations of ml/metrics.py.

Cold start (`ObservedHistoryState.reset_tests`). The history state of the listed tests is reset as for a
test without runs: no past failures (hist_fail_last_* = 0, hist_since_fail = cap, hist_never_failed = 1,
hist_fail_rate = 0), meta_duration_mean = avgDurationMs of the manifest, empty file <-> test associations.
Variant (a): reset once at the first commit of the range, then the history accumulates again;
variant (b): reset before every commit of the range (worst case). Choices left open by the protocol
(recorded in notes/evaluation.md and in the protocol log):
  peer_scope        "own_only" (default): only the cold tests' own history is reset; their past results stay
                    visible to the peer rates of their module neighbours and to hist_prev_commit_fails
                    (commit-level features stay constant within the commit); "remove": the cold tests'
                    past results are removed from the history altogether (also for the neighbours);
  rate_denominator  "since_reset" (default): hist_fail_rate and hist_file_fail_rate of a cold test count only
                    the commits / file changes after its reset; "global": the dataset denominators.
A cold test's own peer rate is computed from its neighbours' history and is not reset (it is not its own history).

Test lock: the history is read only up to `upto` <= 432 (first lines of runs.jsonl), unless
allow_test_history=True (8C). Nothing here reads dataset labels of the test part.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import HIST_WINDOWS, HistoryError, run_outcome
from features_online import ROOT, HistoryState, as_dataset_values, history_diffs
from metrics import selection_columns
from prioritize import Method, Policy, rank_order, scores, select

HISTORY = ROOT / "data" / "history-600"
TEST_START = 432  # first commit of the hold-out test part (splits.json); history beyond needs allow_test_history
COLD_TESTS = ("ADM-01", "AUTH-01", "CART-01", "CAT-01", "CHK-01", "ORD-01", "PRD-01", "SRCH-01", "E2E-01")
PEER_SCOPES = ("own_only", "remove")
RATE_DENOMINATORS = ("since_reset", "global")


class ObservedHistoryState(HistoryState):
    def __init__(self, tests=None, peer_scope: str = "own_only", rate_denominator: str = "since_reset"):
        super().__init__(tests)
        if peer_scope not in PEER_SCOPES or rate_denominator not in RATE_DENOMINATORS:
            raise ValueError(f"peer_scope in {PEER_SCOPES}, rate_denominator in {RATE_DENOMINATORS}")
        self.peer_scope = peer_scope
        self.rate_denominator = rate_denominator
        n = len(self.test_ids)
        self.born = np.zeros(n, dtype=int)  # n_past at the last reset (0 = never reset)
        self.cold = np.zeros(n, dtype=bool)
        self.changes_at_reset: dict[int, dict[str, int]] = {}

    # ------------------------------------------------------------------ partial update (observed history)

    def update_partial(self, diff, run: dict, executed: np.ndarray) -> None:
        """Mirror of HistoryState.update where only the `executed` tests (bool, test_ids order) are observed."""
        executed = np.asarray(executed, dtype=bool)
        if executed.shape != (len(self.test_ids),):
            raise ValueError("executed must be a bool vector over the suite")
        self.commits_seen += 1
        self.last_sha = run.get("sha", self.last_sha)
        if not run.get("buildOk", True):
            return
        label, _flaky, passed, duration = run_outcome(run, self.test_ids)  # the logged run is complete
        obs = (label * executed).astype(label.dtype)
        self.fail_rows.append(obs.astype(float))
        self.last_fail = np.where(obs > 0, self.n_past, self.last_fail)
        self.total_fail += obs
        seen = passed & executed
        self.dur_sum += np.where(seen, duration, 0.0)
        self.dur_cnt += seen
        for path in diff:
            self.file_changes[path] += 1
            self.file_cofail[path] = self.file_cofail[path] + obs
        self.n_past += 1

    # ------------------------------------------------------------------ cold start

    def reset_tests(self, test_ids) -> None:
        idx = [self.test_ids.index(t) for t in test_ids]
        for k in idx:
            self.total_fail[k] = 0
            self.last_fail[k] = -1
            self.dur_sum[k] = 0.0
            self.dur_cnt[k] = 0
            for path in list(self.file_cofail):
                arr = self.file_cofail[path].copy()
                arr[k] = 0.0
                self.file_cofail[path] = arr
            if self.peer_scope == "remove":
                for i, row in enumerate(self.fail_rows):
                    row = row.copy()
                    row[k] = 0.0
                    self.fail_rows[i] = row
            self.born[k] = self.n_past
            self.cold[k] = True
            self.changes_at_reset[k] = dict(self.file_changes)

    def features(self, diff, commit_index: int, sha: str, ci: dict | None = None) -> pd.DataFrame:
        frame = super().features(diff, commit_index, sha, ci)
        if not self.cold.any():
            return frame
        past = np.array(self.fail_rows) if self.fail_rows else np.zeros((0, len(self.test_ids)))
        for k in np.where(self.cold)[0]:
            own = past[self.born[k]:, k]
            for w in HIST_WINDOWS:
                frame.loc[k, f"hist_fail_last_{w}"] = int(own[-w:].sum()) if len(own) else 0
            if self.rate_denominator == "since_reset":
                runs = self.n_past - self.born[k]
                frame.loc[k, "hist_fail_rate"] = float(self.total_fail[k] / runs) if runs else 0.0
                rate = 0.0
                for path in diff:
                    changes = self.file_changes[path] - self.changes_at_reset[k].get(path, 0)
                    if changes > 0:
                        rate = max(rate, float(self.file_cofail[path][k] / changes))
                frame.loc[k, "hist_file_fail_rate"] = rate
        return as_dataset_values(frame)


# ---------------------------------------------------------------------- history (test lock)


def load_runs_prefix(history: Path, upto: int, allow_test_history: bool = False) -> list[dict]:
    """First `upto` runs of runs.jsonl, reading no further line; chain checked as features_online.load_runs."""
    if upto > TEST_START and not allow_test_history:
        raise HistoryError(f"upto {upto} > {TEST_START}: the history of the test part needs allow_test_history=True (8C)")
    runs = []
    with (history / "runs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if len(runs) >= upto:
                break
            if line.strip():
                runs.append(json.loads(line))
    if len(runs) != upto:
        raise HistoryError(f"{history}/runs.jsonl has fewer than {upto} runs")
    for i, run in enumerate(runs):
        if run["index"] != i or (i > 0 and run["parentSha"] != runs[i - 1]["sha"]):
            raise HistoryError(f"broken commit chain at line {i}")
        if not run.get("buildOk", True):
            raise HistoryError(f"commit {i} does not build: the replay expects a full commit x test grid")
    return runs


@dataclass
class History:
    runs: list[dict]
    diffs: list[dict]

    @classmethod
    def load(cls, upto: int, history: Path = HISTORY, allow_test_history: bool = False) -> "History":
        runs = load_runs_prefix(history, upto, allow_test_history)
        return cls(runs, history_diffs(history, runs))

    def state_before(self, start: int, **kw) -> ObservedHistoryState:
        """Observed history after FULL runs of commits 0..start-1 (as the dataset)."""
        state = ObservedHistoryState(**kw)
        for run, diff in zip(self.runs[:start], self.diffs[:start]):
            state.update(diff, run)
        return state

    def ci(self, c: int) -> dict:
        return {"typecheckOk": self.runs[c]["typecheckOk"], "unitOk": self.runs[c]["unitOk"]}


# ---------------------------------------------------------------------- closed loop


def full_run_commits(start: int, end: int, n: float) -> np.ndarray:
    """Bool per commit of [start, end): (c - start + 1) mod N = 0; N = inf -> never."""
    c = np.arange(start, end)
    if n == math.inf:
        return np.zeros(len(c), dtype=bool)
    return (c - start + 1) % int(n) == 0


@dataclass
class LoopResult:
    method: str
    seed: int
    n_full: float
    commits: np.ndarray
    full: np.ndarray  # (C,) bool
    selected: np.ndarray  # (C, n) bool, selection computed on every commit
    executed: np.ndarray  # (C, n) bool, = selected, or all tests on full-run commits
    test_ids: list[str]
    features: list[pd.DataFrame] = field(default_factory=list)  # only with keep_features


def run_loop(history: History, start: int, end: int, method: Method, policy: Policy, n_full: float,
             cold: tuple[str, ...] = (), cold_variant: str | None = None, keep_features: bool = False,
             state: ObservedHistoryState | None = None, **state_kw) -> LoopResult:
    """Closed loop over [start, end) starting from full runs of 0..start-1 (or a copy of `state`)."""
    if end > len(history.runs):
        raise HistoryError("history shorter than the replayed range")
    if cold and cold_variant not in ("a", "b"):
        raise ValueError("cold_variant must be 'a' (reset at the first commit) or 'b' (reset on every commit)")
    st = copy.deepcopy(state) if state is not None else history.state_before(start, **state_kw)
    full = full_run_commits(start, end, n_full)
    C, n = end - start, len(st.test_ids)
    sel_all, exe_all, feats = np.zeros((C, n), bool), np.zeros((C, n), bool), []
    for i, c in enumerate(range(start, end)):
        run, diff = history.runs[c], history.diffs[c]
        if cold and (cold_variant == "b" or i == 0):
            st.reset_tests(cold)
        rows = st.features(diff, run["index"], run["sha"], history.ci(c))
        s = scores(method, rows)
        order = rank_order(s, c)
        sel = select(policy, s, order, rows["meta_duration_mean"].to_numpy(dtype=float))
        exe = np.ones(n, bool) if full[i] else sel
        st.update_partial(diff, run, exe)
        sel_all[i], exe_all[i] = sel, exe
        if keep_features:
            feats.append(rows)
    return LoopResult(method.name, -1 if method.seed is None else int(method.seed), n_full, np.arange(start, end), full,
                      sel_all, exe_all, list(st.test_ids), feats)


def open_loop(rows: pd.DataFrame, method: Method, policy: Policy) -> LoopResult:
    """Selection on the dataset rows (history from full runs), no full runs: the main evaluation's view,
    with the permutation-0 tie key (as in the loops)."""
    commits = np.array(sorted(rows["commit_index"].unique()))
    tests = sorted(rows["test_id"].unique())
    sel = np.zeros((len(commits), len(tests)), bool)
    for i, c in enumerate(commits):
        r = rows[rows["commit_index"] == c].sort_values("test_id", kind="mergesort").reset_index(drop=True)
        s = scores(method, r)
        sel[i] = select(policy, s, rank_order(s, int(c)), r["meta_duration_mean"].to_numpy(dtype=float))
    return LoopResult(method.name, -1 if method.seed is None else int(method.seed), math.inf, commits,
                      np.zeros(len(commits), bool), sel, sel.copy(), tests)


# ---------------------------------------------------------------------- metrics


def loop_metrics(es, res: LoopResult, defects: pd.DataFrame, cold: tuple[str, ...] = ()) -> dict:
    """recall_first and time (all commits / commits without a full run), detection delay, share of defects
    not detected before their fix. `es` = metrics.EvalSet of the same range; `defects` = defects.csv rows
    (introduced_at, fixed_at); fixed_at beyond the range = censored (alive at the end of the range)."""
    if not np.array_equal(es.commits, res.commits) or list(es.tests) != res.test_ids:
        raise ValueError("EvalSet and loop cover different commits / tests")
    end = int(res.commits[-1]) + 1
    ex = selection_columns(es, res.executed, "x")
    se = selection_columns(es, res.selected, "s")
    nf = es.first_per_commit.astype(float)
    part = ~res.full
    out = {
        "method": res.method, "seed": res.seed, "n_full": "inf" if res.n_full == math.inf else int(res.n_full),
        "commits": int(len(res.commits)), "full_run_commits": int(res.full.sum()), "first_exposures": int(nf.sum()),
        "recall_first_all": float(ex["x_first_caught"].sum() / nf.sum()) if nf.sum() else math.nan,
        "time_all": float(ex["x_time"].mean()),
        "recall_first_selected": float(se["s_first_caught"][part].sum() / nf[part].sum()) if nf[part].sum() else math.nan,
        "time_selected": float(se["s_time"][part].mean()) if part.any() else math.nan,
        "recall_test_all": float(ex["x_caught"].sum() / es.y.sum()),
    }
    # Detection delay per defect with a first exposure in the range.
    commit_of = es.commits[es.def_commit]
    detected = (es.def_mask & res.executed[es.def_commit]).any(1)
    fixed = dict(zip(defects["defect_id"].astype(str), defects["fixed_at"]))
    intro = dict(zip(defects["defect_id"].astype(str), defects["introduced_at"]))
    delays, undetected_fixed, censored = [], 0, 0
    first_ids = sorted(set(es.def_ids[es.def_first]))
    for d in first_ids:
        k = np.where(es.def_ids == d)[0]
        hit = k[detected[k]]
        if len(hit):
            delays.append(int(commit_of[hit].min()) - int(intro[d]))
        elif int(fixed[d]) < end:
            undetected_fixed += 1
        else:
            censored += 1
    out.update({
        "defects": len(first_ids), "detected": len(delays),
        "delay_mean": float(np.mean(delays)) if delays else math.nan,
        "delay_median": float(np.median(delays)) if delays else math.nan,
        "delay_max": int(max(delays)) if delays else math.nan,
        "detected_at_first_exposure": int(sum(x == 0 for x in delays)),
        "undetected_before_fix": undetected_fixed, "undetected_alive_at_end": censored,
        "share_undetected_before_fix": undetected_fixed / len(first_ids) if first_ids else math.nan,
    })
    if cold:
        cmask = np.isin(es.tests, list(cold))
        fails = (es.y & cmask[None, :]).sum()
        out["cold_test_recall"] = float((es.y & res.executed & cmask[None, :]).sum() / fails) if fails else math.nan
    return out
