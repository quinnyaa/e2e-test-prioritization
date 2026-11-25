"""Stage 7: online feature computation for a NEW commit (diff -> 36 rows with the dataset.csv columns).

`ml/build_dataset.py` builds the dataset in one pass and needs the results of the current run to emit a
row, so it cannot be called for a commit that has not been run yet. `HistoryState` keeps the same
history state (results of commits 0..i-1) and splits the pass into two steps:

    rows = state.features(diff, commit_index, sha)   # before the E2E run: chg_ + hist_ + meta_ (+ ci_)
    state.update(diff, run)                          # after a FULL run of the suite (results of commit i)

The diff helpers and constants are imported from build_dataset.py (not copied), and the state update
mirrors `build_frame` line by line. `ml/check_cli.py` proves the parity: replaying commits 0..431 of
data/history-600 gives exactly the rows of data/dataset/history-600-540/dataset.csv.

Selection feedback (notes/stage7-plan.md, R4): only results of FULL runs may be passed to `update`.
A run of the selected tests only would make the history features differ from the ones the model was
trained on (every past commit ran the whole suite), so `update` refuses an incomplete run.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import (
    CRITICALITY,
    HIST_WINDOWS,
    MANIFEST,
    PEER_WINDOWS,
    SINCE_FAIL_CAP,
    HistoryError,
    commit_change_features,
    load_tests,
    own_module,
    read_diff,
    run_outcome,
    test_change_features,
)

ROOT = Path(__file__).resolve().parent.parent
Diff = dict[str, dict]  # path -> {"added", "deleted", "hunks", "hunks_modify", "hunks_add", "hunks_delete"}


class IncompleteRunError(HistoryError):
    """A run without a result for every test of the suite (e.g. only the selected tests)."""


class HistoryState:
    """History state after the commits passed to `update` (mirrors ml/build_dataset.py::build_frame)."""

    def __init__(self, tests: list[dict] | None = None):
        self.tests = tests if tests is not None else load_tests()
        self.test_ids = [t["id"] for t in self.tests]
        n = len(self.tests)
        module_sets = [set(t["modules"]) for t in self.tests]
        self.peers = np.array([[i != j and bool(module_sets[i] & module_sets[j]) for j in range(n)]
                               for i in range(n)], dtype=float)
        self.peer_count = self.peers.sum(axis=1)
        self.fail_rows: list[np.ndarray] = []
        self.last_fail = np.full(n, -1)
        self.total_fail = np.zeros(n)
        self.n_past = 0
        self.dur_sum = np.zeros(n)
        self.dur_cnt = np.zeros(n)
        self.manifest_dur = np.array([float(t["avgDurationMs"]) for t in self.tests])
        self.file_changes: dict[str, int] = defaultdict(int)
        self.file_cofail: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n))
        self.commits_seen = 0  # commits passed to `update` (built or not); = index of the next commit
        self.last_sha: str | None = None

    # ------------------------------------------------------------------ features (before the run)

    def features(self, diff: Diff, commit_index: int, sha: str, ci: dict | None = None) -> pd.DataFrame:
        """36 rows (sorted by test_id) with the id and feature columns of dataset.csv.

        `ci` = {"typecheckOk": bool, "unitOk": bool} of the current commit, if known; the ci_ group is
        not in the main feature set (notes/decisions.md, p. 6), so it is NaN when unknown.
        """
        n = len(self.tests)
        chg_commit = commit_change_features(diff)
        past = np.array(self.fail_rows) if self.fail_rows else np.zeros((0, n))
        hist_last = {k: past[-k:].sum(axis=0) if len(past) else np.zeros(n) for k in HIST_WINDOWS}
        peer_rate = {}
        for k in PEER_WINDOWS:
            window = past[-k:]
            if len(window):
                peer_rate[k] = (self.peers @ window.sum(axis=0)) / np.maximum(self.peer_count * len(window), 1)
            else:
                peer_rate[k] = np.zeros(n)
        prev_commit_fails = float(past[-1].sum()) if len(past) else 0.0
        ever = self.last_fail >= 0
        since = np.where(ever, np.minimum(self.n_past - self.last_fail, SINCE_FAIL_CAP), SINCE_FAIL_CAP)
        fail_rate = self.total_fail / self.n_past if self.n_past else np.zeros(n)
        assoc_count = np.zeros(n)
        assoc_rate = np.zeros(n)
        for path in diff:
            if self.file_changes[path]:
                assoc_count += self.file_cofail[path]
                assoc_rate = np.maximum(assoc_rate, self.file_cofail[path] / self.file_changes[path])
        mean_dur = np.where(self.dur_cnt > 0, self.dur_sum / np.maximum(self.dur_cnt, 1), self.manifest_dur)

        records = []
        for k, test in enumerate(self.tests):
            rec: dict = {"commit_index": commit_index, "sha": sha, "test_id": test["id"]}
            rec.update(chg_commit)
            rec.update(test_change_features(diff, test, chg_commit["chg_files"]))
            for w in HIST_WINDOWS:
                rec[f"hist_fail_last_{w}"] = int(hist_last[w][k])
            rec["hist_since_fail"] = int(since[k])
            rec["hist_never_failed"] = int(not ever[k])
            rec["hist_fail_rate"] = float(fail_rate[k])
            for w in PEER_WINDOWS:
                rec[f"hist_peer_fail_rate_{w}"] = float(peer_rate[w][k])
            rec["hist_prev_commit_fails"] = prev_commit_fails
            rec["hist_file_fail_count"] = float(assoc_count[k])
            rec["hist_file_fail_rate"] = float(assoc_rate[k])
            rec["meta_duration_mean"] = float(mean_dur[k])
            rec["meta_criticality"] = CRITICALITY[test["criticality"]]
            rec["meta_module"] = own_module(test["id"])
            rec["meta_n_modules"] = len(test["modules"])
            rec["meta_n_covers"] = len(test["coversPaths"])
            rec["ci_typecheck_ok"] = int(bool(ci["typecheckOk"])) if ci else np.nan
            rec["ci_unit_ok"] = int(bool(ci["unitOk"])) if ci else np.nan
            records.append(rec)
        frame = pd.DataFrame.from_records(records)
        frame["meta_module"] = frame["meta_module"].astype(object)
        return as_dataset_values(frame)

    # ------------------------------------------------------------------ state update (after a FULL run)

    def update(self, diff: Diff, run: dict) -> None:
        """Add the results of one commit. `run` has the runs.jsonl shape: buildOk + tests[] of ALL tests."""
        self.commits_seen += 1
        self.last_sha = run.get("sha", self.last_sha)
        if not run.get("buildOk", True):
            return  # a commit that does not build adds no results (as in build_frame)
        ids = {t["id"] for t in run["tests"]}
        missing = set(self.test_ids) - ids
        if missing:
            raise IncompleteRunError(
                f"run of {run.get('sha', '?')[:10]} has no result for {len(missing)} test(s) {sorted(missing)[:5]}...; "
                "only full runs update the history (selection feedback, notes/stage7-plan.md, R4)"
            )
        label, _flaky, passed, duration = run_outcome(run, self.test_ids)
        self.fail_rows.append(label.astype(float))
        self.last_fail = np.where(label > 0, self.n_past, self.last_fail)
        self.total_fail += label
        self.dur_sum += np.where(passed, duration, 0.0)
        self.dur_cnt += passed
        for path in diff:
            self.file_changes[path] += 1
            self.file_cofail[path] = self.file_cofail[path] + label
        self.n_past += 1


CSV_FLOAT_FORMAT = "%.10g"  # ml/build_dataset.py::write_csv


def as_dataset_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Round float columns as dataset.csv stores them (10 significant digits), so that the model gets
    exactly the values it was trained and validated on."""
    out = frame.copy()
    for c in out.columns:
        if out[c].dtype == float:
            out[c] = [float(CSV_FLOAT_FORMAT % v) if v == v else v for v in out[c]]
    return out


# ---------------------------------------------------------------------- history loading and replay

def load_runs(history: Path, upto: int | None = None) -> list[dict]:
    """runs.jsonl of a history directory (first `upto` commits), with the parent chain checked."""
    runs = [json.loads(line) for line in (history / "runs.jsonl").read_text().splitlines() if line.strip()]
    if upto is not None:
        if not 0 < upto <= len(runs):
            raise HistoryError(f"--upto {upto} is outside 1..{len(runs)}")
        runs = runs[:upto]
    for i, run in enumerate(runs):
        if run["index"] != i or (i > 0 and run["parentSha"] != runs[i - 1]["sha"]):
            raise HistoryError(f"{history}/runs.jsonl: broken commit chain at line {i}")
    return runs


def load_extra_runs(path: Path | None) -> list[dict]:
    """Full runs recorded by `ml/cli.py record` (each carries its own diff)."""
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def diff_cache_path(history: Path) -> Path:
    return ROOT / "data" / "cache" / f"diffs-{history.name}.json"


def history_diffs(history: Path, runs: list[dict], use_cache: bool = True) -> list[Diff]:
    """Per-commit diffs of the history repository (~2 git calls per commit; cached by SHA).

    The cache (data/cache/, reproducible, safe to delete) holds only per-file line/hunk counts, as
    build_dataset.py does; every diff is cross-checked against the `files` field of runs.jsonl.
    """
    cache = diff_cache_path(history)
    known: dict[str, Diff] = {}
    if use_cache and cache.exists():
        known = json.loads(cache.read_text()).get("diffs", {})
    diffs, added = [], False
    for run in runs:
        key = f"{run['parentSha']}..{run['sha']}"
        diff = known.get(key)
        if diff is None:
            diff = read_diff(history / "repo", run["parentSha"], run["sha"])
            known[key], added = diff, True
        logged = {f["path"]: (f["added"], f["deleted"]) for f in run["files"]}
        if logged != {p: (r["added"], r["deleted"]) for p, r in diff.items()}:
            raise HistoryError(f"commit {run['index']}: git diff differs from runs.jsonl files")
        diffs.append(diff)
    if use_cache and added:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"history": str(history), "diffs": known}))
    return diffs


def replay(history: Path, upto: int, extra_runs: list[dict] | None = None, use_cache: bool = True,
           emit: bool = False) -> tuple[HistoryState, pd.DataFrame | None]:
    """State after commits 0..upto-1 of `history` (+ recorded full runs). With `emit`, also returns the
    online feature rows of every replayed history commit (used by the parity check)."""
    runs = load_runs(history, upto)
    diffs = history_diffs(history, runs, use_cache)
    state = HistoryState()
    frames = []
    for run, diff in zip(runs, diffs):
        if emit and run.get("buildOk", True):
            frames.append(state.features(diff, run["index"], run["sha"],
                                         {"typecheckOk": run["typecheckOk"], "unitOk": run["unitOk"]}))
        state.update(diff, run)
    for rec in extra_runs or []:
        if rec["parentSha"] != state.last_sha:
            raise HistoryError(f"recorded run {rec['sha'][:10]}: parent {rec['parentSha'][:10]} is not the "
                               f"last commit of the history {str(state.last_sha)[:10]}")
        state.update(rec["diff"], rec)
    return state, (pd.concat(frames, ignore_index=True) if emit else None)


def manifest_sha256() -> str:
    import hashlib
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def diff_between(repo: Path, base: str, head: str) -> Diff:
    """Per-file counts of `git diff base head` (same parser as the dataset)."""
    return read_diff(repo, base, head)
