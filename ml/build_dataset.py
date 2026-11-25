"""Stage 5: build the (commit, test) dataset for test prioritization.

Input (a finished history produced by `mutation/sim`, e.g. data/history or data/history-600):
  <history>/runs.jsonl   one full run of the frozen E2E suite per commit
  <history>/plan.json    the commit plan (only used to check that the run is complete)
  <history>/repo/        git repository of the history (diffs are taken from here)
  tests/manifest.json    static test metadata (modules, criticality, coversPaths, avgDurationMs)
  data/pilot/runs.jsonl, data/history/isolation.jsonl
                         single-defect runs; used ONLY for the analysis file defects.csv

Output: data/dataset/<history-name>[-<max-commits>]/
  dataset.csv   one row per (commit, test): ids, label, features (chg_/hist_/meta_/ci_), eval_ columns
  commits.csv   per-commit ANALYSIS columns (kind, introduced/open defects, ...) -- never features
  defects.csv   per-defect ANALYSIS columns (lifetime, manual or mutant, silent in isolation)
  positives.csv cause of every positive row (open defects whose single-defect run fails that test)
  splits.json   time-based hold-out split (60/20/20) and rolling-window fold indices
  meta.json     source, versions, feature groups, row counts, positive share per part, file hashes

Leakage rules (verified by ml/check_dataset.py):
  * chg_  -- only `git diff <parent> <sha>` of the commit itself: counts and paths, never line text
            (a StringLiteral mutant inserts "Stryker was here!", any token feature would leak it);
  * hist_ -- only results of commits 0..i-1 (online construction: features first, then state update);
  * meta_ -- static manifest data and durations of past passed runs;
  * ci_   -- typecheck/unit status of the CURRENT commit (they run before E2E in CI); a separate
            group for ablation, not part of the main feature set (notes/decisions.md, section 6);
  * never used: kind, message, introduced, fixed, openAfter, date, startedAt, exitCode, buildError,
            totalDurationMs, testWallMs, rerunWallMs and every per-test field of the current run.

Usage:  ml/.venv/bin/python ml/build_dataset.py --history data/history [--max-commits N] [--out DIR]
        final dataset: --history data/history-600 --max-commits 540  (-> data/dataset/history-600-540/)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "manifest.json"
SINGLE_DEFECT_RUNS = (ROOT / "data" / "pilot" / "runs.jsonl", ROOT / "data" / "history" / "isolation.jsonl")

DATASET_VERSION = 1
HIST_WINDOWS = (1, 3, 5, 10, 20)  # "failures in the last N runs"
PEER_WINDOWS = (1, 5)  # failures of same-module tests in the last N runs
SINCE_FAIL_CAP = 100  # commits since last failure is capped (also the value for "never failed")
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
ROLLING_START_FRAC, ROLLING_WINDOW_FRAC = 0.5, 0.1

CRITICALITY = {"low": 0, "medium": 1, "high": 2}
ID_PREFIX_MODULE = {
    "AUTH": "auth", "CAT": "catalog", "PRD": "product", "SRCH": "search", "CART": "cart",
    "CHK": "checkout", "ORD": "orders", "ADM": "admin", "E2E": "e2e",
}
# Files every test executes but no test declares in `covers` (notes/tests.md, section 5).
INFRA_FILES = {
    "src/app.ts", "src/server.ts", "src/shared/config.ts", "src/shared/db.ts", "src/shared/test-routes.ts",
    "src/shared/views/footer.ejs", "package.json", "package-lock.json", "tsconfig.json", "vitest.config.ts",
}
INFRA_PREFIXES = ("src/public/", "src/seed/", "scripts/")

ID_COLUMNS = ["commit_index", "sha", "test_id", "split"]
LABEL = "label"
EVAL_COLUMNS = ["eval_duration_ms", "eval_flaky"]  # current-run values: for metrics only, never features
CATEGORICAL_FEATURES = ["meta_module"]
# Fields of runs.jsonl that must never reach a feature (checked by ml/check_dataset.py).
FORBIDDEN_RUN_FIELDS = [
    "kind", "message", "introduced", "fixed", "openAfter", "date", "startedAt", "exitCode", "buildError",
    "totalDurationMs", "testWallMs", "rerunWallMs",
]
FORBIDDEN_TEST_FIELDS = ["status", "durationMs", "flaky", "confirmedFail", "rerunStatuses", "errorSnippet"]


# ----------------------------------------------------------------------------- loading

class HistoryError(RuntimeError):
    pass


def load_history(history: Path, max_commits: int | None = None) -> tuple[list[dict], dict]:
    """Load runs.jsonl and plan.json; refuse an unfinished or halted run.

    `max_commits` keeps only the first N commits of a COMPLETE run (e.g. 540 of 600, when the defect pool
    ran out at the end of the history). The cut must not split a defect's life: no defect may be open
    after the last kept commit.
    """
    if (history / "halt.json").exists():
        raise HistoryError(f"{history}/halt.json exists: the run halted, the dataset is not built")
    plan = json.loads((history / "plan.json").read_text())
    runs = [json.loads(line) for line in (history / "runs.jsonl").read_text().splitlines() if line.strip()]
    expected = len(plan["commits"])
    if len(runs) != expected:
        raise HistoryError(f"{history}/runs.jsonl has {len(runs)} runs, plan.json has {expected} commits")
    for i, run in enumerate(runs):
        if run["index"] != i:
            raise HistoryError(f"runs.jsonl line {i} has index {run['index']}")
        if i > 0 and run["parentSha"] != runs[i - 1]["sha"]:
            raise HistoryError(f"commit {i}: parentSha does not match the previous commit")
    if max_commits is not None:
        if not 0 < max_commits <= len(runs):
            raise HistoryError(f"--max-commits {max_commits} is outside 1..{len(runs)}")
        if runs[max_commits - 1]["openAfter"]:
            raise HistoryError(f"--max-commits {max_commits} cuts open defects {runs[max_commits - 1]['openAfter']}")
        runs = runs[:max_commits]
    return runs, plan


def load_tests() -> list[dict]:
    tests = json.loads(MANIFEST.read_text())["tests"]
    return sorted(tests, key=lambda t: t["id"])


# ----------------------------------------------------------------------------- git diffs

def _git(repo: Path, *args: str) -> str:
    cmd = ["git", "-C", str(repo), "-c", "core.quotepath=off", *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def read_diff(repo: Path, parent: str, sha: str) -> dict[str, dict]:
    """Per-file line and hunk counts of `git diff -U0 parent sha` (no line content is kept)."""
    files: dict[str, dict] = {}
    numstat = _git(repo, "diff", "--no-renames", "--no-color", "--no-ext-diff", "--numstat", parent, sha)
    for line in numstat.splitlines():
        added, deleted, path = line.split("\t", 2)
        files[path] = {
            "added": int(added) if added != "-" else 0,
            "deleted": int(deleted) if deleted != "-" else 0,
            "hunks": 0, "hunks_modify": 0, "hunks_add": 0, "hunks_delete": 0,
        }
    patch = _git(repo, "diff", "--no-renames", "--no-color", "--no-ext-diff", "-U0", parent, sha)
    current = None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            current = line.split(" b/", 1)[1]
        elif line.startswith("@@ ") and current is not None:
            old, new = line.split(" ")[1:3]  # "-a[,b]" "+c[,d]"
            old_n = int(old.split(",")[1]) if "," in old else 1
            new_n = int(new.split(",")[1]) if "," in new else 1
            rec = files[current]
            rec["hunks"] += 1
            if old_n and new_n:
                rec["hunks_modify"] += 1
            elif new_n:
                rec["hunks_add"] += 1
            else:
                rec["hunks_delete"] += 1
    return files


def read_all_diffs(repo: Path, runs: list[dict]) -> dict[str, dict[str, dict]]:
    """Diffs of every commit, cross-checked against the `files` field the simulator logged."""
    diffs = {}
    for run in runs:
        diff = read_diff(repo, run["parentSha"], run["sha"])
        logged = {f["path"]: (f["added"], f["deleted"]) for f in run["files"]}
        actual = {p: (r["added"], r["deleted"]) for p, r in diff.items()}
        if logged != actual:
            raise HistoryError(f"commit {run['index']}: git diff differs from runs.jsonl files: {actual} vs {logged}")
        diffs[run["sha"]] = diff
    return diffs


# ----------------------------------------------------------------------------- path helpers

def file_module(path: str) -> str | None:
    parts = path.split("/")
    if len(parts) > 3 and parts[0] == "src" and parts[1] == "modules":
        return parts[2]
    return None


def file_type(path: str) -> str:
    if path.endswith(".ts"):
        return "ts"
    if path.endswith(".ejs"):
        return "ejs"
    if path.endswith(".css"):
        return "css"
    if path.endswith(".md"):
        return "doc"
    return "other"


def is_shared(path: str) -> bool:
    return path.startswith("src/shared/")


def is_infra(path: str) -> bool:
    return path in INFRA_FILES or path.startswith(INFRA_PREFIXES)


def own_module(test_id: str) -> str:
    return ID_PREFIX_MODULE[test_id.split("-")[0]]


# ----------------------------------------------------------------------------- features

def commit_change_features(diff: dict[str, dict]) -> dict[str, float]:
    """Commit-level features from the diff only (identical for every test of the commit)."""
    paths = list(diff)
    types = [file_type(p) for p in paths]
    added = sum(r["added"] for r in diff.values())
    deleted = sum(r["deleted"] for r in diff.values())
    return {
        "chg_files": len(paths),
        "chg_lines_added": added,
        "chg_lines_deleted": deleted,
        "chg_churn": added + deleted,
        "chg_hunks": sum(r["hunks"] for r in diff.values()),
        "chg_hunks_modify": sum(r["hunks_modify"] for r in diff.values()),
        "chg_hunks_add": sum(r["hunks_add"] for r in diff.values()),
        "chg_hunks_delete": sum(r["hunks_delete"] for r in diff.values()),
        "chg_modules": len({m for p in paths if (m := file_module(p))}),
        "chg_ts_files": types.count("ts"),
        "chg_ejs_files": types.count("ejs"),
        "chg_css_files": types.count("css"),
        "chg_doc_files": types.count("doc"),
        "chg_other_files": types.count("other"),
        "chg_shared_files": sum(is_shared(p) for p in paths),
        "chg_infra_files": sum(is_infra(p) for p in paths),
    }


def test_change_features(diff: dict[str, dict], test: dict, n_files: int) -> dict[str, float]:
    """Diff x test features: overlap of the changed files with the declared `coversPaths` and modules."""
    covers = set(test["coversPaths"])
    covered = [p for p in diff if p in covers]
    in_modules = [p for p in diff if file_module(p) in set(test["modules"])]
    return {
        "chg_cov_files": len(covered),
        "chg_cov_frac_test": len(covered) / len(covers),
        "chg_cov_frac_diff": len(covered) / n_files if n_files else 0.0,
        "chg_cov_churn": sum(diff[p]["added"] + diff[p]["deleted"] for p in covered),
        "chg_cov_ts_files": sum(file_type(p) == "ts" for p in covered),
        "chg_mod_files": len(in_modules),
        "chg_same_module": int(bool(in_modules)),
    }


def run_outcome(run: dict, test_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Label (confirmedFail), flaky flag, passed flag and duration per test of one run."""
    by_id = {t["id"]: t for t in run["tests"]}
    missing = set(test_ids) - set(by_id)
    if missing:
        raise HistoryError(f"commit {run['index']}: no result for {sorted(missing)}")
    res = [by_id[tid] for tid in test_ids]
    label = np.array([bool(r["confirmedFail"]) and not r["flaky"] for r in res], dtype=np.int8)
    flaky = np.array([bool(r["flaky"]) for r in res], dtype=np.int8)
    passed = np.array([r["status"] == "passed" for r in res], dtype=bool)
    duration = np.array([float(r["durationMs"]) for r in res])
    return label, flaky, passed, duration


def build_frame(runs: list[dict], diffs: dict[str, dict[str, dict]], tests: list[dict]) -> pd.DataFrame:
    """One row per (commit, test). History state is updated only after the commit's rows are emitted."""
    test_ids = [t["id"] for t in tests]
    n_tests = len(tests)
    module_sets = [set(t["modules"]) for t in tests]
    peers = np.array([[i != j and bool(module_sets[i] & module_sets[j]) for j in range(n_tests)]
                      for i in range(n_tests)], dtype=float)
    peer_count = peers.sum(axis=1)

    # History state (results of commits strictly before the current one).
    fail_rows: list[np.ndarray] = []  # one label vector per past run that built
    last_fail = np.full(n_tests, -1)
    total_fail = np.zeros(n_tests)
    n_past = 0
    dur_sum = np.zeros(n_tests)
    dur_cnt = np.zeros(n_tests)
    manifest_dur = np.array([float(t["avgDurationMs"]) for t in tests])
    file_changes: dict[str, int] = defaultdict(int)
    file_cofail: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_tests))

    records: list[dict] = []
    for run in runs:
        i = run["index"]
        diff = diffs[run["sha"]]
        chg_commit = commit_change_features(diff)

        past = np.array(fail_rows) if fail_rows else np.zeros((0, n_tests))
        hist_last = {n: past[-n:].sum(axis=0) if len(past) else np.zeros(n_tests) for n in HIST_WINDOWS}
        peer_rate = {}
        for n in PEER_WINDOWS:
            window = past[-n:]
            if len(window):
                peer_rate[n] = (peers @ window.sum(axis=0)) / np.maximum(peer_count * len(window), 1)
            else:
                peer_rate[n] = np.zeros(n_tests)
        prev_commit_fails = float(past[-1].sum()) if len(past) else 0.0
        ever = last_fail >= 0
        since = np.where(ever, np.minimum(n_past - last_fail, SINCE_FAIL_CAP), SINCE_FAIL_CAP)
        fail_rate = total_fail / n_past if n_past else np.zeros(n_tests)
        assoc_count = np.zeros(n_tests)
        assoc_rate = np.zeros(n_tests)
        for path in diff:
            if file_changes[path]:
                assoc_count += file_cofail[path]
                assoc_rate = np.maximum(assoc_rate, file_cofail[path] / file_changes[path])
        mean_dur = np.where(dur_cnt > 0, dur_sum / np.maximum(dur_cnt, 1), manifest_dur)

        build_ok = run.get("buildOk", True)
        if build_ok:
            label, flaky, passed, duration = run_outcome(run, test_ids)
            for k, test in enumerate(tests):
                rec = {"commit_index": i, "sha": run["sha"], "test_id": test["id"], LABEL: int(label[k])}
                rec.update(chg_commit)
                rec.update(test_change_features(diff, test, chg_commit["chg_files"]))
                for n in HIST_WINDOWS:
                    rec[f"hist_fail_last_{n}"] = int(hist_last[n][k])
                rec["hist_since_fail"] = int(since[k])
                rec["hist_never_failed"] = int(not ever[k])
                rec["hist_fail_rate"] = float(fail_rate[k])
                for n in PEER_WINDOWS:
                    rec[f"hist_peer_fail_rate_{n}"] = float(peer_rate[n][k])
                rec["hist_prev_commit_fails"] = prev_commit_fails
                rec["hist_file_fail_count"] = float(assoc_count[k])
                rec["hist_file_fail_rate"] = float(assoc_rate[k])
                rec["meta_duration_mean"] = float(mean_dur[k])
                rec["meta_criticality"] = CRITICALITY[test["criticality"]]
                rec["meta_module"] = own_module(test["id"])
                rec["meta_n_modules"] = len(test["modules"])
                rec["meta_n_covers"] = len(test["coversPaths"])
                rec["ci_typecheck_ok"] = int(bool(run["typecheckOk"]))
                rec["ci_unit_ok"] = int(bool(run["unitOk"]))
                rec["eval_duration_ms"] = float(duration[k])
                rec["eval_flaky"] = int(flaky[k])
                records.append(rec)

            # ---- state update with the results of commit i (used from commit i+1 on)
            fail_rows.append(label.astype(float))
            last_fail = np.where(label > 0, n_past, last_fail)
            total_fail += label
            dur_sum += np.where(passed, duration, 0.0)
            dur_cnt += passed
            for path in diff:
                file_changes[path] += 1
                file_cofail[path] = file_cofail[path] + label
            n_past += 1
        # A commit that does not build yields no rows and no test results (none in the current histories).

    frame = pd.DataFrame.from_records(records)
    return frame


def feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {g: [c for c in frame.columns if c.startswith(g + "_")] for g in ("chg", "hist", "meta", "ci")}


def order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    groups = feature_groups(frame)
    cols = ID_COLUMNS + [LABEL] + groups["chg"] + groups["hist"] + groups["meta"] + groups["ci"] + EVAL_COLUMNS
    return frame[cols]


# ----------------------------------------------------------------------------- splits

def split_bounds(n: int) -> dict[str, list[int]]:
    """Half-open commit-index ranges of the 60/20/20 time split (no shuffling)."""
    train_end = n * int(TRAIN_FRAC * 10) // 10
    val_end = n * int((TRAIN_FRAC + VAL_FRAC) * 10) // 10
    return {"train": [0, train_end], "val": [train_end, val_end], "test": [val_end, n]}


def rolling_folds(n: int) -> dict:
    """Fold indices for rolling-window evaluation (stage 8); nothing is trained here."""
    window = max(1, int(round(n * ROLLING_WINDOW_FRAC)))
    start = int(round(n * ROLLING_START_FRAC))
    folds_expanding, folds_sliding = [], []
    for k, s in enumerate(range(start, n, window)):
        test = [s, min(s + window, n)]
        folds_expanding.append({"fold": k, "fit": [0, s - window], "val": [s - window, s], "test": test})
        folds_sliding.append({"fold": k, "fit": [s - start, s - window], "val": [s - window, s], "test": test})
    return {
        "window": window, "start": start,
        "note": "half-open [from, to) commit-index ranges; fit+val = training history before the test window",
        "expanding": folds_expanding,
        "sliding": folds_sliding,
    }


def assign_split(index: pd.Series, bounds: dict[str, list[int]]) -> pd.Series:
    out = pd.Series("", index=index.index, dtype=object)
    for name, (lo, hi) in bounds.items():
        out[(index >= lo) & (index < hi)] = name
    return out


# ----------------------------------------------------------------------------- analysis files

def load_single_defect_runs() -> dict[str, dict]:
    """Single-defect runs on the base version (pilot + isolation), keyed by defect id."""
    lookup: dict[str, dict] = {}
    for path in SINGLE_DEFECT_RUNS:
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    lookup.setdefault(rec["defectId"], rec)
    return lookup


def commits_table(runs: list[dict], frame: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    fails = frame.groupby("commit_index")[LABEL].sum()
    rows = []
    for run in runs:
        i = run["index"]
        open_after = run["openAfter"]
        rows.append({
            "commit_index": i, "sha": run["sha"], "parent_sha": run["parentSha"],
            "kind": run["kind"],
            "introduced": ";".join(run["introduced"]), "fixed": ";".join(run["fixed"]),
            "open_defects": ";".join(open_after), "n_open": len(open_after),
            "n_open_manual": sum(d.startswith("MAN-") for d in open_after),
            "typecheck_ok": int(bool(run["typecheckOk"])), "unit_ok": int(bool(run["unitOk"])),
            "build_ok": int(bool(run.get("buildOk", True))),
            "n_failed_tests": int(fails.get(i, 0)),
        })
    table = pd.DataFrame(rows)
    table.insert(3, "split", assign_split(table["commit_index"], bounds))
    return table


def defects_table(runs: list[dict], bounds: dict) -> pd.DataFrame:
    single = load_single_defect_runs()
    info: dict[str, dict] = {}
    for run in runs:
        for d in run["introduced"]:
            info[d] = {"defect_id": d, "introduced_at": run["index"], "fixed_at": None, "open_commits": 0}
        for d in run["fixed"]:
            info[d]["fixed_at"] = run["index"]
        for d in run["openAfter"]:
            info[d]["open_commits"] += 1
    rows = []
    for d, rec in info.items():
        iso = single.get(d)
        iso_fails = [t["id"] for t in iso["tests"] if t["confirmedFail"]] if iso else None
        rec.update({
            "source": "manual" if d.startswith("MAN-") else "mutant",
            "isolated_fail_tests": None if iso_fails is None else ";".join(iso_fails),
            "isolated_detected": None if iso_fails is None else int(bool(iso_fails)),
        })
        rows.append(rec)
    table = pd.DataFrame(rows).sort_values("introduced_at", kind="stable").reset_index(drop=True)
    table["fixed_at"] = table["fixed_at"].astype("Int64")
    table["isolated_detected"] = table["isolated_detected"].astype("Int64")
    table.insert(2, "split", assign_split(table["introduced_at"], bounds))
    return table


def positives_table(runs: list[dict], frame: pd.DataFrame, defects: pd.DataFrame) -> pd.DataFrame:
    """Cause of every positive row: open defects whose single-defect run fails the same test.

    There is no masking between defects (notes/simulator.md, section 11), so this attribution is exact;
    a positive with no cause would point to an interaction or a flaky failure (counted in meta.json).
    """
    fail_sets = {r.defect_id: set(r.isolated_fail_tests.split(";")) - {""}
                 if isinstance(r.isolated_fail_tests, str) else set() for r in defects.itertuples()}
    open_by_commit = {run["index"]: run["openAfter"] for run in runs}
    rows = []
    for rec in frame.loc[frame[LABEL] == 1, ["commit_index", "test_id", "split"]].itertuples(index=False):
        causes = [d for d in open_by_commit[rec.commit_index] if rec.test_id in fail_sets.get(d, set())]
        sources = {"manual" if d.startswith("MAN-") else "mutant" for d in causes}
        rows.append({
            "commit_index": rec.commit_index, "test_id": rec.test_id, "split": rec.split,
            "causes": ";".join(causes),
            "cause_source": "+".join(sorted(sources)) if sources else "unexplained",
        })
    return pd.DataFrame(rows, columns=["commit_index", "test_id", "split", "causes", "cause_source"])


def split_stats(frame: pd.DataFrame, commits: pd.DataFrame, defects: pd.DataFrame,
                positives: pd.DataFrame) -> dict:
    stats = {}
    for name in ("train", "val", "test", "all"):
        f = frame if name == "all" else frame[frame["split"] == name]
        c = commits if name == "all" else commits[commits["split"] == name]
        d = defects if name == "all" else defects[defects["split"] == name]
        p = positives if name == "all" else positives[positives["split"] == name]
        manual_commits = set(c.loc[c["n_open_manual"] > 0, "commit_index"])
        new_fail = (f[LABEL] == 1) & (f["hist_fail_last_1"] == 0)
        stats[name] = {
            "commits": int(len(c)),
            "rows": int(len(f)),
            "positives": int(f[LABEL].sum()),
            "positive_share": round(float(f[LABEL].mean()), 4) if len(f) else None,
            "commits_with_fail": int((c["n_failed_tests"] > 0).sum()),
            "commits_with_open_defect": int((c["n_open"] > 0).sum()),
            "commits_with_open_defect_no_fail": int(((c["n_open"] > 0) & (c["n_failed_tests"] == 0)).sum()),
            "commits_unit_failed": int((c["unit_ok"] == 0).sum()),
            "defects_introduced": int(len(d)),
            "defects_manual": int((d["source"] == "manual").sum()),
            "defects_silent_in_isolation": int((d["isolated_detected"] == 0).sum()),
            "commits_with_manual_defect": len(manual_commits),
            "positives_new": int(new_fail.sum()),  # the test did not fail on the previous commit
            "positives_by_cause": {k: int(v) for k, v in p["cause_source"].value_counts().sort_index().items()},
        }
    return stats


# ----------------------------------------------------------------------------- main

def build(history: Path, max_commits: int | None = None) -> tuple[pd.DataFrame, list[dict], dict, list[dict], dict]:
    runs, plan = load_history(history, max_commits)
    tests = load_tests()
    diffs = read_all_diffs(history / "repo", runs)
    frame = build_frame(runs, diffs, tests)
    frame["split"] = assign_split(frame["commit_index"], split_bounds(len(runs)))
    return order_columns(frame), runs, plan, tests, diffs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_out(history: Path, max_commits: int | None) -> Path:
    name = history.name if max_commits is None else f"{history.name}-{max_commits}"
    return ROOT / "data" / "dataset" / name


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.10g")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", required=True, type=Path, help="history directory, e.g. data/history")
    parser.add_argument("--max-commits", type=int, help="keep only the first N commits of a complete history")
    parser.add_argument("--out", type=Path,
                        help="output directory (default data/dataset/<history-name>[-<max-commits>])")
    args = parser.parse_args()
    history = args.history.resolve()
    out = (args.out or default_out(history, args.max_commits)).resolve()

    try:
        frame, runs, plan, tests, _ = build(history, args.max_commits)
    except HistoryError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    bounds = split_bounds(len(runs))
    commits = commits_table(runs, frame, bounds)
    defects = defects_table(runs, bounds)
    positives = positives_table(runs, frame, defects)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(frame, out / "dataset.csv")
    write_csv(commits, out / "commits.csv")
    write_csv(defects, out / "defects.csv")
    write_csv(positives, out / "positives.csv")
    splits = {"holdout": bounds, "rolling": rolling_folds(len(runs))}
    (out / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")

    groups = feature_groups(frame)
    history_meta = json.loads((history / "meta.json").read_text()) if (history / "meta.json").exists() else {}
    meta = {
        "datasetVersion": DATASET_VERSION,
        "source": {
            "history": str(history.relative_to(ROOT)),
            "planSeed": plan["seed"], "poolMode": plan.get("poolMode"), "defectRate": plan.get("defectRate"),
            "baseSha": plan["base"], "headSha": runs[-1]["sha"],
            "commitsInHistory": len(plan["commits"]), "maxCommits": args.max_commits,
            "runEnvironment": history_meta,
            "runsSha256": sha256(history / "runs.jsonl"),
            "manifestSha256": sha256(MANIFEST),
        },
        "builder": {
            "script": "ml/build_dataset.py",
            "python": platform.python_version(),
            "pandas": pd.__version__, "numpy": np.__version__,
            "git": _git(history / "repo", "--version").strip(),
            "deterministic": "no randomness in the build; the same inputs give byte-identical CSV files",
        },
        "parameters": {
            "histWindows": list(HIST_WINDOWS), "peerWindows": list(PEER_WINDOWS),
            "sinceFailCap": SINCE_FAIL_CAP, "splitFractions": [TRAIN_FRAC, VAL_FRAC, 1 - TRAIN_FRAC - VAL_FRAC],
        },
        "label": "confirmedFail of the test in the commit's run (failed in the run and both re-runs; flaky -> 0)",
        "columns": {
            "id": ID_COLUMNS, "label": LABEL, "eval": EVAL_COLUMNS,
            "featureGroups": groups,
            "mainFeatureSet": groups["chg"] + groups["hist"] + groups["meta"],
            "ablationOnly": groups["ci"],
            "categorical": CATEGORICAL_FEATURES,
        },
        "forbiddenRunFields": FORBIDDEN_RUN_FIELDS + ["typecheckOk/unitOk outside ci_*"],
        "forbiddenTestFields": FORBIDDEN_TEST_FIELDS,
        "counts": {"commits": len(runs), "tests": len(tests), "rows": int(len(frame)),
                   "features": sum(len(v) for v in groups.values())},
        "splits": split_stats(frame, commits, defects, positives),
        "files": {name: sha256(out / name)
                  for name in ("dataset.csv", "commits.csv", "defects.csv", "positives.csv", "splits.json")},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    print(f"wrote {out.relative_to(ROOT)}: {len(frame)} rows, {meta['counts']['features']} features")
    for name, s in meta["splits"].items():
        print(f"  {name:5s} commits={s['commits']:3d} rows={s['rows']:5d} pos={s['positives']:4d} "
              f"({s['positive_share']:.4f}) red_commits={s['commits_with_fail']} manual={s['defects_manual']} "
              f"new={s['positives_new']} causes={s['positives_by_cause']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
