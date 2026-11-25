"""Stage 5: leakage and consistency checks of the dataset built by ml/build_dataset.py.

Checks (all rebuild the features in memory from the same inputs; the CSV on disk is compared too):
  1. whitelist      -- feature columns use only allowed prefixes, no forbidden run/test field names;
  2. determinism    -- an in-memory rebuild is byte-identical to dataset.csv (sha256 from meta.json);
  3. prefix         -- for cut points k: results of commits >= k are randomized (labels, statuses,
                       durations, flaky, rerun data, typecheck/unit, run timings); every feature of
                       commits < k and every non-ci_ feature of commit k must stay identical.
                       A truncated history (runs[:k+1]) must give identical rows for commits <= k;
  4. metadata       -- kind, message, introduced, fixed, openAfter, dates, errorSnippet, ... are replaced
                       by garbage on all commits; no feature may change;
  4b. attribution   -- every introduced defect has a single-defect run; each commit's failing set equals
                       the union of its open defects' single-defect fail sets (no unexplained failures;
                       masked failures are reported);
  5. diff-only     -- commit-level chg_ features only: can a simple classifier separate commits that
                       introduce a defect / have an open defect? Trained on the train part, evaluated on
                       val+test (ROC-AUC). AUC close to 1 would mean the diff form leaks the commit type;
  6. signal         -- univariate ROC-AUC of every feature on the train part (for notes/dataset.md).

Output: data/dataset/<history-name>/checks.json; exit code 1 if a hard check fails.
Usage:  ml/.venv/bin/python ml/check_dataset.py --history data/history [--max-commits N]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd  # noqa: E402

SEED = 20261117
DIFF_AUC_ALARM = 0.95  # a diff-only commit classifier above this would indicate leakage through diff form
ALLOWED_PREFIXES = ("chg_", "hist_", "meta_", "ci_")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith(ALLOWED_PREFIXES)]


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    frame.to_csv(buf, index=False, lineterminator="\n", float_format="%.10g")
    return buf.getvalue().encode()


def rebuild(runs: list[dict], diffs: dict, tests: list[dict]) -> pd.DataFrame:
    frame = bd.build_frame(runs, diffs, tests)
    frame["split"] = bd.assign_split(frame["commit_index"], bd.split_bounds(len(runs)))
    return bd.order_columns(frame)


# ----------------------------------------------------------------------------- perturbations

def randomize_results(run: dict, rng: np.random.Generator) -> None:
    """Replace everything the run observed about tests and checks by random values."""
    for t in run["tests"]:
        fail = bool(rng.random() < 0.3)
        flaky = bool(not fail and rng.random() < 0.1)
        t["confirmedFail"] = fail
        t["flaky"] = flaky
        t["status"] = "failed" if (fail or flaky) else "passed"
        t["durationMs"] = int(rng.integers(50, 20000))
        t["rerunStatuses"] = ["failed", "failed"] if fail else []
        t["errorSnippet"] = "random" if fail else None
    run["typecheckOk"] = bool(rng.random() < 0.5)
    run["unitOk"] = bool(rng.random() < 0.5)
    for key in ("exitCode", "totalDurationMs", "testWallMs", "rerunWallMs"):
        run[key] = int(rng.integers(0, 100000))


def garble_metadata(run: dict, rng: np.random.Generator) -> None:
    """Replace commit metadata that must never reach a feature."""
    run["kind"] = str(rng.choice(["defect", "fix", "feature", "zzz"]))
    run["message"] = f"garbage {rng.integers(1e9)}"
    run["introduced"] = [f"MUT-{rng.integers(1e6)}"]
    run["fixed"] = [f"MAN-{rng.integers(1e6)}"]
    run["openAfter"] = [f"MUT-{rng.integers(1e6)}"] * int(rng.integers(0, 5))
    run["date"] = run["startedAt"] = "1970-01-01T00:00:00.000Z"
    run["buildError"] = None
    for t in run["tests"]:
        t["errorSnippet"] = "garbage" if t["errorSnippet"] else None
        t["rerunStatuses"] = list(t["rerunStatuses"])[::-1]


def compare(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> list[str]:
    """Columns whose values differ between two row-aligned frames."""
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    if len(a) != len(b):
        return ["<row count>"]
    return [c for c in cols if not a[c].equals(b[c])]


# ----------------------------------------------------------------------------- checks

def check_attribution(runs: list[dict], frame: pd.DataFrame) -> dict:
    """No masking: each commit's failing set equals the union of its open defects' single-defect fail sets.

    Both directions are checked: no unexplained failure (actual minus union) and no masked failure
    (union minus actual). Every introduced defect must have a single-defect record.
    """
    single = bd.load_single_defect_runs()
    introduced = {d for run in runs for d in run["introduced"]}
    missing = sorted(d for d in introduced if d not in single)
    fail_sets = {d: {t["id"] for t in single[d]["tests"] if t["confirmedFail"]} for d in introduced if d in single}
    actual = frame[frame[bd.LABEL] == 1].groupby("commit_index")["test_id"].apply(set).to_dict()
    unexplained, masked = [], []
    for run in runs:
        i = run["index"]
        expected = set().union(*(fail_sets.get(d, set()) for d in run["openAfter"]))
        got = actual.get(i, set())
        if got - expected:
            unexplained.append({"commit": i, "tests": sorted(got - expected)})
        if expected - got:
            masked.append({"commit": i, "tests": sorted(expected - got)})
    return {"ok": not missing and not unexplained, "missingSingleDefectRuns": missing,
            "unexplainedCommits": unexplained, "maskedCommits": masked,
            "note": "maskedCommits is informational: masking is a property of the data, not a dataset error"}


def check_whitelist(frame: pd.DataFrame) -> dict:
    feats = feature_columns(frame)
    others = [c for c in frame.columns if c not in feats]
    expected_others = bd.ID_COLUMNS + [bd.LABEL] + bd.EVAL_COLUMNS
    lowered = {c.lower().replace("_", "") for c in feats}
    forbidden = [f for f in bd.FORBIDDEN_RUN_FIELDS + bd.FORBIDDEN_TEST_FIELDS + ["defectId"]
                 if any(f.lower() in c for c in lowered)]
    ok = sorted(others) == sorted(expected_others) and not forbidden
    return {"ok": ok, "features": len(feats), "nonFeatureColumns": others, "forbiddenNameHits": forbidden}


def check_prefix(runs, diffs, tests, full: pd.DataFrame, cuts: list[int]) -> dict:
    feats = feature_columns(full)
    non_ci = [c for c in feats if not c.startswith("ci_")]
    rng = np.random.default_rng(SEED)
    results = []
    for k in cuts:
        perturbed = copy.deepcopy(runs)
        for run in perturbed[k:]:
            randomize_results(run, rng)
        frame = rebuild(perturbed, diffs, tests)
        before = compare(full[full.commit_index < k], frame[frame.commit_index < k], feats)
        at_k = compare(full[full.commit_index == k], frame[frame.commit_index == k], non_ci)
        at_k_ci = compare(full[full.commit_index == k], frame[frame.commit_index == k], [c for c in feats if c.startswith("ci_")])
        truncated = rebuild(runs[: k + 1], diffs, tests)
        trunc_diff = compare(full[full.commit_index <= k], truncated, feats)
        results.append({
            "cut": k, "changedBefore": before, "changedAtCutNonCi": at_k,
            "changedAtCutCi": at_k_ci, "truncationDiff": trunc_diff,
        })
    ok = all(not r["changedBefore"] and not r["changedAtCutNonCi"] and not r["truncationDiff"] for r in results)
    return {"ok": ok, "cuts": results,
            "note": "changedAtCutCi is expected to be non-empty: ci_ uses the current commit's unit/typecheck run"}


def check_metadata(runs, diffs, tests, full: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED + 1)
    garbled = copy.deepcopy(runs)
    for run in garbled:
        garble_metadata(run, rng)
    frame = rebuild(garbled, diffs, tests)
    changed = compare(full, frame, feature_columns(full) + [bd.LABEL])
    return {"ok": not changed, "changedColumns": changed}


def auc(y: np.ndarray, score: np.ndarray) -> float | None:
    return round(float(roc_auc_score(y, score)), 4) if 0 < y.sum() < len(y) else None


def check_diff_only(frame: pd.DataFrame, commits: pd.DataFrame, bounds: dict) -> dict:
    commit_feats = [c for c in feature_columns(frame) if c.startswith("chg_")
                    and frame.groupby("commit_index")[c].nunique().max() == 1]
    x = frame.groupby("commit_index")[commit_feats].first().sort_index()
    c = commits.set_index("commit_index").sort_index()
    targets = {
        "introducesDefect": (c["introduced"].fillna("") != "").astype(int).to_numpy(),
        "hasOpenDefect": (c["n_open"] > 0).astype(int).to_numpy(),
        "hasFailingTest": (c["n_failed_tests"] > 0).astype(int).to_numpy(),
    }
    # Pure benign commits vs commits that introduce a defect (fix commits and commits with an older
    # open defect excluded): the cleanest test of "does the diff form give the defect away".
    pure = ((c["introduced"].fillna("") != "") | ((c["n_open"] == 0) & (c["fixed"].fillna("") == ""))).to_numpy()
    train_mask = (x.index < bounds["train"][1])
    models = {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "tree_d3": lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEED),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                         class_weight="balanced", random_state=SEED, n_jobs=1),
    }
    out = {"features": commit_feats, "trainCommits": int(train_mask.sum()), "evalCommits": int((~train_mask).sum()),
           "targets": {}}
    worst = 0.0
    for tname, y in targets.items():
        for subset_name, subset in (("all", np.ones(len(y), bool)), ("introVsBenign", pure)):
            if tname != "introducesDefect" and subset_name != "all":
                continue
            tr, ev = train_mask & subset, (~train_mask) & subset
            res = {"positivesTrain": int(y[tr].sum()), "positivesEval": int(y[ev].sum()), "nEval": int(ev.sum())}
            for mname, factory in models.items():
                model = factory().fit(x[tr], y[tr])
                res[mname] = {"trainAuc": auc(y[tr], model.predict_proba(x[tr])[:, 1]),
                              "evalAuc": auc(y[ev], model.predict_proba(x[ev])[:, 1])}
                worst = max(worst, res[mname]["evalAuc"] or 0.0)
            out["targets"][f"{tname}/{subset_name}"] = res
    # Rough rules from notes/simulator.md section 7, now on real git diffs (all 300 commits).
    intro = targets["introducesDefect"].astype(bool)
    rules = {"hasDeleteOnlyHunk": (x["chg_hunks_delete"] > 0).to_numpy(),
             "hasModifyHunk": (x["chg_hunks_modify"] > 0).to_numpy(),
             "hasDeletedLines": (x["chg_lines_deleted"] > 0).to_numpy()}
    out["rules"] = {
        name: {"tp": int((hit & intro).sum()), "fp": int((hit & ~intro).sum()), "fn": int((~hit & intro).sum()),
               "precision": round(float((hit & intro).sum() / max(hit.sum(), 1)), 4)}
        for name, hit in rules.items()
    }
    # Which diff features carry the "introduces a defect" signal (random forest, introVsBenign subset).
    subsets = {
        "allCommitFeatures": commit_feats,
        "withoutHunkTypes": [f for f in commit_feats if not f.startswith("chg_hunks")],
        "lineCountsOnly": ["chg_files", "chg_lines_added", "chg_lines_deleted", "chg_churn"],
        "deletedLinesOnly": ["chg_lines_deleted"],
    }
    tr, ev = train_mask & pure, (~train_mask) & pure
    y = targets["introducesDefect"]
    out["introVsBenignBySubset"] = {
        name: auc(y[ev], models["random_forest"]().fit(x.loc[tr, cols], y[tr]).predict_proba(x.loc[ev, cols])[:, 1])
        for name, cols in subsets.items()
    }
    out["introducingCommits"] = int(intro.sum())
    out["maxEvalAuc"] = round(worst, 4)
    out["ok"] = worst < DIFF_AUC_ALARM
    return out


def check_new_failures(frame: pd.DataFrame) -> dict:
    """Row-level ROC-AUC on val+test rows where the test did NOT fail on the previous commit.

    There `hist_fail_last_1` is useless, so this is where a diff shortcut could inflate ML over the
    history baseline (input for the stage-8 ablation; not a pass/fail check).
    """
    feats = feature_columns(frame)
    groups = {
        "chg_all": [c for c in feats if c.startswith("chg_")],
        "chg_overlap_only": [c for c in feats if c.startswith(("chg_cov_", "chg_mod_", "chg_same_"))],
        "chg_commit_only": [c for c in feats if c.startswith("chg_")
                            and not c.startswith(("chg_cov_", "chg_mod_", "chg_same_"))],
        "hist": [c for c in feats if c.startswith("hist_")],
    }
    train = frame[frame["split"] == "train"]
    ev = frame[(frame["split"] != "train") & (frame["hist_fail_last_1"] == 0)]
    out = {"evalRows": int(len(ev)), "evalPositives": int(ev[bd.LABEL].sum())}
    for name, cols in groups.items():
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                       random_state=SEED, n_jobs=1).fit(train[cols], train[bd.LABEL])
        out[name] = auc(ev[bd.LABEL].to_numpy(), model.predict_proba(ev[cols])[:, 1])
    return out


def check_signal(frame: pd.DataFrame) -> dict:
    train = frame[frame["split"] == "train"]
    y = train[bd.LABEL].to_numpy()
    res = {}
    for col in feature_columns(frame):
        if col in bd.CATEGORICAL_FEATURES:
            continue
        a = auc(y, train[col].to_numpy(dtype=float))
        res[col] = a
    return dict(sorted(res.items(), key=lambda kv: -abs((kv[1] or 0.5) - 0.5)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--max-commits", type=int, help="same value as given to build_dataset.py")
    parser.add_argument("--out", type=Path, help="dataset directory (default as in build_dataset.py)")
    args = parser.parse_args()
    history = args.history.resolve()
    out = (args.out or bd.default_out(history, args.max_commits)).resolve()

    meta = json.loads((out / "meta.json").read_text())
    full, runs, _, tests, diffs = bd.build(history, args.max_commits)
    on_disk = (out / "dataset.csv").read_bytes()
    rebuilt = to_csv_bytes(full)
    n = len(runs)
    bounds = bd.split_bounds(n)
    commits = pd.read_csv(out / "commits.csv", keep_default_na=False, na_values=[])
    commits = commits.replace("", np.nan)
    cuts = sorted({1, 2, 10, n // 4, bounds["train"][1] - 1, bounds["train"][1], bounds["val"][1], n - 1}
                  | set(int(v) for v in np.random.default_rng(SEED).integers(1, n, 5)))

    report = {
        "seed": SEED, "sklearn": sklearn.__version__, "history": meta["source"]["history"],
        "whitelist": check_whitelist(full),
        "determinism": {
            "ok": hashlib.sha256(rebuilt).hexdigest() == meta["files"]["dataset.csv"] == hashlib.sha256(on_disk).hexdigest(),
            "sha256": hashlib.sha256(rebuilt).hexdigest(),
        },
        "prefix": check_prefix(runs, diffs, tests, full, cuts),
        "metadata": check_metadata(runs, diffs, tests, full),
        "attribution": check_attribution(runs, full),
        "diffOnly": check_diff_only(full, commits, bounds),
        "newFailuresRowAuc": check_new_failures(full),
        "univariateAucTrain": check_signal(full),
    }
    hard = ("whitelist", "determinism", "prefix", "metadata", "attribution")
    report["ok"] = all(report[k]["ok"] for k in hard)
    report["diffOnlyAlarm"] = not report["diffOnly"]["ok"]
    (out / "checks.json").write_text(json.dumps(report, indent=2) + "\n")

    for k in hard:
        print(f"{k:12s} {'ok' if report[k]['ok'] else 'FAIL'}")
    print(f"{'diffOnly':12s} {'ok' if report['diffOnly']['ok'] else 'WARN (soft check, see checks.json)'}")
    print(f"cuts: {cuts}; masked commits: {len(report['attribution']['maskedCommits'])}")
    d = report["diffOnly"]
    for t, r in d["targets"].items():
        print(f"  {t:28s} " + "  ".join(f"{m}={r[m]['evalAuc']}" for m in ("logreg", "tree_d3", "random_forest")))
    print("  rules:", d["rules"])
    print("  introVsBenign by feature subset (random forest):", d["introVsBenignBySubset"])
    print("new failures (val+test, hist_fail_last_1 == 0), row AUC:", report["newFailuresRowAuc"])
    print("top univariate AUC (train):", list(report["univariateAucTrain"].items())[:12])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
