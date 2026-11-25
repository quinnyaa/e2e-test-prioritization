"""Stage 8C: helpers shared by the additional experiments `ml/exp_*.py` (notes/stage8-protocol.md, p. 5).

Nothing here defines a new metric: prioritization metrics come from metrics.evaluate / summarize, selection
metrics from policy.py (threshold_for_recall, budget_curve, cheapest_for_target, _orders, _mean_selection,
_aggregate), statistics from stats.py. The test part (432-539) was opened once in 8B; 8C reads it with
allow_test=True but never tunes on the range it evaluates.

Results of experiment <name> go to ml/results/stage8/<name>/ (never into main/); refitted models are not
saved in ml/models/.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dataset_io import ROOT, Dataset, assert_non_negative, drop_constant, load_dataset, sha256
from metrics import N_PERM, _cum_by_test, build_eval_set, classification_diagnostics, evaluate, summarize, tie_keys
from policy import _aggregate, _mean_selection, _orders, budget_curve, cheapest_for_target, threshold_for_recall
from train import SEEDS, fit_pipeline, library_versions, make_pipeline, predict

RESULTS = ROOT / "ml" / "results" / "stage8"
MAIN = RESULTS / "main"
MODELS = ROOT / "ml" / "models"
TARGET = 0.90
CRITERION = "first"
COMBINED_B = 0.34  # combined, first exposure, target 0.90 (stage 6 policies.csv; protocol p. 1)
COVERS_B = 0.36

RF_CARD = json.loads((MODELS / "rf" / "model_card.json").read_text(encoding="utf-8"))
RF_PARAMS = dict(RF_CARD["params"])


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------- data


def load(path: Path | None = None) -> Dataset:
    return load_dataset(path, allow_test=True) if path else load_dataset(allow_test=True)


def eval_part(ds: Dataset, rng) -> tuple[pd.DataFrame, object, np.ndarray]:
    """(rows, EvalSet, 20 tie keys) of a commit range."""
    rows = ds.slice(rng)
    es = build_eval_set(rows, ds.positives_in(rng), ds.introduced_at)
    return rows, es, tie_keys(es, N_PERM)


def features_on(fit_rows: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    """Stage 6 rule (FitContext): drop columns constant on the fit part; order of `cols` is kept."""
    kept, dropped = drop_constant(fit_rows, cols)
    assert_non_negative(fit_rows, kept)
    return kept, dropped


# ---------------------------------------------------------------------- models


def fit_rf(fit_rows: pd.DataFrame, features: list[str], seed: int, params: dict | None = None, family: str = "rf"):
    params = RF_PARAMS if params is None else params
    return fit_pipeline(make_pipeline(family, params, features, seed), family, fit_rows, features)


def frozen_rf(seed: int):
    return joblib.load(MODELS / "rf" / RF_CARD["seed_files"][str(seed)])


def card_tau(seed: int, target: str = "0.90", card: dict = RF_CARD) -> float:
    return float(card["policy"]["by_recall_criterion"]["first"]["threshold_tau"][str(seed)][target])


# ---------------------------------------------------------------------- metrics (wrappers, no new definitions)


def prio(es, scores, keys) -> tuple[dict, pd.DataFrame]:
    per = evaluate(es, scores, n_perm=N_PERM, keys=keys)
    return summarize(per), per


def tau_first(es_val, scores_val, target: float = TARGET) -> float:
    return threshold_for_recall(es_val, scores_val, target, CRITERION)


def threshold_sel(es, scores, tau: float) -> tuple[dict, dict]:
    """(aggregate, per-commit columns) of the selection score >= tau (as policy.threshold_stats)."""
    cols = _mean_selection(es, [es.matrix(scores) >= tau])
    return _aggregate(es, cols), cols


def budget_sels(es, scores, keys, b: float) -> list[np.ndarray]:
    orders = _orders(es, scores, keys)
    pred_total = es.pred_dur.sum(1, keepdims=True)
    return [_cum_by_test(o, es.pred_dur) <= b * pred_total + 1e-9 for o in orders]


def budget_sel(es, scores, keys, b: float) -> tuple[dict, dict]:
    cols = _mean_selection(es, budget_sels(es, scores, keys, b))
    return _aggregate(es, cols), cols


def budget_first(es_val, scores_val, keys, target: float = TARGET) -> float | None:
    """B of the stage 6 rule: cheapest grid budget reaching the first-exposure target on this range."""
    best = cheapest_for_target(budget_curve(es_val, scores_val, keys), target, CRITERION)
    return None if best is None else float(best["setting"])


def per_defect_caught(es, sels: list[np.ndarray]) -> np.ndarray:
    """Share of selections (tie permutations) in which each (commit, defect) pair of `es` has a selected test
    failing because of it -- the per-pair term of metrics.selection_columns (sel_def_caught)."""
    acc = np.zeros(len(es.def_commit))
    for sel in sels:
        acc += (es.def_mask & sel[es.def_commit]).any(1)
    return acc / len(sels)


def diagnostics(y, p) -> dict:
    d = classification_diagnostics(y, p)
    return {"roc_auc": d["roc_auc"], "pr_auc": d["pr_auc"]}


def mean_std(v) -> tuple[float, float]:
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


# ---------------------------------------------------------------------- output


def out_dir(name: str) -> Path:
    if name == "main":
        raise SystemExit("ml/results/stage8/main/ belongs to 8B and is never written by 8C")
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, float):
        return float(f"{x:.10g}")  # stable text across runs; far below any reported precision
    return x


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.6f")


def write_run_info(out: Path, t0: float, started: str, inputs: list[Path], seeds, extra: dict | None = None) -> None:
    """run_info.json: seeds, sha256 of inputs, versions, timing, exact command. Only `started`, `finished`
    and `elapsed_seconds` change between identical reruns."""
    import scipy

    info = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "command": "ml/.venv/bin/python " + " ".join([str(Path(sys.argv[0]).as_posix())] + sys.argv[1:]),
        "seeds": list(seeds),
        "inputs_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        "versions": {**library_versions(), "scipy": scipy.__version__},
        "machine": {"processor": platform.processor(), "machine": platform.machine()},
        **(extra or {}),
    }
    write_json(out / "run_info.json", info)


def standard_inputs(ds: Dataset) -> list[Path]:
    files = ["dataset.csv", "meta.json", "splits.json", "positives.csv", "defects.csv"]
    return [ds.path / f for f in files] + [MODELS / "rf" / "model_card.json", Path(sys.argv[0]).resolve(),
                                            Path(__file__).resolve()]


def started_now() -> tuple[float, str]:
    return time.perf_counter(), datetime.now(timezone.utc).isoformat(timespec="seconds")


def main_policy_rows(method: str, seed: int, policy: str, criterion: str = "first", target: float = 0.9) -> pd.DataFrame:
    """Per-commit rows of one frozen policy of 8B (ml/results/stage8/main/per_commit_policies_test.csv)."""
    p = pd.read_csv(MAIN / "per_commit_policies_test.csv")
    m = (p["method"] == method) & (p["seed"] == seed) & (p["policy"] == policy) & (p["criterion"] == criterion) \
        & np.isclose(p["target"], target)
    return p[m].sort_values("commit_index").reset_index(drop=True)


def fmt6(a) -> np.ndarray:
    """Values as written with float_format %.6f (for comparisons with the stored 8B CSVs)."""
    return np.array([f"{float(x):.6f}" for x in np.asarray(a, dtype=float)])


__all__ = [n for n in dir() if not n.startswith("_") or n in ("_aggregate", "_mean_selection", "_orders")]

if __name__ == "__main__":
    h = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print("exp_common ok", h[:12], SEEDS[0], RF_PARAMS)


# ---------------------------------------------------------------------- per-defect view (experiments 7 and 8)


def first_exposure_rows(es, rows: pd.DataFrame, rf_scores: dict, rf_taus: dict, heuristics: dict, keys) -> pd.DataFrame:
    """One row per first-exposure pair (commit, defect) of `es`: share of RF seeds whose tau selection catches it,
    mean 1-based rank of its first failing test in the RF ranking, and the same for every heuristic with budget B
    (mean over the 20 tie permutations, as in 8B). `rf_scores` / `rf_taus`: seed -> scores / tau;
    `heuristics`: name -> (scores, B). Selection time shares are the per-commit sel_time of the same selections."""
    from metrics import positions, selection_columns

    k_first = np.flatnonzero(es.def_first)
    out = pd.DataFrame({"defect_id": es.def_ids[k_first], "commit_index": es.commits[es.def_commit[k_first]],
                        "n_failing_tests": es.def_mask[k_first].sum(1),
                        "failing_tests": [";".join(es.tests[m]) for m in es.def_mask[k_first]]})

    def block(prefix: str, sels_list: list[list[np.ndarray]], orders_list: list[list[np.ndarray]]) -> None:
        caught, rank, time_ = [], [], []
        for sels, orders in zip(sels_list, orders_list):
            caught.append(per_defect_caught(es, sels)[k_first])
            pos = np.mean([np.where(es.def_mask[k_first], positions(o)[es.def_commit[k_first]], es.n + 1).min(1) for o in orders], axis=0)
            rank.append(pos)
            t = np.mean([selection_columns(es, s, "x")["x_time"] for s in sels], axis=0)
            time_.append(t[es.def_commit[k_first]])
        out[f"{prefix}_caught"] = np.mean(caught, axis=0)
        out[f"{prefix}_rank"] = np.mean(rank, axis=0)
        out[f"{prefix}_time"] = np.mean(time_, axis=0)

    seeds = sorted(rf_scores)
    block("rf_tau", [[es.matrix(rf_scores[s]) >= rf_taus[s]] for s in seeds], [_orders(es, rf_scores[s], keys) for s in seeds])
    for name, (sc, b) in heuristics.items():
        block(f"{name}_B", [budget_sels(es, sc, keys, b)], [_orders(es, sc, keys)])
    return out


def range_first_exposures(ds, rng, label: str) -> tuple[pd.DataFrame, object]:
    """first_exposure_rows for the frozen RF (10 seeds, card tau) and combined / covers (B of the first-exposure
    criterion, target 0.90) on `rng` ('test' or 'val'). Scores are predictions of the frozen models (no fit, no
    tuning). Control (fatal): summed per commit, the caught shares equal first_caught of the frozen policies in
    8B's per_commit_policies_test.csv (test) or the 8A dry run's per_commit_policies_val.csv (val)."""
    from baselines import BASELINES, COMBINED_VARIANTS

    rows, es, keys = eval_part(ds, rng)
    rf_scores = {s: predict(frozen_rf(s), rows) for s in SEEDS}
    rf_taus = {s: card_tau(s) for s in SEEDS}
    heur = {"combined": (COMBINED_VARIANTS["comb_last_cov_mod_recent"](rows), COMBINED_B), "covers": (BASELINES["covers"](rows), COVERS_B)}
    df = first_exposure_rows(es, rows, rf_scores, rf_taus, heur, keys)
    ref_file = MAIN / "per_commit_policies_test.csv" if label == "test" else ROOT / "ml/results/stage8-dryrun/per_commit_policies_val.csv"
    ref = pd.read_csv(ref_file)
    sel = (ref["criterion"] == "first") & np.isclose(ref["target"], 0.9)
    for s in SEEDS:  # per seed: rebuild the per-pair catch for this seed only
        k_first = np.flatnonzero(es.def_first)
        caught = per_defect_caught(es, [es.matrix(rf_scores[s]) >= rf_taus[s]])[k_first]
        got = np.bincount(es.def_commit[k_first], weights=caught, minlength=len(es.commits))
        r = ref[sel & (ref["method"] == "rf") & (ref["seed"] == s) & (ref["policy"] == "threshold")].sort_values("commit_index")
        assert (fmt6(got) == fmt6(r["first_caught"])).all(), f"{label}: rf seed {s} per-defect catch != frozen policy"
    for name in heur:
        got = np.bincount(es.def_commit[es.def_first], weights=df[f"{name}_B_caught"].to_numpy(), minlength=len(es.commits))
        r = ref[sel & (ref["method"] == name) & (ref["policy"] == "budget")].sort_values("commit_index")
        assert (fmt6(got) == fmt6(r["first_caught"])).all(), f"{label}: {name} per-defect catch != frozen policy"
    return df, es
