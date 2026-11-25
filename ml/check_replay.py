"""Stage 8 (8A): self-checks of ml/replay.py, on val only (start = full runs 0-323, commits 324-431, history upto 432).

  1. lock          history beyond commit 432 is refused without allow_test_history; the dataset is loaded without the test part
  2. partial = full  update_partial with every test executed == HistoryState.update (features of every val commit, final state)
  3. hand example  an unexecuted failing test records 0 in the failure vector, its duration counters are unchanged;
                   an executed failing test is recorded
  4. N = 1         for every RF seed (tau, first exposure, 0.90) and for combined + B: the features of every commit
                   (computed before the run) == dataset.csv rows, and the selections == stage 6 (tau on
                   val_predictions.csv; budget mask of permutation 0)
  5. cold start    an empty list of cold tests gives the base features (variants a and b); a reset test looks like
                   a test without runs; own_only leaves the warm tests unchanged
  6. smoke         N = 5 and N = inf (RF seed 20261117, combined + B), cold start (a) and (b): the code runs and the
                   numbers are in range. They are printed, NOT recorded as results (8C runs the experiments on test).

Usage:  ml/.venv/bin/python ml/check_replay.py   (~1 min)
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import pandas as pd

from build_dataset import HistoryError, SINCE_FAIL_CAP
from dataset_io import ROOT, load_dataset
from features_online import HistoryState
from metrics import N_PERM, _cum_by_test, build_eval_set, order_from, tie_keys
from prioritize import load_method, resolve_policy
from replay import COLD_TESTS, History, ObservedHistoryState, loop_metrics, open_loop, run_loop

FAILS: list[str] = []
FEATURE_PREFIXES = ("chg_", "hist_", "meta_", "ci_")


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def same_features(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> list[str]:
    bad = []
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        eq = (x == y).all() if c == "meta_module" else np.array_equal(x.astype(float), y.astype(float), equal_nan=True)
        if not eq:
            bad.append(c)
    return bad


def main() -> int:
    t0 = time.time()
    ds = load_dataset()  # allow_test=False
    a, b = ds.val_range
    print("1. lock")
    check("dataset loaded without the test part", int(ds.rows["commit_index"].max()) < ds.test_range[0])
    try:
        History.load(433)
        check("history upto 433 refused without allow_test_history", False)
    except HistoryError:
        check("history upto 433 refused without allow_test_history", True)
    hist = History.load(b)  # 432 = end of val
    check("history loaded up to the end of val only", len(hist.runs) == b == ds.test_range[0], f"{len(hist.runs)} runs")

    ref = ds.slice((a, b))
    feats = [c for c in ref.columns if c.startswith(FEATURE_PREFIXES)]
    es = build_eval_set(ref, ds.positives_in((a, b)), ds.introduced_at)
    keys = tie_keys(es, N_PERM)

    print("2. update_partial(all executed) == HistoryState.update")
    base = HistoryState()
    obs = ObservedHistoryState()
    for run, diff in zip(hist.runs[:a], hist.diffs[:a]):
        base.update(diff, run)
        obs.update_partial(diff, run, np.ones(36, bool))
    bad_c = 0
    for c in range(a, b):
        run, diff = hist.runs[c], hist.diffs[c]
        fb = base.features(diff, c, run["sha"], hist.ci(c))
        fo = obs.features(diff, c, run["sha"], hist.ci(c))
        bad_c += bool(same_features(fb, fo, feats))
        base.update(diff, run)
        obs.update_partial(diff, run, np.ones(36, bool))
    same_state = (np.array_equal(np.array(base.fail_rows), np.array(obs.fail_rows)) and np.array_equal(base.last_fail, obs.last_fail)
                  and np.array_equal(base.total_fail, obs.total_fail) and np.array_equal(base.dur_sum, obs.dur_sum)
                  and np.array_equal(base.dur_cnt, obs.dur_cnt) and base.n_past == obs.n_past and dict(base.file_changes) == dict(obs.file_changes)
                  and all(np.array_equal(base.file_cofail[p], obs.file_cofail[p]) for p in base.file_cofail))
    check("features of 108 val commits identical", bad_c == 0, f"{bad_c} commits differ")
    check("final state identical (0..431)", same_state)

    print("3. hand example of a partial update")
    st = hist.state_before(a)
    y = es.y
    c = int(es.commits[np.where(y.sum(1) >= 2)[0][0]])  # first val commit with >= 2 failed tests
    fail_idx = np.where(y[c - a])[0]
    k_hidden, k_seen = int(fail_idx[0]), int(fail_idx[1])
    for cc in range(a, c):
        st.update(hist.diffs[cc], hist.runs[cc])
    before = (st.total_fail.copy(), st.dur_sum.copy(), st.dur_cnt.copy(), st.last_fail.copy())
    executed = np.ones(36, bool)
    executed[k_hidden] = False
    st.update_partial(hist.diffs[c], hist.runs[c], executed)
    check(f"commit {c}: unexecuted failing test {st.test_ids[k_hidden]} records 0",
          st.fail_rows[-1][k_hidden] == 0 and st.total_fail[k_hidden] == before[0][k_hidden] and st.last_fail[k_hidden] == before[3][k_hidden])
    check("   its duration counters are unchanged", st.dur_sum[k_hidden] == before[1][k_hidden] and st.dur_cnt[k_hidden] == before[2][k_hidden])
    check(f"   executed failing test {st.test_ids[k_seen]} is recorded", st.fail_rows[-1][k_seen] == 1 and st.total_fail[k_seen] == before[0][k_seen] + 1)
    nxt = st.features(hist.diffs[c + 1], c + 1, hist.runs[c + 1]["sha"], hist.ci(c + 1))
    check("   next commit: hist_fail_last_1 = 0 for the hidden failure, 1 for the observed one",
          nxt.loc[k_hidden, "hist_fail_last_1"] == 0 and nxt.loc[k_seen, "hist_fail_last_1"] == 1)

    print("4. N = 1: features and selections == open loop of stage 6")
    start_state = hist.state_before(a)
    preds = pd.read_csv(ROOT / "ml/results/stage6/val_predictions.csv", float_precision="round_trip")
    rf0 = load_method("rf")
    for s in rf0.card["seeds"]:
        m = load_method("rf", seed=int(s))
        pol = resolve_policy(m, "threshold", "first", "0.90")
        res = run_loop(hist, a, b, m, pol, 1, keep_features=True, state=start_state)
        got = pd.concat(res.features, ignore_index=True)
        bad = same_features(ref, got, feats)
        sc = preds[(preds["model"] == "rf") & (preds["seed"] == int(s))].sort_values(["commit_index", "test_id"], kind="mergesort")["score"].to_numpy()
        sel_ref = es.matrix(sc) >= pol.setting
        check(f"rf seed {s}: features == dataset.csv, selection == tau on val_predictions",
              not bad and np.array_equal(res.selected, sel_ref) and res.executed.all() and list(got["test_id"][:36]) == res.test_ids,
              f"{int(res.selected.sum())} selected; bad columns {bad[:3]}")
    comb = load_method("combined")
    pol_c = resolve_policy(comb, "budget", "first", "0.90")
    res = run_loop(hist, a, b, comb, pol_c, 1, keep_features=True, state=start_state)
    from baselines import COMBINED_VARIANTS
    sc = COMBINED_VARIANTS["comb_last_cov_mod_recent"](ref)
    o0 = order_from(es.matrix(sc), keys[0])
    sel_ref = _cum_by_test(o0, es.pred_dur) <= pol_c.setting * es.pred_dur.sum(1, keepdims=True) + 1e-9
    check(f"combined + B = {pol_c.setting:.2f}: features == dataset.csv, selection == budget mask of permutation 0",
          not same_features(ref, pd.concat(res.features, ignore_index=True), feats) and np.array_equal(res.selected, sel_ref))
    ol = open_loop(ref, rf0, resolve_policy(rf0, "threshold", "first", "0.90"))
    check("open_loop(rf seed 0) == closed loop with N = 1", np.array_equal(ol.selected, es.matrix(preds[(preds["model"] == "rf") & (preds["seed"] == rf0.seed)]
          .sort_values(["commit_index", "test_id"], kind="mergesort")["score"].to_numpy()) >= resolve_policy(rf0, "threshold", "first", "0.90").setting))

    print("5. cold start")
    pol0 = resolve_policy(rf0, "threshold", "first", "0.90")
    for variant in ("a", "b"):
        res = run_loop(hist, a, b, rf0, pol0, 1, cold=(), cold_variant=variant, keep_features=True, state=start_state)
        check(f"variant ({variant}) with no cold tests: features == base", not same_features(ref, pd.concat(res.features, ignore_index=True), feats))
    manifest = {t["id"]: float(t["avgDurationMs"]) for t in start_state.tests}
    for scope in ("own_only", "remove"):
        st = hist.state_before(a, peer_scope=scope)
        warm_before = st.features(hist.diffs[a], a, hist.runs[a]["sha"], hist.ci(a))
        st.reset_tests(COLD_TESTS)
        f = st.features(hist.diffs[a], a, hist.runs[a]["sha"], hist.ci(a))
        cm = f["test_id"].isin(COLD_TESTS).to_numpy()
        fresh = ((f.loc[cm, [f"hist_fail_last_{w}" for w in (1, 3, 5, 10, 20)]] == 0).all().all()
                 and (f.loc[cm, "hist_since_fail"] == SINCE_FAIL_CAP).all() and (f.loc[cm, "hist_never_failed"] == 1).all()
                 and (f.loc[cm, "hist_fail_rate"] == 0).all() and (f.loc[cm, "hist_file_fail_count"] == 0).all()
                 and (f.loc[cm, "hist_file_fail_rate"] == 0).all()
                 and np.allclose(f.loc[cm, "meta_duration_mean"].to_numpy(), [manifest[t] for t in f.loc[cm, "test_id"]]))
        check(f"{scope}: 9 reset tests look like tests without runs", bool(fresh))
        warm_same = not same_features(warm_before[~cm], f[~cm], feats)
        if scope == "own_only":
            check("own_only: warm tests' features unchanged", warm_same)
            check("own_only: chg_ features of cold tests unchanged",
                  not same_features(warm_before[cm], f[cm], [c for c in feats if c.startswith("chg_")]))
        else:
            check("remove: some warm peer features change (the cold tests' failures are removed)", not warm_same)

    print("6. smoke runs (printed, not recorded)")
    defects = ds.defects
    cols = ["method", "seed", "n_full", "full_run_commits", "recall_first_all", "time_all", "recall_first_selected", "time_selected",
            "defects", "detected", "delay_mean", "delay_max", "undetected_before_fix", "undetected_alive_at_end"]
    rows = []
    for n_full in (1, 5, 10, math.inf):
        for m, pol in ((rf0, pol0), (comb, pol_c)):
            res = run_loop(hist, a, b, m, pol, n_full, state=start_state)
            rows.append(loop_metrics(es, res, defects))
    rows.append({**loop_metrics(es, open_loop(ref, rf0, pol0), defects), "method": "rf (open loop)"})
    for variant in ("a", "b"):
        res = run_loop(hist, a, b, rf0, pol0, math.inf, cold=COLD_TESTS, cold_variant=variant, state=start_state)
        rows.append({**loop_metrics(es, res, defects, cold=COLD_TESTS), "method": f"rf cold ({variant})"})
    res = run_loop(hist, a, b, rf0, pol0, 1, cold=COLD_TESTS, cold_variant="a", state=start_state)
    rows.append({**loop_metrics(es, res, defects, cold=COLD_TESTS), "method": "rf cold (a), N=1"})
    tab = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(tab[cols + ["cold_test_recall"]].round(3).to_string(index=False))
    n1 = tab[(tab["n_full"] == 1) & (tab["method"].isin(["rf", "combined"]))]
    check("N = 1: recall_first_all = 1 and time_all = 1 (trivial)", bool(((n1["recall_first_all"] == 1) & (np.isclose(n1["time_all"], 1))).all()))
    n5 = tab[tab["n_full"] == 5]
    check("N = 5: 21 full-run commits, time_all between time_selected and 1",
          bool(((n5["full_run_commits"] == 21) & (n5["time_all"] >= n5["time_selected"]) & (n5["time_all"] <= 1)).all()))
    ok_range = tab[["recall_first_all", "recall_first_selected", "time_all", "time_selected"]].apply(lambda s: s.dropna().between(0, 1).all()).all()
    check("all recalls and time shares in [0, 1]; delays >= 0", bool(ok_range and (tab["delay_mean"].dropna() >= 0).all()))
    check("detected + undetected = defects", bool((tab["detected"] + tab["undetected_before_fix"] + tab["undetected_alive_at_end"] == tab["defects"]).all()))
    print(f"done in {time.time() - t0:.0f} s: {'all checks passed' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
