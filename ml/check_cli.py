"""Stage 7: self-checks of the CLI pipeline (diff -> features -> scores -> policy), all on commits 0-431.

  1. feature parity   online features (HistoryState replay of data/history-600) == dataset.csv rows, 0-431
  2. score parity     scores of the online rows == val_predictions.csv (rf, seed 20261117), 324-431
  3. policy parity    tau / B / top-k settings from the card reproduce policies.csv (rf, seed 20261117) on val,
                      and the one-commit `select` equals the policy.py mask of permutation 0
  4. determinism      ranking a commit twice gives the same order; ties are never broken by test_id
  5. calibration      calibrated p in [0, 1] and orders the tests like the raw score
  6. test lock        nothing here reads commits >= 432 (replay stops at 432, load_dataset without allow_test)
  7. CLI end to end   `ml/cli.py rank --json` on history commits equals the library result

Usage:  ml/.venv/bin/python ml/check_cli.py   (~20 s; the first run also caches the history diffs)
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from dataset_io import load_dataset
from features_online import ROOT, replay
from metrics import N_PERM, _cum_by_test, build_eval_set, order_from, tie_keys
from policy import budget_curve, threshold_stats, topk_curve
from prioritize import load_method, prioritize, rank_order, resolve_policy, scores, select

HISTORY = ROOT / "data" / "history-600"
VAL_END = 432
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def main() -> int:
    t0 = time.time()
    ds = load_dataset()  # allow_test=False: the test part is dropped right after reading
    check("6. test lock: dataset loaded without the test part", int(ds.rows["commit_index"].max()) < ds.test_range[0],
          f"max commit in memory {int(ds.rows['commit_index'].max())}, test part {list(ds.test_range)}")
    upto = ds.test_range[0]
    check("6. test lock: replay stops before the test part", upto == VAL_END, f"upto={upto}")

    print("1. feature parity")
    state, online = replay(HISTORY, upto, emit=True)
    check("   replayed commits", state.commits_seen == upto, f"{state.commits_seen}")
    ref = ds.slice((0, upto))
    online = online.sort_values(["commit_index", "test_id"], kind="mergesort").reset_index(drop=True)
    feats = [c for c in ref.columns if c.startswith(("chg_", "hist_", "meta_", "ci_"))]
    same_ids = (ref[["commit_index", "sha", "test_id"]].to_numpy() == online[["commit_index", "sha", "test_id"]].to_numpy()).all()
    check("   ids (commit, sha, test)", bool(same_ids) and len(ref) == len(online), f"{len(online)} rows")
    bad = []
    for c in feats:
        a, b = ref[c].to_numpy(), online[c].to_numpy()
        equal = (a == b).all() if c == "meta_module" else np.array_equal(a.astype(float), b.astype(float))
        if not equal:
            bad.append(c)
    check(f"   {len(feats)} feature columns exactly equal", not bad, ", ".join(bad))

    print("2. score parity (rf, seed 20261117, val)")
    method = load_method("rf")
    a, b = ds.val_range
    val_online = online[(online["commit_index"] >= a) & (online["commit_index"] < b)].reset_index(drop=True)
    s = scores(method, val_online)
    preds = pd.read_csv(ROOT / "ml/results/stage6/val_predictions.csv", float_precision="round_trip")
    ref_s = preds[(preds["model"] == "rf") & (preds["seed"] == method.seed)].sort_values(
        ["commit_index", "test_id"], kind="mergesort")["score"].to_numpy()
    check("   scores equal val_predictions.csv", len(ref_s) == len(s) and np.array_equal(s, ref_s),
          f"max |Δ| = {np.abs(s - ref_s).max() if len(ref_s) == len(s) else 'n/a'}")
    features_card = set(method.card["features"])
    check("   model features = card features (no ci_, constant columns dropped)",
          not any(f.startswith("ci_") for f in features_card) and not (set(method.card["dropped_constant_features"]) & features_card))

    print("3. policy parity (val, in-sample settings of stage 6)")
    val_rows = ds.slice(ds.val_range)
    es = build_eval_set(val_rows, ds.positives_in(ds.val_range), ds.introduced_at)
    keys = tie_keys(es, N_PERM)
    pol_csv = pd.read_csv(ROOT / "ml/results/stage6/policies.csv")
    pol_csv = pol_csv[(pol_csv["method"] == "rf") & (pol_csv["seed"] == method.seed)]
    for criterion in ("first", "commit"):
        for name in ("threshold", "budget", "topk"):
            pol = resolve_policy(method, name, criterion, "0.90")
            ref_row = pol_csv[(pol_csv["criterion"] == criterion) & (pol_csv["policy"] == name)
                              & np.isclose(pol_csv["target"], 0.90)].iloc[0]
            if name == "threshold":
                got = threshold_stats(es, s, pol.setting)
            elif name == "budget":
                got = budget_curve(es, s, keys, grid=(pol.setting,)).iloc[0].to_dict()
            else:
                curve = topk_curve(es, s, keys)
                got = curve[curve["n_tests"] == int(round(pol.setting * es.n))].iloc[0].to_dict()
            ok = all(np.isclose(got[k], ref_row[k], atol=1e-6) for k in ("recall_first", "recall_commit", "time", "tests"))
            check(f"   {criterion:6s} {name:9s} {pol.setting:.4f}", ok,
                  f"recall_first {got['recall_first']:.3f}, recall_commit {got['recall_commit']:.3f}, "
                  f"time {100 * got['time']:.1f} %")
    # one-commit select == policy.py mask of permutation 0
    S = es.matrix(s)
    order0 = order_from(S, keys[0])
    cum0 = _cum_by_test(order0, es.pred_dur)
    mism = 0
    for i, c in enumerate(es.commits):
        order = rank_order(S[i], int(c))
        mism += not np.array_equal(order, order0[i])
        for name in ("budget", "threshold", "topk"):
            pol = resolve_policy(method, name, "first", "0.90")
            mine = select(pol, S[i], order, es.pred_dur[i])
            if name == "budget":
                ref_mask = cum0[i] <= pol.setting * es.pred_dur[i].sum() + 1e-9
            elif name == "threshold":
                ref_mask = S[i] >= pol.setting
            else:
                pos = np.empty(es.n, dtype=int)
                pos[order0[i]] = np.arange(1, es.n + 1)
                ref_mask = pos <= int(round(pol.setting * es.n))
            mism += not np.array_equal(mine, ref_mask)
    check("   one-commit ranking and selection == metrics/policy (perm 0), 108 commits", mism == 0, f"{mism} mismatches")

    print("4. determinism and ties")
    one = online[online["commit_index"] == a].reset_index(drop=True)
    pol = resolve_policy(method, "threshold")
    t1, t2 = prioritize(method, one, pol), prioritize(method, one, pol)
    check("   same table twice", t1.equals(t2))
    flat = np.zeros(36)
    o1, o2 = rank_order(flat, 1000), rank_order(flat, 1001)
    check("   all-tie order is random (not by test_id) and depends on the commit",
          not np.array_equal(o1, np.arange(36)) and not np.array_equal(o1, o2))

    print("5. calibration")
    p = method.calibrated.predict_proba(val_online)[:, 1]
    check("   calibrated p in [0, 1]", bool(((p >= 0) & (p <= 1)).all()))
    check("   calibrated p is monotone in the raw score", bool((np.diff(p[np.argsort(s, kind="mergesort")]) >= -1e-12).all()))

    print("7. CLI end to end (rank on history commits 325 and 400)")
    for c in (325, 400):
        sha, parent = ds.rows.loc[ds.rows["commit_index"] == c, "sha"].iloc[0], ds.rows.loc[ds.rows["commit_index"] == c - 1, "sha"].iloc[0]
        cmd = [sys.executable, str(ROOT / "ml/cli.py"), "rank", "--repo", str(HISTORY / "repo"), "--base", parent,
               "--head", sha, "--history", str(HISTORY), "--upto", str(c), "--json"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            check(f"   cli rank {c}", False, res.stderr.strip()[-300:])
            continue
        out = json.loads(res.stdout)
        lib = prioritize(method, online[online["commit_index"] == c].reset_index(drop=True), pol)
        got = [(t["test_id"], t["selected"]) for t in out["tests"]]
        exp = list(zip(lib["test_id"], lib["selected"].astype(bool)))
        check(f"   cli rank {c}: order and selection equal the library", got == exp and out["commit_index"] == c,
              f"selected {sum(x[1] for x in got)} of 36")
    print(f"done in {time.time() - t0:.1f} s: {'all checks passed' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
