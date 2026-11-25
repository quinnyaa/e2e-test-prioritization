"""Stage 6 self-checks of ml/metrics.py: hand examples with known answers, brute force over all
permutations of small commits, analytic random expectations on the val part of the real dataset.

Run: ml/.venv/bin/python ml/check_metrics.py   (exit code 1 on any failure)
"""

from __future__ import annotations

import itertools
import math
import sys

import numpy as np
import pandas as pd

from baselines import BASELINES, INVERSE_ORACLE, ORACLE
from dataset_io import TestLockError, load_dataset
from metrics import (
    N_PERM_RANDOM,
    EvalSet,
    _apfd,
    _apfdc,
    build_eval_set,
    evaluate,
    perm_metrics,
    positions,
    random_commit_hit_expectation,
    random_defect_apfd_expectation,
    selection_columns,
    summarize,
    summarize_selection,
    tie_keys,
)
from policy import threshold_for_recall, threshold_stats

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def toy_set(y: list[int], dur: list[float], new: list[int] | None = None, defects: list[list[int]] | None = None,
            first: list[bool] | None = None) -> EvalSet:
    n = len(y)
    y_arr = np.array([y], dtype=bool)
    new_arr = np.array([new if new is not None else y], dtype=bool) & y_arr
    d = np.array([dur], dtype=float)
    defects = defects or []
    return EvalSet(
        commits=np.array([0]), tests=np.array([f"T{i}" for i in range(n)]), y=y_arr, new=new_arr, dur=d, pred_dur=d.copy(),
        def_commit=np.zeros(len(defects), dtype=int),
        def_mask=np.array([[i in dset for i in range(n)] for dset in defects], dtype=bool).reshape(len(defects), n),
        def_ids=np.array([f"D{i}" for i in range(len(defects))], dtype=object),
        def_first=np.array(first if first is not None else [True] * len(defects), dtype=bool),
    )


def hand_examples() -> None:
    # Rothermel/Elbaum style: n = 5, failures at positions 1 and 3 -> APFD = 1 - 4/10 + 1/10 = 0.7.
    es = toy_set([1, 0, 1, 0, 0], [1, 1, 1, 1, 1])
    order = np.array([[0, 1, 2, 3, 4]])
    m = perm_metrics(es, order)
    check("APFD hand example", math.isclose(m["apfd"][0], 0.7), f"{m['apfd'][0]:.4f}")
    # Equal durations: APFDc equals APFD.
    check("APFDc == APFD for equal durations", math.isclose(m["apfdc"][0], 0.7), f"{m['apfdc'][0]:.4f}")
    # Durations 4,1,1,1,1 (T=8), faults at positions 1 and 3:
    # fault 1: (8 - 2) = 6; fault 2: (sum_{j>=3} = 3) - 0.5 = 2.5 -> (6 + 2.5) / (2 * 8) = 0.53125.
    es2 = toy_set([1, 0, 1, 0, 0], [4, 1, 1, 1, 1])
    m2 = perm_metrics(es2, order)
    check("APFDc hand example", math.isclose(m2["apfdc"][0], 0.53125), f"{m2['apfdc'][0]:.5f}")
    # Supervisor's control case (review 2026-11-25): two tests of equal duration, one fault.
    # APFDc = 1 - (C - t/2) / (m T): found by the first test -> 0.75, by the second -> 0.25.
    es3 = toy_set([1, 0], [1, 1])
    a = perm_metrics(es3, np.array([[0, 1]]))["apfdc"][0]
    b = perm_metrics(es3, np.array([[1, 0]]))["apfdc"][0]
    check("APFDc control case 0.75 / 0.25", math.isclose(a, 0.75) and math.isclose(b, 0.25), f"{a:.4f} / {b:.4f}")
    # Time to first failure: 4 / 8 = 0.5; recall@50% time = 1, @30% = 0.
    check("time to first failure", math.isclose(m2["ttd_share"][0], 0.5) and m2["rt50_hit"][0] == 1 and m2["rt30_hit"][0] == 0)
    # Top-10% of 5 tests = ceil(0.5) = 1 test -> hit; top-20% = 1 test.
    check("top-k hit", m2["top10_hit"][0] == 1 and m2["top10_caught"][0] == 1 and m2["top50_caught"][0] == 2)
    # Budget 50% of predicted 8 ms = 4 ms: only the first test (4 ms) fits.
    check("budget prefix", m2["bud50_tests"][0] == 0.2 and m2["bud50_hit"][0] == 1 and m2["bud10_tests"][0] == 0.0)
    # New vs repeated: failure at position 3 is new, at position 1 repeated.
    es3 = toy_set([1, 0, 1, 0, 0], [1] * 5, new=[0, 0, 1, 0, 0])
    m3 = perm_metrics(es3, order)
    check("APFD_new counts only new failures", math.isclose(m3["apfd_new"][0], 1 - 3 / 5 + 0.1)
          and math.isclose(m3["apfd_repeat"][0], 1 - 1 / 5 + 0.1))
    # Defect level: D0 detected by tests {0, 2} -> TF = 1; D1 by {2} -> TF = 3 -> 1 - 4/10 + 0.1.
    es4 = toy_set([1, 0, 1, 0, 0], [1] * 5, defects=[[0, 2], [2]])
    m4 = perm_metrics(es4, order)
    check("defect-level APFD", math.isclose(m4["apfd_defect"][0], 0.7), f"{m4['apfd_defect'][0]:.4f}")
    first_exposure_example()


def first_exposure_example() -> None:
    # Two commits, n = 4. Commit 0: defect A fails {0} (first exposure), B fails {2, 3} (repeat).
    # Commit 1: defect C fails {1} (first exposure). Select test 0 in both commits.
    y = np.array([[1, 0, 1, 1], [0, 1, 0, 0]], dtype=bool)
    d = np.ones((2, 4))
    new = y & ~np.array([[0, 0, 1, 1], [0, 0, 0, 0]], dtype=bool)  # B's failures are repeats
    es = EvalSet(commits=np.array([10, 11]), tests=np.array(["T0", "T1", "T2", "T3"]), y=y, new=new, dur=d, pred_dur=d.copy(),
                 def_commit=np.array([0, 0, 1]), def_mask=np.array([[1, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]], dtype=bool),
                 def_ids=np.array(["A", "B", "C"], dtype=object), def_first=np.array([True, False, True]))
    sel = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=bool)
    per = pd.DataFrame(selection_columns(es, sel, "x"))
    per["n_fail"], per["n_new"], per["n_repeat"] = y.sum(1), new.sum(1), (y & ~new).sum(1)
    per["n_defects"], per["n_first"] = es.def_per_commit, es.first_per_commit
    s = summarize_selection(per, "x")
    check("defect / first-exposure recall hand example",
          math.isclose(s["x_recall_commit"], 0.5) and math.isclose(s["x_recall_defect"], 1 / 3) and math.isclose(s["x_recall_first"], 0.5),
          f"commit {s['x_recall_commit']:.3f}, defect {s['x_recall_defect']:.3f}, first {s['x_recall_first']:.3f}")
    # tau for first exposure: A max score 0.9, C max score 0.3 -> target 1.0 -> 0.3, target 0.5 -> 0.9.
    sc = np.array([0.9, 0.1, 0.5, 0.4, 0.2, 0.3, 0.8, 0.1])
    t1, t05 = threshold_for_recall(es, sc, 1.0, "first"), threshold_for_recall(es, sc, 0.5, "first")
    tc = threshold_for_recall(es, sc, 1.0, "commit")
    check("tau by first-exposure criterion", math.isclose(t1, 0.3) and math.isclose(t05, 0.9) and math.isclose(tc, 0.3)
          and threshold_stats(es, sc, t1)["recall_first"] == 1.0 and threshold_stats(es, sc, t05)["recall_first"] == 0.5,
          f"tau(1.0) {t1}, tau(0.5) {t05}")


def brute_force() -> None:
    rng = np.random.default_rng(1)
    for trial in range(6):
        n = 6
        y = np.zeros(n, dtype=int)
        y[rng.choice(n, size=rng.integers(1, 4), replace=False)] = 1
        dur = rng.integers(1, 20, size=n).astype(float)
        es = toy_set(list(y), list(dur))
        apfd, apfdc = [], []
        for perm in itertools.permutations(range(n)):
            order = np.array([perm])
            pos = positions(order)
            apfd.append(_apfd(pos, es.y, n)[0])
            apfdc.append(_apfdc(order, es.dur, es.y)[0])
        apfd, apfdc = np.array(apfd), np.array(apfdc)
        rows = pd.DataFrame({"label": y, "eval_duration_ms": dur})
        best = evaluate(es, ORACLE(rows), n_perm=5)
        worst = evaluate(es, INVERSE_ORACLE(rows), n_perm=5)
        ok = (math.isclose(best["apfd"][0], apfd.max()) and math.isclose(worst["apfd"][0], apfd.min())
              and math.isclose(best["apfdc"][0], apfdc.max()) and math.isclose(worst["apfdc"][0], apfdc.min())
              and math.isclose(apfd.mean(), 0.5) and math.isclose(apfdc.mean(), 0.5))
        m = int(y.sum())
        check(f"brute force n=6, m={m} (trial {trial})", ok,
              f"oracle APFD {best['apfd'][0]:.4f} = max {apfd.max():.4f} = 1 - m/2n {1 - m / (2 * n):.4f}; "
              f"oracle APFDc {best['apfdc'][0]:.4f} = max {apfdc.max():.4f}; mean over n! perms APFD {apfd.mean():.4f}, APFDc {apfdc.mean():.4f}")


def real_data() -> None:
    ds = load_dataset()
    try:
        ds.slice((430, 440))
        check("test lock refuses [430, 440)", False)
    except TestLockError:
        check("test lock refuses [430, 440)", True)
    check("test rows are not in memory", int(ds.rows["commit_index"].max()) < ds.test_range[0])

    rows = ds.slice(ds.val_range)
    es = build_eval_set(rows, ds.positives_in(ds.val_range), ds.introduced_at)
    red = es.y.any(1)
    check("val: 43 red commits, 194 positives, 114 new", int(red.sum()) == 43 and int(es.y.sum()) == 194 and int(es.new.sum()) == 114,
          f"{int(red.sum())}, {int(es.y.sum())}, {int(es.new.sum())}")
    check("val: 61 (commit, defect) pairs, 35 first exposures", len(es.def_commit) == 61 and int(es.def_first.sum()) == 35,
          f"{len(es.def_commit)}, {int(es.def_first.sum())}")
    # No masking: every defect that ever fails (train + val) fails already at its introduction commit,
    # so 'first exposure' = 'introduction commit' (the test part is not in memory).
    pos = ds.positives.assign(d=ds.positives["causes"].astype(str).str.split(";")).explode("d")
    first_seen = pos.groupby("d")["commit_index"].min()
    intro = ds.introduced_at
    check("first failure of every detected defect = its introduction commit",
          all(int(c) == intro[d] for d, c in first_seen.items()), f"{len(first_seen)} defects")

    keys = tie_keys(es, N_PERM_RANDOM)
    per = evaluate(es, BASELINES["random"](rows), n_perm=N_PERM_RANDOM, keys=keys)
    s = summarize(per)
    check("random APFD ≈ 0.5", abs(s["apfd"] - 0.5) < 0.01, f"{s['apfd']:.4f} (per-commit exact 0.5; MC)")
    check("random APFDc ≈ 0.5", abs(s["apfdc"] - 0.5) < 0.015, f"{s['apfdc']:.4f}")
    # Commit-level recall at top-k: analytic 1 - C(n-m, s)/C(n, s) averaged over red commits.
    n = es.n
    for k, sk in ((0.1, 4), (0.3, 11)):
        exp = np.mean([random_commit_hit_expectation(n, int(m), sk) for m in es.y.sum(1)[red]])
        got = s[f"top{int(k * 100)}_recall_commit"]
        check(f"random commit recall@top{int(k * 100)}% vs analytic", abs(got - exp) < 0.02, f"{got:.4f} vs {exp:.4f}")
        exp_t = sk / n
        got_t = s[f"top{int(k * 100)}_recall_test"]
        check(f"random test recall@top{int(k * 100)}% vs s/n", abs(got_t - exp_t) < 0.02, f"{got_t:.4f} vs {exp_t:.4f}")
    # Defect level: E[TF] = (n+1)/(k+1) per defect.
    exp_def = []
    for c in np.where(red)[0]:
        ks = es.def_mask[es.def_commit == c].sum(1)
        exp_def.append(random_defect_apfd_expectation(n, list(ks)))
    got_def = s["apfd_defect"]
    check("random defect-level APFD vs analytic", abs(got_def - np.mean(exp_def)) < 0.015, f"{got_def:.4f} vs {np.mean(exp_def):.4f}")
    # First-exposure recall of random top-s: mean over first pairs of 1 - C(n-k, s)/C(n, s), k = tests failing because of the defect.
    ks = es.def_mask[es.def_first].sum(1)
    for k_, sk in ((0.1, 4), (0.3, 11)):
        exp = np.mean([random_commit_hit_expectation(n, int(k), sk) for k in ks])
        got = s[f"top{int(k_ * 100)}_recall_first"]
        check(f"random first-exposure recall@top{int(k_ * 100)}% vs analytic", abs(got - exp) < 0.03, f"{got:.4f} vs {exp:.4f}")

    # Oracle: APFD = 1 - m/(2n) for each red commit; inverse oracle is the minimum.
    o = evaluate(es, ORACLE(rows))
    exp_o = (1 - es.y.sum(1)[red] / (2 * n)).mean()
    check("oracle APFD = mean(1 - m/2n)", math.isclose(summarize(o)["apfd"], exp_o), f"{summarize(o)['apfd']:.4f}")
    inv = evaluate(es, INVERSE_ORACLE(rows))
    check("inverse oracle is the minimum", (inv["apfd"][red] <= per["apfd"][red] + 1e-12).all() and (o["apfd"][red] >= per["apfd"][red] - 1e-12).all())
    check("oracle commit recall@top10% = 1", summarize(o)["top10_recall_commit"] == 1.0)

    # Common random numbers: the first 20 keys equal the prefix of the 200 keys.
    check("tie keys are prefix-stable", np.array_equal(tie_keys(es, 20), keys[:20]))
    # Ties are never broken by test_id: a constant score gives different orders across perms.
    check("random baseline used all permutations", int(per["n_perm"].iloc[0]) == N_PERM_RANDOM)


def main() -> None:
    hand_examples()
    brute_force()
    real_data()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print("all metric checks passed")


if __name__ == "__main__":
    main()
