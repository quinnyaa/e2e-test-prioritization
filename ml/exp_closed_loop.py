"""Stage 8C, experiment 2: closed-loop replay with selection feedback (notes/stage8-protocol.md, p. 5.6).

    ml/.venv/bin/python ml/exp_closed_loop.py   # -> ml/results/stage8/closed_loop/, ≈ 3 min

Start: observed history = full runs of commits 0-431; commits 432-539 (History.load(540, allow_test_history=True)).
For every commit the features come from the observed history, the frozen method selects tests, and only the
executed tests are recorded (ml/replay.py). Full run on commits with (c - 432 + 1) mod N = 0, N in {5, 10, inf}.
Methods: frozen RF + tau (first exposure, 0.90), 10 seeds, one loop per seed; combined + B = 0.34 (ties: key of
permutation 0, protocol section 9). Open loop (selection on dataset rows, i.e. history from full runs) of the same
methods for comparison. Metrics: replay.loop_metrics only.

Controls (fatal): N = 1 for RF seed 20261117 and combined -- features of every test commit equal dataset.csv and
the selection equals tau on the frozen predictions / the permutation-0 budget mask (the 8A check, on test);
open loop of RF reproduces 8B's per-seed recall_first and time (h1.json).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

import exp_common as ec
from metrics import order_from
from prioritize import load_method, resolve_policy
from replay import History, loop_metrics, open_loop, run_loop
from train import SEEDS, predict

N_FULL = (5, 10, math.inf)


def same_features(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> list[str]:
    bad = []
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        eq = (x == y).all() if c == "meta_module" else np.array_equal(x.astype(float), y.astype(float), equal_nan=True)
        if not eq:
            bad.append(c)
    return bad


def main() -> None:
    t0, started = ec.started_now()
    out = ec.out_dir("closed_loop")
    ds = ec.load()
    a, b = ds.test_range
    rows, es, keys = ec.eval_part(ds, (a, b))
    defects = ds.defects
    hist = History.load(b, allow_test_history=True)
    start = hist.state_before(a)
    feats = ds.feature_set("main+ci")
    h1 = json.loads((ec.MAIN / "h1.json").read_text(encoding="utf-8"))
    controls = {}

    methods = [(load_method("rf", seed=int(s)), None) for s in SEEDS] + [(load_method("combined"), None)]
    methods = [(m, resolve_policy(m, "budget" if m.kind == "heuristic" else "threshold", "first", "0.90")) for m, _ in methods]
    assert abs(methods[-1][1].setting - ec.COMBINED_B) < 1e-12

    # ---- controls: N = 1 equals the open loop on test
    rf0, pol0 = methods[0]
    res = run_loop(hist, a, b, rf0, pol0, 1, keep_features=True, state=start)
    bad = same_features(rows, pd.concat(res.features, ignore_index=True), feats)
    sel_ref = es.matrix(predict(rf0.pipe, rows)) >= pol0.setting
    assert not bad and np.array_equal(res.selected, sel_ref), f"N=1 rf: features {bad[:3]} / selection differ"
    comb, polc = methods[-1]
    res = run_loop(hist, a, b, comb, polc, 1, keep_features=True, state=start)
    from baselines import COMBINED_VARIANTS
    o0 = order_from(es.matrix(COMBINED_VARIANTS["comb_last_cov_mod_recent"](rows)), keys[0])
    sel_c = ec._cum_by_test(o0, es.pred_dur) <= polc.setting * es.pred_dur.sum(1, keepdims=True) + 1e-9
    assert not same_features(rows, pd.concat(res.features, ignore_index=True), feats) and np.array_equal(res.selected, sel_c)
    controls["n1_equals_open_loop"] = "rf seed 20261117 and combined: features of 108 test commits == dataset.csv; selections equal"
    ec.log("controls N = 1 passed")

    recs = []
    for m, pol in methods:
        ol = open_loop(rows, m, pol)
        r = {**loop_metrics(es, ol, defects), "loop": "open", "setting": pol.setting}
        if m.kind == "model":
            i = h1["main"]["seeds"].index(m.seed)
            assert abs(r["recall_first_all"] - h1["main"]["recall_first"]["per_seed"][i]) < 1e-12
            assert abs(r["time_all"] - h1["main"]["time"]["per_seed"][i]) < 1e-9
        recs.append(r)
        for n in N_FULL:
            res = run_loop(hist, a, b, m, pol, n, state=start)
            recs.append({**loop_metrics(es, res, defects), "loop": "closed", "setting": pol.setting})
        ec.log(f"{m.name} seed {m.seed}: open recall {r['recall_first_all']:.3f}, closed N=inf recall {recs[-1]['recall_first_all']:.3f}")
    controls["open_loop_rf_reproduces_8b"] = "per-seed recall_first and time equal h1.json"
    runs = pd.DataFrame(recs)
    runs["n_full"] = runs["n_full"].astype(str)
    lead = ["method", "seed", "loop", "n_full", "setting"]
    runs = runs[lead + [c for c in runs.columns if c not in lead]]

    num = [c for c in runs.columns if c not in lead and pd.api.types.is_numeric_dtype(runs[c])]
    g = runs.groupby(["method", "loop", "n_full"], sort=False)
    summary = g[num].mean().add_suffix("").reset_index()
    sd = g[num].std(ddof=1).reset_index()
    for c in ("recall_first_all", "time_all", "recall_first_selected", "time_selected", "delay_mean", "share_undetected_before_fix"):
        summary[f"{c}_std"] = sd[c].fillna(0.0)
    summary.insert(3, "seeds", g.size().to_numpy())

    ec.write_csv(runs, out / "runs.csv")
    ec.write_csv(summary, out / "summary.csv")
    ec.write_json(out / "summary.json", {
        "experiment": "closed-loop replay (protocol p. 5.6)", "range": [a, b], "start_state": "full runs of commits 0-431",
        "n_full": ["5", "10", "inf"], "methods": {"rf": "frozen RF, tau (first, 0.90) from the card, 10 seeds",
                                                 "combined": f"combined + B = {ec.COMBINED_B} (first, 0.90), ties: permutation-0 key"},
        "open_loop": "selection on dataset rows (history from full runs) = the main evaluation's view; combined with the "
                     "permutation-0 key only, so it differs slightly from 8B's 20-permutation mean",
        "combined_8b_20perm": {"recall_first": next(x["recall_first"]["mean"] for x in h1["comparison_target_0.90"]
                                                    if x["method"] == "combined"),
                               "time": next(x["time"]["mean"] for x in h1["comparison_target_0.90"] if x["method"] == "combined")},
        "summary": summary.to_dict(orient="records"), "controls": controls,
    })
    ec.write_run_info(out, t0, started, ec.standard_inputs(ds) + [ec.ROOT / "data/history-600/runs.jsonl", ec.MAIN / "h1.json",
                      ec.ROOT / "ml/replay.py"] + [ec.MODELS / "rf" / f for f in ec.RF_CARD["seed_files"].values()], SEEDS)
    ec.log(f"done -> {out}")


if __name__ == "__main__":
    main()
