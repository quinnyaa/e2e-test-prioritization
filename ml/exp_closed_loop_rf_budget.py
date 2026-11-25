"""Post-hoc, exploratory addition to the closed-loop replay (supervisor's review 2026-11-25; logged in
notes/stage8-protocol.md, section 9).

    ml/.venv/bin/python ml/exp_closed_loop_rf_budget.py   # -> ml/results/stage8/closed_loop_rf_budget/

The 8C closed loop (ml/exp_closed_loop.py) compares RF + tau with combined + B, so the ranking and the selection
rule change together. Here the frozen RF ranking is run with the frozen budget policy B (first exposure, target
0.90, from the model card), with the same start state, the same N in {5, 10, inf} and the same metrics, so that
RF + tau vs RF + B isolates the selection rule. Nothing is tuned; 8C results are not rewritten.
"""

from __future__ import annotations

import json

import pandas as pd

import exp_common as ec
from exp_closed_loop import N_FULL
from prioritize import load_method, resolve_policy
from replay import History, loop_metrics, open_loop, run_loop
from train import SEEDS


def main() -> None:
    t0, started = ec.started_now()
    out = ec.out_dir("closed_loop_rf_budget")
    ds = ec.load()
    a, b = ds.test_range
    rows, es, keys = ec.eval_part(ds, (a, b))
    hist = History.load(b, allow_test_history=True)
    start = hist.state_before(a)

    recs = []
    for s in SEEDS:
        m = load_method("rf", seed=int(s))
        pol = resolve_policy(m, "budget", "first", "0.90")
        ol = open_loop(rows, m, pol)
        recs.append({**loop_metrics(es, ol, ds.defects), "loop": "open", "setting": pol.setting})
        for n in N_FULL:
            res = run_loop(hist, a, b, m, pol, n, state=start)
            recs.append({**loop_metrics(es, res, ds.defects), "loop": "closed", "setting": pol.setting})
        ec.log(f"rf seed {s} B = {pol.setting:.3f}: open {recs[-4]['recall_first_all']:.3f}, closed N=inf {recs[-1]['recall_first_all']:.3f}")

    runs = pd.DataFrame(recs)
    runs["n_full"] = runs["n_full"].astype(str)
    lead = ["method", "seed", "loop", "n_full", "setting"]
    runs = runs[lead + [c for c in runs.columns if c not in lead]]
    num = [c for c in runs.columns if c not in lead and pd.api.types.is_numeric_dtype(runs[c])]
    g = runs.groupby(["method", "loop", "n_full"], sort=False)
    summary = g[num].mean().reset_index()
    sd = g[num].std(ddof=1).reset_index()
    for c in ("recall_first_all", "time_all", "delay_mean", "share_undetected_before_fix"):
        summary[f"{c}_std"] = sd[c].fillna(0.0)
    summary.insert(3, "seeds", g.size().to_numpy())

    h1 = json.loads((ec.MAIN / "h1.json").read_text(encoding="utf-8"))
    ref = next(x for x in h1["comparison_target_0.90"] if x["method"] == "rf" and x["policy"] == "budget")
    ec.write_csv(runs, out / "runs.csv")
    ec.write_csv(summary, out / "summary.csv")
    ec.write_json(out / "summary.json", {
        "experiment": "closed-loop replay, RF + budget B (post-hoc, exploratory; review 2026-11-25)",
        "range": [a, b], "start_state": "full runs of commits 0-431", "n_full": ["5", "10", "inf"],
        "method": "frozen RF ranking, budget B (first, 0.90) from the model card, 10 seeds",
        "rf_budget_8b": {"recall_first": ref["recall_first"]["mean"], "time": ref["time"]["mean"]},
        "summary": summary.to_dict(orient="records"),
    })
    ec.write_run_info(out, t0, started, ec.standard_inputs(ds) + [ec.ROOT / "data/history-600/runs.jsonl", ec.MAIN / "h1.json",
                      ec.ROOT / "ml/replay.py"] + [ec.MODELS / "rf" / f for f in ec.RF_CARD["seed_files"].values()], SEEDS)
    ec.log(f"done -> {out}")


if __name__ == "__main__":
    main()
