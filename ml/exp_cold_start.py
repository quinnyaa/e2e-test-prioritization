"""Stage 8C, experiment 5: cold start of new tests (notes/stage8-protocol.md, p. 5.4; section 9 for the parameters).

    ml/.venv/bin/python ml/exp_cold_start.py    # -> ml/results/stage8/cold_start/, ≈ 5 min

Cold tests (rule without looking at labels): ADM-01, AUTH-01, CART-01, CAT-01, CHK-01, ORD-01, PRD-01, SRCH-01,
E2E-01. Frozen RF (10 seeds), tau from the card unchanged. Replay over 432-539 from full runs of 0-431 with a full
run on every commit (n_full = 1: "the history then accumulates from full runs"), so only the cold tests' state
differs from the main evaluation:
  (a) reset at commit 432, history accumulates afterwards;   (b) reset before every commit (worst case).
Default parameters peer_scope = "own_only", rate_denominator = "since_reset"; the alternatives "remove" and
"global" only as sensitivity rows of the same table.
Metrics (of the *selection*; replay.loop_metrics on a LoopResult whose executed = selected; its delay fields
describe a selection, not a detection, and are dropped): recall of cold tests' failures (test level),
recall_first over first-exposure pairs caught only by cold tests, overall recall_first and time; the same for
the main evaluation (frozen predictions on dataset rows, tau).
Control (fatal): an empty cold list gives the main evaluation's selection.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

import exp_common as ec
from prioritize import load_method, resolve_policy
from replay import COLD_TESTS, History, loop_metrics, run_loop
from train import SEEDS, predict

CONFIGS = (("a", "own_only", "since_reset", "main"), ("b", "own_only", "since_reset", "main"),
           ("a", "remove", "since_reset", "sensitivity"), ("b", "remove", "since_reset", "sensitivity"),
           ("a", "own_only", "global", "sensitivity"), ("b", "own_only", "global", "sensitivity"))
DROP = ("delay_mean", "delay_median", "delay_max", "detected", "detected_at_first_exposure", "undetected_before_fix",
        "undetected_alive_at_end", "share_undetected_before_fix", "recall_first_selected", "time_selected", "defects")


def selection_metrics(es, defects, res, cold_mask, cold_only) -> dict:
    as_exec = dataclasses.replace(res, executed=res.selected, full=np.zeros_like(res.full))
    m = {k: v for k, v in loop_metrics(es, as_exec, defects, cold=COLD_TESTS).items() if k not in DROP}
    caught = ec.per_defect_caught(es, [res.selected])
    m["cold_only_first_pairs"] = int(cold_only.sum())
    m["recall_first_cold_only"] = float(caught[cold_only].mean()) if cold_only.any() else np.nan
    m["cold_tests_selected_share"] = float(res.selected[:, cold_mask].mean())
    return m


def main() -> None:
    t0, started = ec.started_now()
    out = ec.out_dir("cold_start")
    ds = ec.load()
    a, b = ds.test_range
    rows, es, _ = ec.eval_part(ds, (a, b))
    defects = ds.defects
    hist = History.load(b, allow_test_history=True)
    cold_mask = np.isin(es.tests, COLD_TESTS)
    assert cold_mask.sum() == len(COLD_TESTS)
    # First-exposure pairs whose failing tests are all cold (caught only by cold tests).
    cold_only = es.def_first & es.def_mask.any(1) & ~(es.def_mask & ~cold_mask[None, :]).any(1)
    states = {(ps, rd): hist.state_before(a, peer_scope=ps, rate_denominator=rd)
              for ps, rd in {(c[1], c[2]) for c in CONFIGS}}
    recs = []
    for s in SEEDS:
        m = load_method("rf", seed=int(s))
        pol = resolve_policy(m, "threshold", "first", "0.90")
        sel_main = es.matrix(predict(m.pipe, rows)) >= pol.setting
        main_res = run_loop(hist, a, b, m, pol, 1, state=states[("own_only", "since_reset")])  # empty cold list
        assert np.array_equal(main_res.selected, sel_main), f"seed {s}: empty cold list != main evaluation"
        recs.append({"seed": int(s), "variant": "main evaluation (8B)", "peer_scope": "", "rate_denominator": "", "role": "reference",
                     **selection_metrics(es, defects, main_res, cold_mask, cold_only)})
        for variant, ps, rd, role in CONFIGS:
            res = run_loop(hist, a, b, m, pol, 1, cold=COLD_TESTS, cold_variant=variant, state=states[(ps, rd)])
            recs.append({"seed": int(s), "variant": variant, "peer_scope": ps, "rate_denominator": rd, "role": role,
                         **selection_metrics(es, defects, res, cold_mask, cold_only)})
        ec.log(f"seed {s}: cold recall main {recs[-7]['cold_test_recall']:.3f}, (a) {recs[-6]['cold_test_recall']:.3f}, "
               f"(b) {recs[-5]['cold_test_recall']:.3f}")
    runs = pd.DataFrame(recs).drop(columns=["method"])
    keys_ = ["variant", "peer_scope", "rate_denominator", "role"]
    num = ["cold_test_recall", "recall_first_cold_only", "recall_first_all", "time_all", "recall_test_all", "cold_tests_selected_share"]
    g = runs.groupby(keys_, sort=False)
    summary = g[num].mean().reset_index()
    sd = g[num].std(ddof=1).reset_index()
    for c in num:
        summary[f"{c}_std"] = sd[c]
    summary["cold_only_first_pairs"] = int(cold_only.sum())
    cold_fail = int((es.y & cold_mask[None, :]).sum())
    ec.write_csv(runs, out / "runs.csv")
    ec.write_csv(summary, out / "summary.csv")
    ec.write_json(out / "summary.json", {
        "experiment": "cold start (protocol p. 5.4)", "cold_tests": list(COLD_TESTS), "range": [a, b], "n_full": 1,
        "failures_of_cold_tests": cold_fail, "first_pairs_cold_only": int(cold_only.sum()),
        "first_pairs_cold_only_ids": sorted(set(es.def_ids[cold_only])),
        "first_pairs_with_any_cold_failing_test": int((es.def_first & (es.def_mask & cold_mask[None, :]).any(1)).sum()),
        "note": "metrics of the selection (score >= tau); delay fields of loop_metrics dropped (no closed loop here)",
        "controls": {"empty_cold_list_equals_main_selection": "all 10 seeds"},
        "summary": summary.to_dict(orient="records"),
    })
    ec.write_run_info(out, t0, started, ec.standard_inputs(ds) + [ec.ROOT / "data/history-600/runs.jsonl", ec.ROOT / "ml/replay.py"]
                      + [ec.MODELS / "rf" / f for f in ec.RF_CARD["seed_files"].values()], SEEDS)
    ec.log(f"done -> {out}")


if __name__ == "__main__":
    main()
