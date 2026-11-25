"""Stage 8C, experiment 9: overhead of the tool (notes/stage8-protocol.md, p. 5.9).

    ml/.venv/bin/python ml/exp_overhead.py      # -> ml/results/stage8/overhead/, ≈ 1 min

Measured in one warm process, on every test commit 432-539, the CLI path of `rank` (ml/prioritize.py,
ml/features_online.py): features from the history state -> frozen RF score (seed 20261117, single-threaded
predict as saved) -> ranking -> tau selection. 30 repeats per commit; median and p95. After the timed repeats the
state is updated with the commit's full run (not timed), as the CLI history replay does. Model loading
(joblib.load of model.joblib) is timed 30 times separately. The diff of the commit is read from the replay cache
(the CLI measures it separately: notes/cli.md). Playwright is not run.
Compared with: the sequential sum of the commit's actual test durations (eval_duration_ms, the time unit of all
metrics) and ≈ 7.5 s of a real full run with 4 workers (notes/cli.md).
Timings depend on the machine and load: these outputs are exempt from the byte-identical rerun requirement.
"""

from __future__ import annotations

import os
import platform
import time

import joblib
import numpy as np
import pandas as pd

import exp_common as ec
from features_online import HistoryState
from prioritize import load_method, rank_order, resolve_policy, scores, select
from replay import History

REPEATS = 30
FULL_RUN_4_WORKERS_S = 7.5


def main() -> None:
    t0, started = ec.started_now()
    out = ec.out_dir("overhead")
    ds = ec.load()
    a, b = ds.test_range
    rows = ds.slice((a, b))
    seq = rows.groupby("commit_index")["eval_duration_ms"].sum()
    hist = History.load(b, allow_test_history=True)
    state = HistoryState()
    for run, diff in zip(hist.runs[:a], hist.diffs[:a]):
        state.update(diff, run)
    m = load_method("rf")
    pol = resolve_policy(m, "threshold", "first", "0.90")
    loads = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        joblib.load(ec.MODELS / "rf" / "model.joblib")
        loads.append(time.perf_counter() - t)
    recs = []
    for c in range(a, b):
        run, diff = hist.runs[c], hist.diffs[c]
        comp = {"features": [], "score": [], "rank_select": [], "total": []}
        for _ in range(REPEATS):
            t1 = time.perf_counter()
            f = state.features(diff, run["index"], run["sha"], hist.ci(c))
            t2 = time.perf_counter()
            s = scores(m, f)
            t3 = time.perf_counter()
            sel = select(pol, s, rank_order(s, c), f["meta_duration_mean"].to_numpy(dtype=float))
            t4 = time.perf_counter()
            comp["features"].append(t2 - t1), comp["score"].append(t3 - t2), comp["rank_select"].append(t4 - t3)
            comp["total"].append(t4 - t1)
        state.update(diff, run)
        r = {"commit_index": c, "selected": int(sel.sum()), "full_run_sequential_ms": float(seq.loc[c])}
        for k, v in comp.items():
            r[f"{k}_median_ms"] = 1000 * float(np.median(v))
            r[f"{k}_p95_ms"] = 1000 * float(np.percentile(v, 95))
        r["rank_share_of_sequential_run"] = r["total_median_ms"] / r["full_run_sequential_ms"]
        recs.append(r)
    per = pd.DataFrame(recs)
    summ = {
        "experiment": "overhead (protocol p. 5.9)", "repeats": REPEATS, "commits": int(len(per)),
        "rank_ms": {k: {"median_over_commits_of_median": float(per[f"{k}_median_ms"].median()),
                        "max_over_commits_of_median": float(per[f"{k}_median_ms"].max()),
                        "p95_over_commits_of_p95": float(np.percentile(per[f"{k}_p95_ms"], 95))}
                    for k in ("features", "score", "rank_select", "total")},
        "model_load_ms": {"median": 1000 * float(np.median(loads)), "p95": 1000 * float(np.percentile(loads, 95))},
        "full_run_sequential_ms": {"median": float(per["full_run_sequential_ms"].median()), "mean": float(per["full_run_sequential_ms"].mean())},
        "full_run_4_workers_s": FULL_RUN_4_WORKERS_S,
        "rank_share_of_sequential_run_median": float(per["rank_share_of_sequential_run"].median()),
        "rank_share_of_4_worker_run": float(per["total_median_ms"].median()) / (1000 * FULL_RUN_4_WORKERS_S),
        "platform": {"system": platform.platform(), "processor": platform.processor(), "machine": platform.machine(),
                     "cpu_count": os.cpu_count(), "rf_predict_n_jobs": int(m.pipe[-1].n_jobs), "python": platform.python_version()},
        "note": "timings depend on the machine; not part of the byte-identical rerun check",
    }
    ec.write_csv(per, out / "per_commit.csv")
    ec.write_json(out / "summary.json", summ)
    ec.write_run_info(out, t0, started, ec.standard_inputs(ds) + [ec.MODELS / "rf" / "model.joblib"], [ec.SEEDS[0]],
                      {"deterministic_outputs": False})
    ec.log(f"rank median {summ['rank_ms']['total']['median_over_commits_of_median']:.1f} ms, model load "
           f"{summ['model_load_ms']['median']:.0f} ms, sequential full run median {summ['full_run_sequential_ms']['median'] / 1000:.1f} s")


if __name__ == "__main__":
    main()
