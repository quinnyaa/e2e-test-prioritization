"""Stage 8C, experiment 4: learning curve (notes/stage8-protocol.md, p. 5.3).

    ml/.venv/bin/python ml/exp_learning_curve.py   # -> ml/results/stage8/learning_curve/, ≈ 2 min

Main history (history-600-540): fit prefixes [0, 81), [0, 162), [0, 243), [0, 324); RF with the card
hyper-parameters, 10 seeds; columns constant on each prefix dropped (stage 6 rule, applied here directly:
make_pipeline + fit_pipeline via exp_common.fit_rf, because FitContext is not needed for a fixed val);
tau (first exposure, 0.90) on val 324-431; APFD and recall_first / actual time on test 432-539.
Control (fatal): the prefix [0, 324) reproduces the frozen RF (test predictions np.array_equal, tau = card).
Auxiliary history (data/dataset/history/, 300 commits; descriptive): the same shares (1/4 ... 4/4) of its train
0-179, tau on its val 180-239, evaluation on its test 240-299, with the same card hyper-parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import exp_common as ec
from train import SEEDS, predict

SHARES = (0.25, 0.5, 0.75, 1.0)
AUX = ec.ROOT / "data" / "dataset" / "history"


def curve(ds, history: str, control: bool) -> tuple[list[dict], list[dict]]:
    (ta, tb), val, test = ds.train_range, ds.val_range, ds.test_range
    val_rows, es_val, _ = ec.eval_part(ds, val)
    test_rows, es, keys = ec.eval_part(ds, test)
    y = test_rows["label"].to_numpy()
    recs, feat = [], []
    for share in SHARES:
        end = ta + int(round(share * (tb - ta)))
        fit_rows = ds.slice((ta, end))
        kept, dropped = ec.features_on(fit_rows, ds.feature_set("main"))
        feat.append({"history": history, "fit": f"{ta}-{end - 1}", "n_kept": len(kept), "dropped_constant": ";".join(dropped)})
        for s in SEEDS:
            pipe = ec.fit_rf(fit_rows, kept, s)
            sv, stt = predict(pipe, val_rows), predict(pipe, test_rows)
            tau = ec.tau_first(es_val, sv)
            summ, _ = ec.prio(es, stt, keys)
            agg, _ = ec.threshold_sel(es, stt, tau)
            if control and end == tb:
                assert np.array_equal(stt, predict(ec.frozen_rf(s), test_rows)), f"seed {s}: prefix [0, 324) != frozen RF"
                assert tau == ec.card_tau(s), f"seed {s}: tau != card"
            recs.append({"history": history, "fit": f"{ta}-{end - 1}", "fit_commits": end - ta, "share": share, "seed": s,
                         "positives_fit": int(fit_rows["label"].sum()), "tau": tau, "apfd": summ["apfd"], "apfdc": summ["apfdc"],
                         "apfd_new": summ["apfd_new"], "apfd_repeat": summ["apfd_repeat"], **ec.diagnostics(y, stt),
                         "recall_first": agg["recall_first"], "time": agg["time"], "pred_time": agg["pred_time"],
                         "first_exposures_test": int(es.first_per_commit.sum()), "first_exposures_val": int(es_val.first_per_commit.sum())})
        r = pd.DataFrame(recs[-len(SEEDS):])
        ec.log(f"{history} fit {ta}-{end - 1}: APFD {r['apfd'].mean():.3f}, recall_first {r['recall_first'].mean():.3f}, "
               f"time {100 * r['time'].mean():.1f} %")
    return recs, feat


def main() -> None:
    t0, started = ec.started_now()
    out = ec.out_dir("learning_curve")
    ds = ec.load()
    aux = ec.load(AUX)
    r1, f1 = curve(ds, "history-600-540", True)
    r2, f2 = curve(aux, "history (300, auxiliary)", False)
    runs = pd.DataFrame(r1 + r2)
    num = ["positives_fit", "tau", "apfd", "apfdc", "apfd_new", "apfd_repeat", "roc_auc", "pr_auc", "recall_first", "time", "pred_time"]
    g = runs.groupby(["history", "fit", "fit_commits", "share"], sort=False)
    summary = g[num].mean().reset_index()
    sd = g[num].std(ddof=1).reset_index()
    for c in ("apfd", "recall_first", "time", "tau"):
        summary[f"{c}_std"] = sd[c]
    ec.write_csv(runs, out / "runs.csv")
    ec.write_csv(summary, out / "summary.csv")
    ec.write_csv(pd.DataFrame(f1 + f2), out / "features.csv")
    ec.write_json(out / "summary.json", {
        "experiment": "learning curve (protocol p. 5.3)", "rf_params": ec.RF_PARAMS, "seeds": list(SEEDS),
        "main": {"fit_prefixes": "[0,81) [0,162) [0,243) [0,324)", "tau_on": list(ds.val_range), "evaluated_on": list(ds.test_range)},
        "auxiliary": {"dataset": "data/dataset/history", "fit_prefixes": "[0,45) [0,90) [0,135) [0,180)",
                      "tau_on": list(aux.val_range), "evaluated_on": list(aux.test_range),
                      "note": "descriptive; hyper-parameters of the main card, not tuned on the auxiliary history"},
        "controls": {"prefix_0_324_reproduces_frozen_rf": "all 10 seeds: test predictions and tau equal"},
        "summary": summary.to_dict(orient="records"),
    })
    ec.write_run_info(out, t0, started, ec.standard_inputs(ds) + [AUX / f for f in ("dataset.csv", "meta.json", "splits.json",
                      "positives.csv", "defects.csv")] + [ec.MODELS / "rf" / f for f in ec.RF_CARD["seed_files"].values()], SEEDS)
    ec.log(f"done -> {out}")


if __name__ == "__main__":
    main()
