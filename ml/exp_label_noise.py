"""Stage 8C, experiment 6: label noise in the training labels (notes/stage8-protocol.md, p. 5.5).

    ml/.venv/bin/python ml/exp_label_noise.py   # -> ml/results/stage8/label_noise/, ≈ 1 min

rho in {0, 0.05, 0.10}. In fit 0-323, k = round(rho |P|) random positives become 0 and k random negatives become 1
(the positive share is unchanged), drawn with default_rng(seed) for noise seed = RF seed 20261117...20261126.
Unchanged: val / test labels and the hist_ features (noise only in the training labels -- a limitation).
RF with the card hyper-parameters; columns constant on the fit part dropped (they do not depend on labels);
tau (first exposure, 0.90) on clean val 324-431; APFD and recall_first / actual time on test 432-539.
Control (fatal): rho = 0 reproduces the frozen RF (test predictions np.array_equal, tau = card) -- that is 8B.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import exp_common as ec
from train import SEEDS, predict

RHOS = (0.0, 0.05, 0.10)


def noisy(fit_rows: pd.DataFrame, rho: float, seed: int) -> tuple[pd.DataFrame, int]:
    y = fit_rows["label"].to_numpy()
    k = int(round(rho * int(y.sum())))
    if k == 0:
        return fit_rows, 0
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    flip_p = rng.choice(pos, size=k, replace=False)
    flip_n = rng.choice(neg, size=k, replace=False)
    out = fit_rows.copy()
    lab = y.copy()
    lab[flip_p], lab[flip_n] = 0, 1
    out["label"] = lab
    assert out["label"].sum() == y.sum()
    return out, k


def main() -> None:
    t0, started = ec.started_now()
    out = ec.out_dir("label_noise")
    ds = ec.load()
    fit_rows = ds.slice(ds.train_range)
    val_rows, es_val, _ = ec.eval_part(ds, ds.val_range)
    test_rows, es, keys = ec.eval_part(ds, ds.test_range)
    y = test_rows["label"].to_numpy()
    kept, dropped = ec.features_on(fit_rows, ds.feature_set("main"))
    recs = []
    for rho in RHOS:
        for s in SEEDS:
            rows, k = noisy(fit_rows, rho, s)
            pipe = ec.fit_rf(rows, kept, s)
            sv, stt = predict(pipe, val_rows), predict(pipe, test_rows)
            tau = ec.tau_first(es_val, sv)
            summ, _ = ec.prio(es, stt, keys)
            agg, _ = ec.threshold_sel(es, stt, tau)
            vagg, _ = ec.threshold_sel(es_val, sv, tau)
            if rho == 0:
                assert np.array_equal(stt, predict(ec.frozen_rf(s), test_rows)) and tau == ec.card_tau(s), f"rho = 0 seed {s} != frozen"
            recs.append({"rho": rho, "seed": s, "flipped_each_way": k, "tau": tau, "apfd": summ["apfd"], "apfdc": summ["apfdc"],
                         "apfd_new": summ["apfd_new"], "apfd_repeat": summ["apfd_repeat"], **ec.diagnostics(y, stt),
                         "recall_first": agg["recall_first"], "time": agg["time"], "pred_time": agg["pred_time"],
                         "val_recall_first": vagg["recall_first"], "val_time": vagg["time"]})
        r = pd.DataFrame(recs[-len(SEEDS):])
        ec.log(f"rho {rho}: APFD {r['apfd'].mean():.3f}, recall_first {r['recall_first'].mean():.3f}, time {100 * r['time'].mean():.1f} %")
    runs = pd.DataFrame(recs)
    num = ["flipped_each_way", "tau", "apfd", "apfdc", "apfd_new", "apfd_repeat", "roc_auc", "pr_auc", "recall_first", "time",
           "pred_time", "val_time"]
    g = runs.groupby("rho", sort=True)
    summary = g[num].mean().reset_index()
    sd = g[num].std(ddof=1).reset_index()
    for c in ("apfd", "recall_first", "time", "tau"):
        summary[f"{c}_std"] = sd[c]
    for c in ("apfd", "recall_first", "time"):
        summary[f"{c}_minus_rho0"] = summary[c] - float(summary.loc[summary["rho"] == 0, c].iloc[0])
    ec.write_csv(runs, out / "runs.csv")
    ec.write_csv(summary, out / "summary.csv")
    ec.write_json(out / "summary.json", {
        "experiment": "label noise (protocol p. 5.5)", "rhos": list(RHOS), "positives_fit": int(fit_rows["label"].sum()),
        "rounding": "k = round(rho * |P|) flips in each direction", "rf_params": ec.RF_PARAMS, "features": kept,
        "dropped_constant": dropped, "controls": {"rho_0_reproduces_frozen_rf": "all 10 seeds"},
        "summary": summary.to_dict(orient="records"),
    })
    ec.write_run_info(out, t0, started, ec.standard_inputs(ds) + [ec.MODELS / "rf" / f for f in ec.RF_CARD["seed_files"].values()], SEEDS)
    ec.log(f"done -> {out}")


if __name__ == "__main__":
    main()
