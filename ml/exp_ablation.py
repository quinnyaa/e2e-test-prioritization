"""Stage 8C, experiment 1: ablation by feature group (notes/stage8-protocol.md, p. 5.1).

    ml/.venv/bin/python ml/exp_ablation.py      # -> ml/results/stage8/ablation/, ≈ 3 min

Sets: main (control), hist, chg, meta, main-hist, main-chg, main-meta, no_commit_chg, main+ci. Subtractive sets
are built here from meta.json -> featureGroups (dataset_io.py unchanged), in mainFeatureSet order. Columns
constant on the fit part are dropped (stage 6 FitContext rule) and listed in the results.
RF with the card hyper-parameters (no re-tuning), fit 0-323, 10 seeds; tau per set and seed chosen on val 324-431
(first exposure, target 0.90); evaluation on test 432-539: APFD (main), APFDc, new / repeated failures,
recall_first and actual time under tau. Comparison with main: Wilcoxon over red commits (per-commit mean over
seeds) + Cliff's delta, Holm within the ablation family (exploratory).
Sensitivity: tune() on val for no_commit_chg and main+ci, then the same evaluation with the tuned parameters.

Controls (fatal): for every seed, the main set reproduces the frozen RF -- test predictions np.array_equal to
ml/models/rf, tau equal to the card, per-commit APFD and tau selection equal to 8B's per_commit_test.csv /
per_commit_policies_test.csv (stored %.6f); no_commit_chg val predictions equal stage 6 rf@no_commit_chg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import exp_common as ec
import stats as st
from train import SEEDS, predict, tune

SETS = ("main", "hist", "chg", "meta", "main-hist", "main-chg", "main-meta", "no_commit_chg", "main+ci")
SENSITIVITY = ("no_commit_chg", "main+ci")


def set_columns(ds, name: str) -> list[str]:
    if name.startswith("main-"):
        drop = set(ds.groups[name.removeprefix("main-")])
        cols = [c for c in ds.feature_set("main") if c not in drop]
        ds._assert_features(cols)
        return cols
    return ds.feature_set(name)


def main() -> None:
    t0, started = ec.started_now()
    out = ec.out_dir("ablation")
    ds = ec.load()
    fit_rows = ds.slice(ds.train_range)
    val_rows, es_val, _ = ec.eval_part(ds, ds.val_range)
    test_rows, es, keys = ec.eval_part(ds, ds.test_range)
    red = es.y.any(1)
    y_test = test_rows["label"].to_numpy()
    s8_pc = pd.read_csv(ec.MAIN / "per_commit_test.csv")
    s6_pred = pd.read_csv(ec.ROOT / "ml" / "results" / "stage6" / "val_predictions.csv")

    per_rows, summ_rows, feat_rows, apfd_red, controls = [], [], [], {}, {}

    def run_set(label: str, cols: list[str], params: dict, check: str | None = None) -> None:
        kept, dropped = ec.features_on(fit_rows, cols)
        feat_rows.append({"set": label, "n_columns": len(cols), "n_kept": len(kept), "dropped_constant": ";".join(dropped),
                          "features": ";".join(kept), "params": str(params)})
        recs = []
        for s in SEEDS:
            pipe = ec.fit_rf(fit_rows, kept, s, params)
            sv, stt = predict(pipe, val_rows), predict(pipe, test_rows)
            tau = ec.tau_first(es_val, sv)
            summ, per = ec.prio(es, stt, keys)
            agg, cols_sel = ec.threshold_sel(es, stt, tau)
            vsum, _ = ec.prio(es_val, sv, ec.tie_keys(es_val, ec.N_PERM))
            vagg, _ = ec.threshold_sel(es_val, sv, tau)  # in-sample on val (tau chosen there)
            if check == "main":
                frozen = ec.frozen_rf(s)
                assert np.array_equal(stt, predict(frozen, test_rows)), f"main seed {s}: test predictions differ from frozen RF"
                assert tau == ec.card_tau(s), f"main seed {s}: tau {tau} != card {ec.card_tau(s)}"
                ref = s8_pc[(s8_pc["method"] == "rf") & (s8_pc["seed"] == s)].sort_values("commit_index")
                for m in ("apfd", "apfdc", "apfd_new", "apfd_repeat"):
                    a, b = ec.fmt6(per[m].fillna(-1)), ec.fmt6(ref[m].fillna(-1))
                    assert (a == b).all(), f"main seed {s}: per-commit {m} differs from 8B"
                pol = ec.main_policy_rows("rf", s, "threshold")
                for m, k in (("first_caught", "sel_first_caught"), ("time", "sel_time"), ("pred_time", "sel_pred_time")):
                    assert (ec.fmt6(cols_sel[k]) == ec.fmt6(pol[m])).all(), f"main seed {s}: tau selection {m} differs from 8B"
            if check == "no_commit_chg":
                ref = s6_pred[(s6_pred["model"] == "rf@no_commit_chg") & (s6_pred["seed"] == s)].sort_values(["commit_index", "test_id"])
                diff = float(np.abs(ref["score"].to_numpy() - sv).max())
                controls.setdefault("no_commit_chg_vs_stage6_val_max_abs_diff", {})[str(s)] = diff
                assert diff < 1e-12, f"no_commit_chg seed {s}: val predictions differ from stage 6 ({diff})"
            d = ec.diagnostics(y_test, stt)
            recs.append({"seed": s, "tau": tau, "val_apfd": vsum["apfd"], "apfd": summ["apfd"], "apfdc": summ["apfdc"],
                         "apfd_new": summ["apfd_new"], "apfd_repeat": summ["apfd_repeat"], "apfd_defect": summ["apfd_defect"],
                         **d, "recall_first": agg["recall_first"], "recall_defect": agg["recall_defect"],
                         "recall_commit": agg["recall_commit"], "time": agg["time"], "time_red": agg["time_red"],
                         "pred_time": agg["pred_time"], "tests": agg["tests"],
                         "val_recall_first": vagg["recall_first"], "val_time": vagg["time"]})
            per_rows.append(pd.DataFrame({
                "set": label, "seed": s, "commit_index": es.commits, "n_fail": es.y.sum(1), "n_new": es.new.sum(1),
                "n_first": es.first_per_commit, "apfd": per["apfd"], "apfdc": per["apfdc"], "apfd_new": per["apfd_new"],
                "apfd_repeat": per["apfd_repeat"], "tau_first_caught": cols_sel["sel_first_caught"],
                "tau_time": cols_sel["sel_time"], "tau_pred_time": cols_sel["sel_pred_time"]}))
            apfd_red.setdefault(label, []).append(per["apfd"].to_numpy()[red])
        r = pd.DataFrame(recs)
        row = {"set": label, "n_features": len(kept)}
        for m in ("val_apfd", "apfd", "apfdc", "apfd_new", "apfd_repeat", "apfd_defect", "roc_auc", "pr_auc", "tau",
                  "recall_first", "recall_defect", "recall_commit", "time", "time_red", "pred_time", "tests",
                  "val_recall_first", "val_time"):
            row[m], row[f"{m}_std"] = ec.mean_std(r[m])
        summ_rows.append(row)
        ec.log(f"{label}: {len(kept)} features, APFD {row['apfd']:.3f}, recall_first {row['recall_first']:.3f}, "
               f"time {100 * row['time']:.1f} %")

    for name in SETS:
        run_set(name, set_columns(ds, name), ec.RF_PARAMS, check=name if name in ("main", "no_commit_chg") else None)
    controls["main_reproduces_8b"] = "all 10 seeds: test predictions, tau, per-commit APFD/APFDc/new/repeat and tau selection equal"

    # Sensitivity: hyper-parameters re-tuned on val (fit 0-323) for two sets.
    tuning = []
    for name in SENSITIVITY:
        table, best = tune(ds, "rf", name, ds.train_range, ds.val_range, verbose=False)
        tuning.append(table)
        run_set(f"{name}@tuned", set_columns(ds, name), best)
        summ_rows[-1]["params"] = str(best)

    summary = pd.DataFrame(summ_rows)
    # Comparisons with main (exploratory): per-commit APFD over red commits, mean over seeds.
    base = np.mean(apfd_red["main"], axis=0)
    cmp_rows = []
    for label in [s for s in summary["set"] if s != "main"]:
        a = np.mean(apfd_red[label], axis=0)
        w = st.wilcoxon(a, base)
        dlt = st.cliffs_delta(a, base)
        cmp_rows.append({"set": label, "reference": "main", "family": "ablation" if "@" not in label else "sensitivity",
                         **w, "mean_set": float(a.mean()), "mean_main": float(base.mean()), "delta": dlt, "magnitude": st.romano(dlt)})
    cmp = pd.DataFrame(cmp_rows)
    cmp["p_holm"] = np.nan
    fam = cmp["family"] == "ablation"
    cmp.loc[fam, "p_holm"] = st.holm(cmp.loc[fam, "p"].to_numpy())
    cmp.loc[~fam, "p_holm"] = st.holm(cmp.loc[~fam, "p"].to_numpy())

    ec.write_csv(pd.concat(per_rows, ignore_index=True), out / "per_commit.csv")
    ec.write_csv(summary, out / "summary.csv")
    ec.write_csv(cmp, out / "comparisons.csv")
    ec.write_csv(pd.DataFrame(feat_rows), out / "features.csv")
    ec.write_csv(pd.concat(tuning, ignore_index=True).drop(columns=["fit_seconds"]), out / "tuning_sensitivity.csv")  # wall-clock column dropped (reruns byte-identical)
    ec.write_json(out / "summary.json", {
        "experiment": "ablation (protocol p. 5.1)", "fit": list(ds.train_range), "tau_on": list(ds.val_range),
        "evaluated_on": list(ds.test_range), "rf_params": ec.RF_PARAMS, "seeds": list(SEEDS),
        "sets": summary.to_dict(orient="records"), "comparisons_with_main": cmp.to_dict(orient="records"),
        "controls": controls,
        "note": "exploratory comparisons; verdicts of H1/H2 are not revised",
    })
    ec.write_run_info(out, t0, started, ec.standard_inputs(ds) + [ec.MAIN / "per_commit_test.csv",
                      ec.MAIN / "per_commit_policies_test.csv", ec.ROOT / "ml/results/stage6/val_predictions.csv"]
                      + [ec.MODELS / "rf" / f for f in ec.RF_CARD["seed_files"].values()], SEEDS)
    ec.log(f"done -> {out}")


if __name__ == "__main__":
    main()
