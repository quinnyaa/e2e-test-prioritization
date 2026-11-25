"""Stage 8C, experiment 3: rolling-window evaluation (notes/stage8-protocol.md, p. 5.2).

    ml/.venv/bin/python ml/exp_rolling.py       # -> ml/results/stage8/rolling/, ≈ 8 min

Windows from splits.json (54 commits): expanding and sliding, folds 0-4. In every fold, on the fold's own val:
  * tune(fit, val) for each family (lr, rf, hgb, lgbm), seed 20261117 (one seed per family; protocol section 9);
    the selected configuration is fitted once on `fit` and applied unchanged to `test` (no refit on fit+val:
    tau depends on the score scale, and tau transfer is the question of 8B);
  * the combined variant by mean val APFD over baselines.COMBINED_VARIANTS (stage 6 rule);
  * tau of each model and B of combined / covers by the stage 6 rule (first exposure, target 0.90).
Reported on the fold's test window: APFD (and APFDc, new failures) of every family and heuristic; recall_first and
actual time under tau / B. Mean ± std over folds; how often a chosen configuration lies on a grid edge.
Exploratory (protocol section 9): the val -> test gap of recall_first and time of tau and B in every fold.
Folds 3-4 evaluate on 432-539 and fold 4 tunes on 432-485 (labels of the hold-out test part): a stability check
over time, not an independent confirmation of H1/H2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import exp_common as ec
from baselines import BASELINES, COMBINED_VARIANTS, ORACLE
from dataset_io import SEED
from train import FAMILIES, FitContext, fit_final, predict, tune

SCHEMES = ("expanding", "sliding")
POLICY_HEURISTICS = {"combined": None, "covers": BASELINES["covers"]}


def on_edge(family: str, p: dict) -> list[str]:
    """Ordered grid dimensions whose chosen value is the smallest or largest of the grid (train.grid)."""
    edges = {
        "lr": {"C": (0.01, 10.0)},
        "rf": {"max_depth": (8, None), "min_samples_leaf": (1, 20)},  # depth order 8 < 16 < None (unlimited)
        "hgb": {"max_leaf_nodes": (7, 31), "max_iter": (100, 400)},
        "lgbm": {"num_leaves": (7, 31), "n_estimators": (100, 400)},
    }[family]
    return [f"{k}={p[k]}" for k, v in edges.items() if p[k] in v]


def main() -> None:
    t0, started = ec.started_now()
    out = ec.out_dir("rolling")
    ds = ec.load()
    rows_out, tuning, variants = [], [], []
    for scheme in SCHEMES:
        for f in ds.splits["rolling"][scheme]:
            fold, fit, val, test = f["fold"], tuple(f["fit"]), tuple(f["val"]), tuple(f["test"])
            val_rows, es_val, kv = ec.eval_part(ds, val)
            test_rows, es, kt = ec.eval_part(ds, test)
            info = {"scheme": scheme, "fold": fold, "fit": f"{fit[0]}-{fit[1] - 1}", "val": f"{val[0]}-{val[1] - 1}",
                    "test": f"{test[0]}-{test[1] - 1}", "n_first_val": int(es_val.first_per_commit.sum()),
                    "n_first_test": int(es.first_per_commit.sum()), "red_val": int(es_val.y.any(1).sum()),
                    "red_test": int(es.y.any(1).sum()), "evaluates_on_test_part": test[1] > 432,
                    "tunes_on_test_part": val[1] > 432}

            def add(method, kind, sv, stt, policy=None, setting=None, params=None, edge=""):
                vs, _ = ec.prio(es_val, sv, kv)
                ts, _ = ec.prio(es, stt, kt)
                r = {**info, "method": method, "kind": kind, "params": "" if params is None else str(params), "on_edge": edge,
                     "val_apfd": vs["apfd"], "apfd": ts["apfd"], "apfdc": ts["apfdc"], "apfd_new": ts["apfd_new"],
                     "apfd_repeat": ts["apfd_repeat"], "policy": policy or "", "setting": setting}
                if policy == "threshold":
                    va, _ = ec.threshold_sel(es_val, sv, setting)
                    ta, _ = ec.threshold_sel(es, stt, setting)
                elif policy == "budget":
                    va, _ = ec.budget_sel(es_val, sv, kv, setting)
                    ta, _ = ec.budget_sel(es, stt, kt, setting)
                if policy:
                    for k in ("recall_first", "time", "pred_time", "recall_commit"):
                        r[f"val_{k}"], r[f"test_{k}"] = va[k], ta[k]
                    r["gap_recall_first"] = ta["recall_first"] - va["recall_first"]
                    r["gap_time"] = ta["time"] - va["time"]
                rows_out.append(r)

            # Models.
            for fam in FAMILIES:
                ctx = FitContext(ds, fit, val, "main")
                table, best = tune(ds, fam, "main", fit, val, seed=SEED, ctx=ctx, verbose=False)
                table.insert(0, "scheme", scheme)
                table.insert(1, "fold", fold)
                tuning.append(table)
                pipe = fit_final(ctx, fam, best, SEED)
                sv, stt = predict(pipe, val_rows), predict(pipe, test_rows)
                tau = ec.tau_first(es_val, sv) if info["n_first_val"] else np.nan
                add(fam, "model", sv, stt, "threshold" if tau == tau else None, tau, best, ";".join(on_edge(fam, best)))
            # Heuristics: every baseline's APFD; combined variant chosen on this val; B for combined / covers.
            vrows = []
            for name, fn in COMBINED_VARIANTS.items():
                s, _ = ec.prio(es_val, fn(val_rows), kv)
                vrows.append({"scheme": scheme, "fold": fold, "variant": name, "val_apfd": s["apfd"]})
            vdf = pd.DataFrame(vrows)
            chosen = str(vdf.loc[vdf["val_apfd"].to_numpy().argmax(), "variant"])  # exact tie -> earlier variant
            vdf["selected"] = vdf["variant"] == chosen
            variants.append(vdf)
            for name, fn in BASELINES.items():
                if name == "random":
                    continue  # random APFD is ~0.5 by construction (200 permutations in 8B); not needed here
                sv, stt = fn(val_rows), fn(test_rows)
                if name == "covers":
                    bb = ec.budget_first(es_val, sv, kv)
                    add(name, "baseline", sv, stt, "budget", bb)
                else:
                    add(name, "baseline", sv, stt)
            sv, stt = COMBINED_VARIANTS[chosen](val_rows), COMBINED_VARIANTS[chosen](test_rows)
            add("combined", "baseline", sv, stt, "budget", ec.budget_first(es_val, sv, kv), params=chosen)
            add("oracle", "oracle", ORACLE(val_rows), ORACLE(test_rows))
            ec.log(f"{scheme} fold {fold}: first exposures val {info['n_first_val']} / test {info['n_first_test']}; "
                   + ", ".join(f"{r['method']} {r['apfd']:.3f}" for r in rows_out[-10:] if r["kind"] == "model"))

    per = pd.DataFrame(rows_out)
    num = ["val_apfd", "apfd", "apfdc", "apfd_new", "apfd_repeat", "setting", "val_recall_first", "test_recall_first",
           "val_time", "test_time", "test_pred_time", "gap_recall_first", "gap_time"]
    g = per.groupby(["scheme", "method"], sort=False)
    summary = g[num].mean().reset_index()
    sd = g[num].std(ddof=1).reset_index()
    for c in num:
        summary[f"{c}_std"] = sd[c]
    summary["folds"] = g.size().to_numpy()
    summary["folds_on_edge"] = g["on_edge"].apply(lambda s: int((s != "").sum())).to_numpy()
    summary["test_recall_first_ge_0.90"] = g["test_recall_first"].apply(lambda s: int((s >= 0.90 - 1e-12).sum())).to_numpy()
    summary["test_time_le_0.30"] = g["test_time"].apply(lambda s: int((s <= 0.30 + 1e-12).sum())).to_numpy()
    # Folds not touching the hold-out test part (0-2) separately.
    early = per[~per["evaluates_on_test_part"]].groupby(["scheme", "method"], sort=False)[
        ["apfd", "test_recall_first", "test_time", "gap_recall_first", "gap_time"]].mean().add_suffix("_folds_0_2").reset_index()
    summary = summary.merge(early, on=["scheme", "method"], how="left")

    ec.write_csv(per, out / "per_fold.csv")
    ec.write_csv(summary, out / "summary.csv")
    ec.write_csv(pd.concat(tuning, ignore_index=True).drop(columns=["fit_seconds"]), out / "tuning.csv")  # wall-clock column dropped (reruns byte-identical)
    ec.write_csv(pd.concat(variants, ignore_index=True), out / "combined_variants.csv")
    ec.write_json(out / "summary.json", {
        "experiment": "rolling-window (protocol p. 5.2)", "seed": SEED, "target": 0.90, "criterion": "first",
        "note": "one seed per family; the model fitted on `fit` is applied to `test` without refit; folds 3-4 evaluate "
                "on 432-539 and fold 4 tunes on 432-485 (stability check, not an independent confirmation)",
        "gap_note": "exploratory (protocol section 9): test minus val of recall_first and actual time under the setting "
                    "chosen on the fold's val",
        "summary": summary.to_dict(orient="records"),
    })
    ec.write_run_info(out, t0, started, ec.standard_inputs(ds), [SEED])
    ec.log(f"done -> {out}")


if __name__ == "__main__":
    main()
