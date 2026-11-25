"""Stage 8D (exploratory, protocol section 9, entry of 8B): recall of first exposures vs actual time on test.

    ml/.venv/bin/python ml/exp_cost_curve.py  # -> ml/results/stage8/cost_curve/, ≈ 1 min

NOT a test of H1: every setting of the sweep is evaluated on the same test data it is read from, so the read-offs
describe the cost / recall trade-off of each ranking, not a frozen policy. Frozen models (train 0-323), frozen
Kim-Porter alpha (stage8-dryrun/kp.json); nothing is trained or tuned.

Curves (one point per setting; selections are nested in the setting, so time and recall are monotone):
  RF            threshold tau over every distinct score of the seed on test (exact curve), and budget B;
  LightGBM, HGB, LR, combined, covers, Kim-Porter, oracle, random   budget B (policy.BUDGET_GRID, 0.01 ... 1.00);
budgets use the 20 tie permutations of 8B (random: 200). Each curve is anchored at (0, 0); B = 1 and the lowest
tau select every test (1, 1).
Read-offs (rule fixed here, before the curves were looked at):
  recall at actual time t in {0.20, 0.30, 0.40, 0.50}: linear interpolation of recall_first over the curve points
  sorted by time (for equal times the largest recall); seeded methods: per seed, then mean ± std;
  min_time_recall_0.90: the smallest actual time of a realised curve point with recall_first >= 0.90.
Fatal controls: at the frozen settings the curves reproduce main/policies_test.csv (recall_first and time, %.6f):
RF tau of the card (10 seeds), budgets of every model seed, combined / covers / Kim-Porter / oracle B.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

import baseline_kp as kp
import exp_common as ec
from baselines import BASELINES, COMBINED_VARIANTS, ORACLE
from metrics import N_PERM_RANDOM
from policy import BUDGET_GRID
from train import predict

READ_TIMES = (0.20, 0.30, 0.40, 0.50)
TARGET = 0.90
TIME_GRID = np.round(np.linspace(0, 1, 201), 4)  # common grid for averaging seed curves (plots only)
FAMILIES = ("rf", "lgbm", "hgb", "lr")
KP_FILE = ec.ROOT / "ml" / "results" / "stage8-dryrun" / "kp.json"
LABEL_UK = {"rf_tau": "Random Forest, поріг τ", "rf_B": "Random Forest, бюджет B", "lgbm_B": "LightGBM, бюджет B",
            "hgb_B": "HistGradientBoosting, бюджет B", "lr_B": "логістична регресія, бюджет B",
            "combined_B": "комбінована евристика, бюджет B", "covers_B": "covers ∩ діф, бюджет B",
            "kp_B": "Кім–Портер, бюджет B", "oracle_B": "оракул, бюджет B", "random_B": "випадковий порядок, бюджет B"}


def budget_points(es, scores, keys) -> pd.DataFrame:
    rows = [{"setting": 0.0, "recall_first": 0.0, "time": 0.0}]
    for b in BUDGET_GRID:
        agg, _ = ec.budget_sel(es, scores, keys, float(b))
        rows.append({"setting": float(b), "recall_first": agg["recall_first"], "time": agg["time"]})
    return pd.DataFrame(rows)


def threshold_points(es, scores) -> pd.DataFrame:
    rows = [{"setting": float("inf"), "recall_first": 0.0, "time": 0.0}]
    for t in np.unique(scores)[::-1]:
        agg, _ = ec.threshold_sel(es, scores, float(t))
        rows.append({"setting": float(t), "recall_first": agg["recall_first"], "time": agg["time"]})
    return pd.DataFrame(rows)


def envelope(pts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Curve points sorted by time; for equal times the largest recall."""
    g = pts.groupby("time", sort=True)["recall_first"].max()
    return g.index.to_numpy(dtype=float), g.to_numpy(dtype=float)


def readouts(pts: pd.DataFrame) -> dict:
    t, r = envelope(pts)
    out = {f"recall_at_{int(x * 100)}": float(np.interp(x, t, r)) for x in READ_TIMES}
    ok = pts[pts["recall_first"] >= TARGET - 1e-12]
    out["min_time_recall_0.90"] = float(ok["time"].min()) if len(ok) else float("nan")
    return out


def frozen_rows(method: str, policy: str, seed: int) -> pd.Series:
    p = pd.read_csv(ec.MAIN / "policies_test.csv")
    m = (p["method"] == method) & (p["policy"] == policy) & (p["seed"] == seed) & (p["criterion"] == "first") & np.isclose(p["target"], 0.9)
    assert m.sum() == 1, (method, policy, seed)
    return p[m].iloc[0]


def check_frozen_point(pts: pd.DataFrame, method: str, policy: str, seed: int, es=None, scores=None) -> None:
    ref = frozen_rows(method, policy, seed)
    if policy == "budget":
        row = pts[np.isclose(pts["setting"], ref["setting"])].iloc[0]
        got = (row["recall_first"], row["time"])
    else:
        agg, _ = ec.threshold_sel(es, scores, float(ref["setting"]))
        got = (agg["recall_first"], agg["time"])
    assert (ec.fmt6(got) == ec.fmt6([ref["recall_first"], ref["time"]])).all(), (method, policy, seed, got, ref[["recall_first", "time"]].tolist())


def main() -> None:
    t0, started = ec.started_now()
    out = ec.out_dir("cost_curve")
    ds = ec.load()
    rng = ds.test_range
    rows = ds.slice(rng)
    es = ec.build_eval_set(rows, ds.positives_in(rng), ds.introduced_at)
    keys200 = ec.tie_keys(es, N_PERM_RANDOM)
    keys = keys200[:ec.N_PERM]

    curves: list[pd.DataFrame] = []

    def add(name: str, seed: int, pts: pd.DataFrame) -> None:
        curves.append(pts.assign(method=name, seed=seed)[["method", "seed", "setting", "recall_first", "time"]])

    inputs = []
    for fam in FAMILIES:
        card = json.loads((ec.MODELS / fam / "model_card.json").read_text(encoding="utf-8"))
        for s in card["seeds"]:
            f = ec.MODELS / fam / card["seed_files"][str(s)]
            inputs.append(f)
            sc = predict(joblib.load(f), rows)
            pts = budget_points(es, sc, keys)
            check_frozen_point(pts, fam, "budget", int(s))
            add(f"{fam}_B", int(s), pts)
            if fam == "rf":
                check_frozen_point(None, "rf", "threshold", int(s), es, sc)
                add("rf_tau", int(s), threshold_points(es, sc))
        ec.log(f"{fam}: {len(card['seeds'])} seed(s)")
    heur = {"combined": COMBINED_VARIANTS["comb_last_cov_mod_recent"](rows), "covers": BASELINES["covers"](rows), "oracle": ORACLE(rows)}
    alpha = float(json.loads(KP_FILE.read_text(encoding="utf-8"))["alpha"])
    heur["kp"] = kp.kp_scores(ds.slice((0, rng[1])), rng, alpha)
    for name, sc in heur.items():
        pts = budget_points(es, sc, keys)
        check_frozen_point(pts, name, "budget", -1)
        add(f"{name}_B", -1, pts)
    add("random_B", -1, budget_points(es, BASELINES["random"](rows), keys200))
    ec.log("frozen settings reproduce main/policies_test.csv (recall_first, time)")

    cur = pd.concat(curves, ignore_index=True)
    per_seed = []
    for (m, s), g in cur.groupby(["method", "seed"], sort=False):
        per_seed.append({"method": m, "seed": s, "points": len(g), **readouts(g)})
    per_seed = pd.DataFrame(per_seed)
    cols = [c for c in per_seed.columns if c.startswith(("recall_at", "min_time"))]
    summ = []
    for m, g in per_seed.groupby("method", sort=False):
        row = {"method": m, "label": LABEL_UK[m], "seeds": len(g)}
        for c in cols:
            v = g[c].to_numpy(dtype=float)
            row[c] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            row[f"{c}_std"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else 0.0
            row[f"{c}_seeds_defined"] = int(np.isfinite(v).sum())
        summ.append(row)
    summ = pd.DataFrame(summ)
    mean_curves = []
    for (m, s), g in cur.groupby(["method", "seed"], sort=False):
        t, r = envelope(g)
        mean_curves.append(pd.DataFrame({"method": m, "seed": s, "time": TIME_GRID, "recall_first": np.interp(TIME_GRID, t, r)}))
    mean_curves = pd.concat(mean_curves).groupby(["method", "time"], sort=False)["recall_first"].agg(["mean", "std"]).reset_index()

    ec.write_csv(cur, out / "curves.csv")
    ec.write_csv(per_seed, out / "readouts_per_seed.csv")
    ec.write_csv(summ, out / "readouts.csv")
    ec.write_csv(mean_curves.fillna(0.0), out / "mean_curves.csv")
    ec.write_json(out / "summary.json", {
        "experiment": "recall of first exposures vs actual time on test (exploratory; NOT a test of H1)",
        "caveat": "settings are swept and read on the same test data (in-sample); describes the trade-off, not a frozen policy",
        "rule": {"read_times": READ_TIMES, "interpolation": "linear in actual time over curve points sorted by time; equal times -> max recall",
                 "min_time": "smallest actual time of a realised point with recall_first >= 0.90", "anchor": "(0, 0)",
                 "budget_grid": "0.01 ... 1.00 step 0.01", "tau_grid": "every distinct score of the seed on test",
                 "ties": "20 permutations of 8B (random: 200)"},
        "first_exposures": int(es.def_first.sum()), "kp_alpha": alpha,
        "readouts": summ.to_dict(orient="records"),
    })
    ec.write_run_info(out, t0, started, ec.standard_inputs(ds) + [KP_FILE, ec.MAIN / "policies_test.csv"] + inputs, ec.SEEDS)
    for _, r in summ.iterrows():
        ec.log(f"{r['method']:11s} recall@20/30/40/50 " + " / ".join(f"{r[f'recall_at_{int(x * 100)}']:.3f}" for x in READ_TIMES)
               + f"; min time for 0.90: {r['min_time_recall_0.90']:.3f} ± {r['min_time_recall_0.90_std']:.3f}")


if __name__ == "__main__":
    main()
