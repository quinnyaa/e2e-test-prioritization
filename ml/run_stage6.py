"""Stage 6: reproduce baselines, tuning, frozen models, val predictions, policies and tables.

Run from the repository root:
    ml/.venv/bin/python ml/run_stage6.py                       # final dataset -> ml/models, ml/results/stage6
    ml/.venv/bin/python ml/run_stage6.py --dataset data/dataset/history --out <dir> --models-dir <dir>  # smoke run

Protocol (notes/decisions.md, p. 7): the hold-out test part is never loaded (dataset_io lock);
every choice (combined-heuristic variant, hyper-parameters, model family, policy, tau, ranking by
p or p/duration) is made on the hold-out val part only; models are fitted on the train part only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score

import figures
from baselines import BASELINES, COMBINED_VARIANTS, LABELS_UK, ORACLE
from dataset_io import ROOT, SEED, dataset_fingerprint, load_dataset
from metrics import BUDGETS, N_PERM, N_PERM_RANDOM, TOPK, evaluate, pct_key, summarize, tie_keys
from policy import fixed_settings, policy_summary
from train import FAMILIES, FAMILY_LABELS, SEEDS, FitContext, fit_final, library_versions, predict, tune

PER_COMMIT_COLUMNS = (
    ["commit_index", "n_fail", "n_new", "n_repeat", "n_defects", "n_first", "n_perm", "apfd", "apfdc", "apfd_new", "apfd_repeat",
     "apfd_defect", "ttd_share", "first_pos"]
    + [f"rt{pct_key(b)}_hit" for b in BUDGETS]
    + [f"top{pct_key(k)}_{s}" for k in TOPK for s in ("hit", "hit_new", "caught", "def_caught", "first_caught", "time")]
    + [f"bud{pct_key(b)}_{s}" for b in BUDGETS for s in ("hit", "hit_new", "caught", "def_caught", "first_caught", "time", "pred_time", "tests")]
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/dataset/history-600-540")
    ap.add_argument("--out", default="ml/results/stage6")
    ap.add_argument("--models-dir", default="ml/models")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    ds_path = (ROOT / args.dataset).resolve()
    out = (ROOT / args.out).resolve()
    models_dir = (ROOT / args.models_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(ds_path)  # allow_test=False: the test part is not even in memory
    ds.assert_commit_level_constant(ds.train_range)
    train_r, val_r = ds.train_range, ds.val_range
    log(f"dataset {ds.path.name}: train {train_r}, val {val_r}, test {ds.test_range} locked")

    ctx_main = FitContext(ds, train_r, val_r, "main")
    es, val_rows = ctx_main.es, ctx_main.val_rows
    keys = tie_keys(es, N_PERM_RANDOM)
    ctx_main.keys = keys[:N_PERM]

    metric_rows: list[dict] = []
    per_commit: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    score_store: dict[tuple[str, int], np.ndarray] = {}

    def record(method: str, kind: str, seed: int | None, scores: np.ndarray, probabilistic: bool) -> dict:
        n_perm = N_PERM_RANDOM if method == "random" else N_PERM
        per = evaluate(es, scores, n_perm=n_perm, keys=keys)
        summ = summarize(per)
        if probabilistic:
            from metrics import classification_diagnostics
            summ.update(classification_diagnostics(val_rows["label"].to_numpy(), scores))
        row = {"method": method, "kind": kind, "seed": seed if seed is not None else -1, **summ}
        metric_rows.append(row)
        pc = per[PER_COMMIT_COLUMNS].copy()
        pc.insert(0, "method", method)
        pc.insert(1, "seed", seed if seed is not None else -1)
        per_commit.append(pc)
        predictions.append(pd.DataFrame({
            "commit_index": val_rows["commit_index"].to_numpy(), "test_id": val_rows["test_id"].to_numpy(),
            "model": method, "seed": seed if seed is not None else -1, "score": scores,
        }))
        score_store[(method, seed if seed is not None else -1)] = scores
        return row

    # ---------- 1. baselines ----------
    log("baselines")
    for name, fn in BASELINES.items():
        record(name, "baseline", None, fn(val_rows), False)
    variant_rows = []
    for name, fn in COMBINED_VARIANTS.items():
        s = summarize(evaluate(es, fn(val_rows), n_perm=N_PERM, keys=keys))
        variant_rows.append({"variant": name, "description": LABELS_UK[name], **s})
    variants = pd.DataFrame(variant_rows)
    chosen_variant = str(variants.loc[variants["apfd"].idxmax(), "variant"])
    variants["selected"] = variants["variant"] == chosen_variant
    record("combined", "baseline", None, COMBINED_VARIANTS[chosen_variant](val_rows), False)
    record("oracle", "oracle", None, ORACLE(val_rows), False)
    log(f"combined heuristic: {chosen_variant}")

    # ---------- 2. models ----------
    tuning_tables, cards = [], {}
    ctx_ctrl = FitContext(ds, train_r, val_r, "no_commit_chg")
    ctx_ctrl.keys = keys[:N_PERM]
    for fam in FAMILIES:
        log(f"tuning {fam}")
        table, best = tune(ds, fam, "main", train_r, val_r, seed=SEED, ctx=ctx_main)
        tuning_tables.append(table)
        tuned_apfd = float(table.loc[table["selected"], "apfd"].iloc[0])

        # Seed behaviour: identical predictions for two seeds -> seed-invariant, else 10 seeds.
        pipes = {SEEDS[0]: fit_final(ctx_main, fam, best, SEEDS[0]), SEEDS[1]: fit_final(ctx_main, fam, best, SEEDS[1])}
        p0, p1 = predict(pipes[SEEDS[0]], val_rows), predict(pipes[SEEDS[1]], val_rows)
        seed_invariant = bool(np.array_equal(p0, p1))
        seeds = [SEEDS[0]] if seed_invariant else list(SEEDS)
        for s in seeds:
            if s not in pipes:
                pipes[s] = fit_final(ctx_main, fam, best, s)
        # Reproducibility of a fixed seed (refit must give bitwise-identical predictions).
        refit_same = bool(np.array_equal(p0, predict(fit_final(ctx_main, fam, best, SEEDS[0]), val_rows)))
        if not refit_same:
            raise SystemExit(f"{fam}: refitting seed {SEEDS[0]} does not reproduce the predictions bitwise")
        for s in seeds:
            record(fam, "model", s, predict(pipes[s], val_rows), True)
        final_apfd = float(next(r["apfd"] for r in metric_rows if r["method"] == fam and r["seed"] == SEEDS[0]))

        # Cost-aware ranking p / predicted duration (all seeds).
        for s in seeds:
            sc = score_store[(fam, s)] / val_rows["meta_duration_mean"].to_numpy(dtype=float)
            record(f"{fam}/cost", "cost", s, sc, False)

        # Control: same configuration on no_commit_chg (commit-level chg_ removed).
        for s in seeds:
            pc = fit_final(ctx_ctrl, fam, best, s)
            record(f"{fam}@no_commit_chg", "control", s, predict(pc, val_rows), True)

        # Save frozen models (fitted on the train part only).
        fam_dir = models_dir / fam
        fam_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipes[SEEDS[0]], fam_dir / "model.joblib", compress=3)
        seed_files = {str(SEEDS[0]): "model.joblib"}
        if not seed_invariant:
            size0 = (fam_dir / "model.joblib").stat().st_size
            if size0 < 10_000_000:  # small enough to keep every seed (RF ~4.7 MB, LightGBM ~0.1 MB)
                (fam_dir / "seeds").mkdir(exist_ok=True)
                for s in seeds[1:]:
                    joblib.dump(pipes[s], fam_dir / "seeds" / f"seed-{s}.joblib", compress=3)
                    seed_files[str(s)] = f"seeds/seed-{s}.joblib"
        cards[fam] = {
            "model": fam,
            "family": FAMILY_LABELS[fam],
            "feature_set": "main",
            "features": ctx_main.features,
            "dropped_constant_features": ctx_main.dropped,
            "categorical": [c for c in ctx_main.features if c == "meta_module"],
            "input": "raw dataset.csv columns (pandas DataFrame); the pipeline selects and transforms the features",
            "output": "predict_proba(X)[:, 1] = uncalibrated failure score; ranking = descending score, ties broken randomly",
            "train_range": [train_r[0], train_r[1]],
            "val_range": [val_r[0], val_r[1]],
            "test_range_locked": [ds.test_range[0], ds.test_range[1]],
            "selection_criterion": "mean APFD over val commits with >= 1 failed test (fault = failed test), ties: 20 random permutations",
            "params": best,
            "grid_size": int(len(table)),
            "tuning_seed": SEED,
            "seeds": seeds,
            "seed_invariant": seed_invariant,
            "refit_same_seed_bitwise_identical": refit_same,
            "seed_files": seed_files,
            "seeds_not_saved": [s for s in seeds if str(s) not in seed_files],
            "val_apfd_tuning": tuned_apfd,
            "val_apfd_final_seed0": final_apfd,
            "scale_pos_weight_fit": ctx_main.spw,
            **dataset_fingerprint(ds),
            "versions": library_versions(),
            "created_by": "ml/run_stage6.py (stage 6)",
        }
        log(f"{fam}: seeds {len(seeds)} (invariant={seed_invariant}), refit identical={refit_same}, "
            f"APFD tuning {tuned_apfd:.4f} / final {final_apfd:.4f}")

    metrics_df = pd.DataFrame(metric_rows)

    # ---------- 3. choose the main model family on val ----------
    fam_mean = metrics_df[metrics_df["kind"] == "model"].groupby("method")["apfd"].mean()
    main_family = str(fam_mean.idxmax())
    log(f"main family by mean val APFD over seeds: {main_family} ({fam_mean.max():.4f})")

    # ---------- 4. policies ----------
    log("policies")
    pol_rows, fixed_rows = [], []
    methods = [("recent", -1, False), ("covers", -1, False), ("combined", -1, False), ("oracle", -1, False)]
    for fam in FAMILIES:
        for s in cards[fam]["seeds"]:
            methods += [(fam, s, True), (f"{fam}/cost", s, False), (f"{fam}@no_commit_chg", s, True)]
    for method, s, prob in methods:
        sc = score_store[(method, s)]
        ps = policy_summary(es, sc, keys[:N_PERM], method, prob)
        ps.insert(1, "seed", s)
        pol_rows.append(ps)
        fx = fixed_settings(es, sc, keys[:N_PERM], method)
        fx.insert(1, "seed", s)
        fixed_rows.append(fx)
    policies = pd.concat(pol_rows, ignore_index=True)
    fixed = pd.concat(fixed_rows, ignore_index=True)

    # ---------- 5. calibration of the main model (interpretation only) ----------
    log("calibration")
    calib = calibration_report(ds, ctx_main, main_family, cards[main_family]["params"], models_dir)

    # ---------- 6. reference single-feature AUCs on val ----------
    y_val = val_rows["label"].to_numpy()
    single_auc = {
        "hist_since_fail (reversed)": roc_auc_score(y_val, -val_rows["hist_since_fail"]),
        "hist_fail_last_1": roc_auc_score(y_val, val_rows["hist_fail_last_1"]),
        "hist_fail_rate": roc_auc_score(y_val, val_rows["hist_fail_rate"]),
        "chg_cov_files": roc_auc_score(y_val, val_rows["chg_cov_files"]),
    }

    # ---------- 7. policies of every family for the model cards; chosen policy of the main family ----------
    # Candidates for the main family: three policies on p, top-k/budget on p / predicted duration;
    # the cheapest mean actual time at the 0.90 target wins (in-sample on val).
    pol_c = policies[policies["criterion"] == "commit"]  # stage 6 choice: commit-level criterion
    cand = pol_c[pol_c["method"].isin([main_family, f"{main_family}/cost"]) & (pol_c["target"] == 0.90)]
    cand_time = cand.groupby(["method", "policy"])["time"].mean()
    best_method, chosen_policy = cand_time.idxmin()
    chosen_ranking = "p" if best_method == main_family else "p / meta_duration_mean"
    def per_seed_settings(fam: str, criterion: str) -> dict:
        fp = policies[(policies["method"] == fam) & (policies["criterion"] == criterion)]
        out = {}
        for pol_name in ("topk", "budget", "threshold"):
            out[pol_name] = {
                str(s): {f"{t:.2f}": float(fp[(fp["seed"] == s) & (fp["policy"] == pol_name) & (fp["target"] == t)]["setting"].iloc[0])
                         for t in (0.90, 0.95)}
                for s in cards[fam]["seeds"]
            }
        return out

    def budget_time(method: str, criterion: str) -> float:
        b = policies[(policies["method"] == method) & (policies["criterion"] == criterion) & (policies["policy"] == "budget") & (policies["target"] == 0.90)]
        return float(b["time"].mean())

    for fam in FAMILIES:
        settings = per_seed_settings(fam, "commit")
        by_crit = {c: per_seed_settings(fam, c) for c in ("defect", "first")}
        cards[fam]["policy"] = {
            "chosen_on_val": chosen_policy if fam == main_family else None,
            "ranking": chosen_ranking if fam == main_family else None,
            "budget_time_at_0.90_target": {"p": budget_time(fam, "commit"), "p / meta_duration_mean": budget_time(f"{fam}/cost", "commit")},
            "target_commit_recall": 0.90,
            "settings_per_seed": {
                "topk_share_of_tests": settings["topk"],
                "budget_share_of_predicted_time": settings["budget"],
                "threshold_tau": settings["threshold"],
            },
            "note": ("each setting is the cheapest (lowest mean actual time) reaching the target commit-level recall on val; "
                     "tau = largest threshold with recall >= target; all chosen and measured on val, i.e. in-sample until stage 8"),
            "by_recall_criterion": {
                "note": ("same procedure with another recall criterion: defect = (commit, defect) pairs, "
                         "first = first-exposure pairs (defect introduction commits); notes/decisions.md, p. 7"),
                "budget_time_at_0.90_target_p": {c: budget_time(fam, c) for c in ("defect", "first")},
                **{c: {"topk_share_of_tests": v["topk"], "budget_share_of_predicted_time": v["budget"], "threshold_tau": v["threshold"]}
                   for c, v in by_crit.items()},
            },
        }
    cards[main_family]["main_model"] = True
    cards[main_family]["calibration"] = calib

    # ---------- 8. write results ----------
    log("writing results")
    tuning = pd.concat(tuning_tables, ignore_index=True)
    # Fit times vary between runs; they go to a separate file so that tuning.csv is byte-reproducible.
    tuning.drop(columns=["fit_seconds"]).to_csv(out / "tuning.csv", index=False, float_format="%.6f")
    tuning[["family", "feature_set", "config", "params_json", "fit_seconds"]].to_csv(out / "timing.csv", index=False, float_format="%.3f")
    variants.to_csv(out / "combined_variants.csv", index=False, float_format="%.6f")
    pd.concat(predictions, ignore_index=True).to_csv(out / "val_predictions.csv", index=False)
    metrics_df.to_csv(out / "val_metrics.csv", index=False, float_format="%.6f")
    pd.concat(per_commit, ignore_index=True).to_csv(out / "per_commit_val.csv", index=False, float_format="%.6f")
    policies.to_csv(out / "policies.csv", index=False, float_format="%.6f")
    fixed.to_csv(out / "policy_fixed.csv", index=False, float_format="%.6f")
    for fam, card in cards.items():
        (models_dir / fam / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    import tables
    tables.write(out, ds, metrics_df, tuning, variants, chosen_variant, policies, fixed, cards, main_family, calib, single_auc)
    if not args.no_figures:
        figures.detection_curve(out / "detection_curve_val.png", es, keys, score_store, main_family)
        figures.policy_tradeoff(out / "policy_tradeoff_val.png", es, keys[:N_PERM], score_store, main_family)

    # ---------- 9. fresh-process check of the saved models ----------
    log("fresh-process model check")
    res = subprocess.run([sys.executable, str(ROOT / "ml" / "check_models.py"), "--models-dir", str(models_dir),
                          "--dataset", str(ds_path), "--predictions", str(out / "val_predictions.csv")],
                         cwd=ROOT, capture_output=True, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit("saved models do not reproduce the val predictions")

    elapsed = time.perf_counter() - t_start
    run_info = {
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "seed": SEED, "seeds": list(SEEDS), "n_perm": N_PERM, "n_perm_random": N_PERM_RANDOM,
        "main_family": main_family, "combined_variant": chosen_variant, "chosen_policy": chosen_policy, "chosen_ranking": chosen_ranking,
        **dataset_fingerprint(ds), "versions": library_versions(),
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    log(f"done in {elapsed:.0f} s")


def calibration_report(ds, ctx: FitContext, family: str, params: dict, models_dir: Path) -> dict:
    """Sigmoid vs isotonic calibration of the seed-0 main model on val (interpretation only).

    In-sample Brier (calibrated and scored on the same val) is optimistic; the cross-fitted value
    calibrates on one half of val (by time) and scores the other half, then swaps.
    """
    pipe = fit_final(ctx, family, params, SEEDS[0])
    X, y = ctx.val_rows, ctx.val_rows["label"].to_numpy()
    raw = predict(pipe, X)
    out = {"brier_raw": brier_score_loss(y, raw)}
    a, b = ds.val_range
    mid = (a + b) // 2
    first = (X["commit_index"] < mid).to_numpy()
    best_method, best_cf = None, np.inf
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(FrozenEstimator(pipe), method=method).fit(X, y)
        out[f"brier_{method}_in_sample"] = brier_score_loss(y, cal.predict_proba(X)[:, 1])
        cf = np.empty(len(y))
        for fit_mask in (first, ~first):
            c = CalibratedClassifierCV(FrozenEstimator(pipe), method=method).fit(X[fit_mask], y[fit_mask])
            cf[~fit_mask] = c.predict_proba(X[~fit_mask])[:, 1]
        out[f"brier_{method}_cross_fitted"] = brier_score_loss(y, cf)
        if out[f"brier_{method}_cross_fitted"] < best_cf:
            best_method, best_cf = method, out[f"brier_{method}_cross_fitted"]
    out["chosen_method"] = best_method
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method=best_method).fit(X, y)
    joblib.dump(cal, models_dir / family / "calibrated.joblib", compress=3)
    out["file"] = "calibrated.joblib"
    out["note"] = ("calibrator fitted on the whole val part; use only to interpret probabilities (CLI). "
                   "Ranking and tau use the uncalibrated score; isotonic calibration is not strictly monotone "
                   "and would create ties.")
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


if __name__ == "__main__":
    main()
