"""Stage 8: main evaluation of the frozen models and baselines on one range (notes/stage8-protocol.md, p. 2-4).

    ml/.venv/bin/python ml/run_stage8.py --range val                               # 8A dry run -> ml/results/stage8-dryrun/
    ml/.venv/bin/python ml/run_stage8.py --range test --allow-test --h1-approved   # 8B, once -> ml/results/stage8/main/

Nothing is fitted or tuned here. Scores come from the frozen models (`ml/models/<lr|rf|hgb|lgbm>/`, trained on
commits 0-323; RF and LightGBM with 10 seeds) and from the fixed baselines; policy settings come from the
model cards (tau, B, top-k per seed; full precision) and from `ml/results/stage6/policies.csv` (heuristics).
The only exception is the Kim-Porter baseline (ml/baseline_kp.py): its alpha and B are chosen on val in
`--range val` and written to kp.json; `--range test` reads them from `ml/results/stage8-dryrun/kp.json`.

The run is refused when (a) a frozen file differs from ml/stage8_frozen.json (ml/check_frozen.py);
(b) `--range test` is given without both `--allow-test` and `--h1-approved` (checked before any data is read);
(c) a val run would write into ml/results/stage8/ (its existence unlocks ml/train_production.py);
(d) a test run would overwrite a finished test run (run_info.json, written last, exists): the test part is opened once;
(e) `--range test --kp-settings` names any file other than the frozen ml/results/stage8-dryrun/kp.json.
Data checks that could only fail on unseen data (hist_fail_last_1 = previous label) are fatal on val and
recorded in run_info.json on test, so that a run on the opened test part always writes its results.

Outputs (<r> = val | test): <r>_metrics.csv, per_commit_<r>.csv, policies_<r>.csv, per_commit_policies_<r>.csv,
diagnostics_<r>.csv, h1.json, h2.json, exploratory.csv, kp.json, detection_curve_<r>.png,
policy_tradeoff_<r>.png, run_info.json. On val, H1/H2 numbers only check the pipeline (in-sample).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRYRUN_DIR = ROOT / "ml" / "results" / "stage8-dryrun"
STAGE8_DIR = ROOT / "ml" / "results" / "stage8"
TEST_OUT = STAGE8_DIR / "main"
STAGE6_DIR = ROOT / "ml" / "results" / "stage6"
MODELS_DIR = ROOT / "ml" / "models"
KP_FROZEN = DRYRUN_DIR / "kp.json"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--range", choices=("val", "test"), required=True, dest="range_name")
    ap.add_argument("--out", default=None, help="output directory (default: stage8-dryrun for val, stage8/main for test)")
    ap.add_argument("--allow-test", action="store_true", help="open the hold-out test part (8B only)")
    ap.add_argument("--h1-approved", action="store_true", help="the supervisor approved the main H1 definition (gate of 8B)")
    ap.add_argument("--kp-settings", default=None,
                    help="read the Kim-Porter alpha / B from this kp.json instead of choosing them on val "
                         "(always the case for --range test; for --range val it is a self-check of the test path)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    if args.range_name == "test" and not (args.allow_test and args.h1_approved):
        ap.error("--range test needs both --allow-test and --h1-approved (notes/stage8-protocol.md: gate of 8B)")
    if args.range_name == "val" and args.allow_test:
        ap.error("--allow-test is only meaningful with --range test")
    if args.range_name == "test" and args.kp_settings and Path(args.kp_settings).resolve() != KP_FROZEN:
        ap.error(f"--range test reads the Kim-Porter settings only from the frozen {KP_FROZEN.relative_to(ROOT)}")
    return args


# Heavy imports after the argument check: a refused test run never imports the data code.
def _imports():
    global np, pd, joblib, figures, BASELINES, COMBINED_VARIANTS, ORACLE, load_dataset, dataset_fingerprint, sha256
    global N_PERM, N_PERM_RANDOM, build_eval_set, classification_diagnostics, evaluate, summarize, tie_keys
    global policy_summary, _orders, _mean_selection, _aggregate, positions, _cum_by_test, selection_columns
    global FAMILIES, SEEDS, library_versions, predict, PER_COMMIT_COLUMNS, kp, st, check_frozen
    import joblib as _joblib
    import numpy as _np
    import pandas as _pd

    np, pd, joblib = _np, _pd, _joblib
    import baseline_kp as _kp
    import check_frozen as _cf
    import figures as _figures
    import stats as _st
    from baselines import BASELINES as _B, COMBINED_VARIANTS as _CV, ORACLE as _O
    from dataset_io import dataset_fingerprint as _dfp, load_dataset as _ld, sha256 as _sha
    from metrics import (N_PERM as _NP, N_PERM_RANDOM as _NPR, _cum_by_test as _cbt, build_eval_set as _bes,
                         classification_diagnostics as _cd, evaluate as _ev, positions as _pos, selection_columns as _sc,
                         summarize as _su, tie_keys as _tk)
    from policy import _aggregate as _agg, _mean_selection as _ms, _orders as _ord, policy_summary as _ps
    from run_stage6 import PER_COMMIT_COLUMNS as _PCC
    from train import FAMILIES as _F, SEEDS as _S, library_versions as _lv, predict as _pr

    figures, kp, st, check_frozen = _figures, _kp, _st, _cf
    BASELINES, COMBINED_VARIANTS, ORACLE = _B, _CV, _O
    load_dataset, dataset_fingerprint, sha256 = _ld, _dfp, _sha
    N_PERM, N_PERM_RANDOM, build_eval_set, classification_diagnostics = _NP, _NPR, _bes, _cd
    evaluate, summarize, tie_keys, positions, _cum_by_test, selection_columns = _ev, _su, _tk, _pos, _cbt, _sc
    policy_summary, _orders, _mean_selection, _aggregate = _ps, _ord, _ms, _agg
    FAMILIES, SEEDS, library_versions, predict, PER_COMMIT_COLUMNS = _F, _S, _lv, _pr, _PCC


COMBINED_VARIANT = "comb_last_cov_mod_recent"  # chosen on val in stage 6 (run_info.json); never re-selected
HEURISTIC_POLICY_METHODS = ("recent", "covers", "combined", "oracle")  # methods with policies in stage 6 policies.csv
H1_TARGET, H1_RECALL, H1_TIME = "0.90", 0.90, 0.30
H1_COMPARE = (("lgbm", "threshold"), ("lgbm", "budget"), ("hgb", "threshold"), ("hgb", "budget"), ("lr", "threshold"),
              ("lr", "budget"), ("rf", "budget"), ("combined", "budget"), ("covers", "budget"), ("kp", "budget"),
              ("oracle", "budget"))
EXPLORATORY_MODELS = ("rf", "lgbm", "hgb", "lr")
EXPLORATORY_REFS = ("combined", "covers", "recent", "fail_rate", "kp", "random")
PRIMARY_PAIR = ("rf", "combined")
CRITERIA = ("commit", "defect", "first")
TARGETS = ("0.90", "0.95")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def clean(x):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@contextmanager
def range_titles(range_name: str, rng: tuple[int, int]):
    """figures.py (frozen with stage 6) hard-codes the val range in two titles; for another range they are
    rewritten at draw time without touching figures.py (protocol, section 9)."""
    if range_name == "val":
        yield
        return
    from matplotlib.axes import Axes

    original = Axes.set_title
    label = f"{range_name.capitalize()} (коміти {rng[0]}–{rng[1] - 1})"

    def patched(self, text, *a, **k):
        return original(self, str(text).replace("Val (коміти 324–431)", label).replace("Val:", f"{range_name.capitalize()}:"), *a, **k)

    Axes.set_title = patched
    try:
        yield
    finally:
        Axes.set_title = original


# ---------------------------------------------------------------------- frozen policy settings


def card_settings(card: dict, seed: int) -> list[dict]:
    """Frozen policy settings of one model seed (full precision from model_card.json)."""
    pol = card["policy"]
    out = []
    for criterion in CRITERIA:
        table = pol["settings_per_seed"] if criterion == "commit" else pol["by_recall_criterion"][criterion]
        for target in TARGETS:
            for name, key in (("topk", "topk_share_of_tests"), ("budget", "budget_share_of_predicted_time"), ("threshold", "threshold_tau")):
                v = table[key].get(str(seed), {}).get(target)
                if v is not None:
                    out.append({"criterion": criterion, "target": target, "policy": name, "setting": float(v)})
    return out


def csv_settings(policies: "pd.DataFrame", method: str, seed: int) -> list[dict]:
    """Frozen settings of a heuristic / cost-ranked variant from a policies table (stage 6 policies.csv or kp.json).
    Budgets are 2-decimal grid points and top-k is a test count, so %.6f storage is exact."""
    p = policies[(policies["method"] == method) & (policies["seed"] == seed)]
    out = []
    for r in p.itertuples(index=False):
        if r.policy == "threshold":
            continue  # tau of a probabilistic model is always read from its card at full precision
        out.append({"criterion": r.criterion, "target": f"{float(r.target):.2f}", "policy": r.policy,
                    "setting": float(r.setting), "n_tests": None if r.policy != "topk" else int(round(r.n_tests))})
    return out


def apply_policy(es, scores, keys20, policy: str, setting: float, n_tests: int | None = None) -> tuple[dict, dict]:
    """(aggregate as in policy.py, per-commit columns averaged over tie permutations) of one frozen setting.
    Uses the same private helpers as policy.topk_curve / budget_curve / threshold_stats, so the numbers are
    bitwise those of the stage 6 curves at that setting."""
    if policy == "threshold":
        sels = [es.matrix(scores) >= setting]
    else:
        orders = _orders(es, scores, keys20)
        if policy == "topk":
            k = n_tests if n_tests is not None else int(round(setting * es.n))
            sels = [positions(o) <= k for o in orders]
        elif policy == "budget":
            pred_total = es.pred_dur.sum(1, keepdims=True)
            sels = [_cum_by_test(o, es.pred_dur) <= setting * pred_total + 1e-9 for o in orders]
        else:
            raise ValueError(policy)
    cols = _mean_selection(es, sels)
    return _aggregate(es, cols), cols


# ---------------------------------------------------------------------- main


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    r = args.range_name
    out = Path(args.out).resolve() if args.out else (DRYRUN_DIR if r == "val" else TEST_OUT)
    if r == "val" and (out == STAGE8_DIR or STAGE8_DIR in out.parents):
        raise SystemExit(f"refused: a val run must not write into {STAGE8_DIR.relative_to(ROOT)}/ (it unlocks train_production.py)")
    if r == "test" and (out / "run_info.json").exists():
        raise SystemExit(f"refused: {out / 'run_info.json'} exists; the test part is evaluated once (notes/stage8-protocol.md)")

    _imports()
    log("frozen files (ml/check_frozen.py)")
    problems = check_frozen.verify(verbose=True)
    if problems:
        for p in problems:
            print(f"  [FAIL] {p}")
        raise SystemExit("refused: frozen files differ from ml/stage8_frozen.json")
    frozen = check_frozen.load_frozen()

    ds = load_dataset(allow_test=(r == "test"))
    rng = ds.val_range if r == "val" else ds.test_range
    log(f"range {r} {rng}; allow_test={ds.allow_test}; commits in memory up to {int(ds.rows['commit_index'].max())}")
    rows = ds.slice(rng)
    history_rows = ds.slice((0, rng[1]))  # Kim-Porter needs every earlier commit
    es = build_eval_set(rows, ds.positives_in(rng), ds.introduced_at)
    keys = tie_keys(es, N_PERM_RANDOM)
    keys20 = keys[:N_PERM]
    y = rows["label"].to_numpy()

    # Consistency of the frozen setup.
    s6_info = json.loads((STAGE6_DIR / "run_info.json").read_text(encoding="utf-8"))
    if s6_info["combined_variant"] != COMBINED_VARIANT:
        raise SystemExit("combined heuristic variant differs from stage 6")
    commits_h, y_h = kp.label_matrix(history_rows)
    h1 = history_rows["hist_fail_last_1"].to_numpy().reshape(y_h.shape)
    # Kim-Porter f = previous label (protocol, p. 7.1); it equals hist_fail_last_1 when every commit builds.
    kp_mismatch = int((h1[1:] != y_h[:-1]).any(1).sum() + bool(h1[0].any()))
    if kp_mismatch and r == "val":
        raise SystemExit("hist_fail_last_1 is not the previous commit's label: Kim-Porter f would differ from the features")
    if kp_mismatch:
        log(f"WARNING: {kp_mismatch} commit(s) where hist_fail_last_1 != previous label (recorded in run_info.json)")

    metric_rows: list[dict] = []
    per_commit: list = []
    per_store: dict = {}
    score_store: dict = {}
    diag_rows: list[dict] = []

    def record(method: str, kind: str, seed: int, scores, probabilistic: bool) -> None:
        n_perm = N_PERM_RANDOM if method == "random" else N_PERM
        per = evaluate(es, scores, n_perm=n_perm, keys=keys)
        summ = summarize(per)
        if probabilistic:
            d = classification_diagnostics(y, scores)
            summ.update(d)
            diag_rows.append({"method": method, "seed": seed, "calibrated": False, **d})
        metric_rows.append({"method": method, "kind": kind, "seed": seed, **summ})
        pc = per[PER_COMMIT_COLUMNS].copy()
        pc.insert(0, "method", method)
        pc.insert(1, "seed", seed)
        per_commit.append(pc)
        per_store[(method, seed)] = per
        score_store[(method, seed)] = np.asarray(scores, dtype=float)

    # ---------- baselines ----------
    log("baselines")
    for name, fn in BASELINES.items():
        record(name, "baseline", -1, fn(rows), False)
    record("combined", "baseline", -1, COMBINED_VARIANTS[COMBINED_VARIANT](rows), False)
    record("oracle", "oracle", -1, ORACLE(rows), False)

    # ---------- Kim-Porter: alpha and B chosen on val, read from kp.json otherwise ----------
    kp_source = args.kp_settings or (str(KP_FROZEN) if r == "test" else None)
    if kp_source is None:
        log("Kim-Porter: choosing alpha and B on val")
        alpha_rows = []
        for a in kp.ALPHAS:
            s = summarize(evaluate(es, kp.kp_scores(history_rows, rng, a), n_perm=N_PERM, keys=keys))
            alpha_rows.append({"alpha": a, "apfd": s["apfd"], "apfdc": s["apfdc"], "apfd_new": s["apfd_new"]})
        best = max(range(len(alpha_rows)), key=lambda i: (alpha_rows[i]["apfd"], -i))  # exact tie -> earlier grid entry
        alpha = kp.ALPHAS[best]
        kp_pol = policy_summary(es, kp.kp_scores(history_rows, rng, alpha), keys20, "kp", False)
        kp_pol.insert(1, "seed", -1)
        kp_doc = {
            "method": "kp", "description": "Kim-Porter: h_c = alpha f_{c-1} + (1 - alpha) h_{c-1}, h_0 = 0 (ml/baseline_kp.py)",
            "chosen_on": f"val {list(rng)}", "selection": "mean APFD over red val commits, 20 tie permutations; exact tie -> earlier alpha",
            "alpha_grid": alpha_rows, "alpha": alpha,
            "policies": kp_pol.to_dict(orient="records"),
            "note": "policies = cheapest setting reaching the target on val (policy.policy_summary), in-sample on val; "
                    "frozen for the test part (notes/stage8-protocol.md, p. 7.1)",
        }
    else:
        kp_doc = json.loads(Path(kp_source).read_text(encoding="utf-8"))
        alpha = float(kp_doc["alpha"])
        log(f"Kim-Porter: settings read from {kp_source} (alpha = {alpha})")
    record("kp", "baseline", -1, kp.kp_scores(history_rows, rng, alpha), False)

    # ---------- frozen models ----------
    cards = {}
    for fam in FAMILIES:
        card = json.loads((MODELS_DIR / fam / "model_card.json").read_text(encoding="utf-8"))
        if card["train_range"] != [0, 324]:
            raise SystemExit(f"{fam}: train_range {card['train_range']} != [0, 324)")
        cards[fam] = card
        log(f"{fam}: {len(card['seeds'])} seed(s)")
        for s in card["seeds"]:
            pipe = joblib.load(MODELS_DIR / fam / card["seed_files"][str(s)])
            record(fam, "model", int(s), predict(pipe, rows), True)
        for s in card["seeds"]:
            record(f"{fam}/cost", "cost", int(s), score_store[(fam, int(s))] / rows["meta_duration_mean"].to_numpy(dtype=float), False)
    cal = joblib.load(MODELS_DIR / "rf" / "calibrated.joblib")
    d = classification_diagnostics(y, cal.predict_proba(rows)[:, 1])
    diag_rows.append({"method": "rf", "seed": int(cards["rf"]["seeds"][0]), "calibrated": True, **d})

    # ---------- frozen policies ----------
    log("frozen policies")
    s6_pol = pd.read_csv(STAGE6_DIR / "policies.csv")
    kp_pol_df = pd.DataFrame(kp_doc["policies"])
    plan = []  # (method, seed, probabilistic settings list)
    for m in HEURISTIC_POLICY_METHODS:
        plan.append((m, -1, csv_settings(s6_pol, m, -1)))
    plan.append(("kp", -1, csv_settings(kp_pol_df, "kp", -1)))
    for fam in FAMILIES:
        for s in cards[fam]["seeds"]:
            plan.append((fam, int(s), card_settings(cards[fam], int(s))))
            plan.append((f"{fam}/cost", int(s), csv_settings(s6_pol, f"{fam}/cost", int(s))))
    pol_rows, pol_commit = [], []
    pol_cols: dict = {}
    for method, s, settings in plan:
        if not settings:
            raise SystemExit(f"no frozen policy settings for {method} seed {s}")
        for st_ in settings:
            agg, cols = apply_policy(es, score_store[(method, s)], keys20, st_["policy"], st_["setting"], st_.get("n_tests"))
            n_tests = st_.get("n_tests")
            if st_["policy"] == "topk" and n_tests is None:
                n_tests = int(round(st_["setting"] * es.n))
            setting = n_tests / es.n if st_["policy"] == "topk" else st_["setting"]
            pol_rows.append({"method": method, "seed": s, "criterion": st_["criterion"], "target": float(st_["target"]),
                             "policy": st_["policy"], "setting": setting, "n_tests": n_tests if st_["policy"] == "topk" else np.nan, **agg})
            pol_cols[(method, s, st_["criterion"], st_["target"], st_["policy"])] = cols
            pol_commit.append(pd.DataFrame({
                "method": method, "seed": s, "criterion": st_["criterion"], "target": float(st_["target"]), "policy": st_["policy"],
                "commit_index": es.commits, "n_fail": es.y.sum(1), "n_defects": es.def_per_commit, "n_first": es.first_per_commit,
                "hit": cols["sel_hit"], "caught": cols["sel_caught"], "def_caught": cols["sel_def_caught"],
                "first_caught": cols["sel_first_caught"], "tests": cols["sel_tests"], "time": cols["sel_time"],
                "pred_time": cols["sel_pred_time"],
            }))
    policies = pd.DataFrame(pol_rows)

    # ---------- H1 (protocol, p. 2) ----------
    log("H1")
    counts = st.bootstrap_counts(len(es.commits))
    red = es.y.any(1)
    n_first, n_def = es.first_per_commit.astype(float), es.def_per_commit.astype(float)

    def seeds_of(method: str) -> list[int]:
        base = method.split("/")[0]
        return [int(s) for s in cards[base]["seeds"]] if base in cards else [-1]

    def policy_block(method: str, policy: str, criterion: str, target: str) -> dict | None:
        ss = seeds_of(method)
        keys_ = [(method, s, criterion, target, policy) for s in ss]
        if not all(k in pol_cols for k in keys_):
            return None
        C = [pol_cols[k] for k in keys_]
        fc = np.array([c["sel_first_caught"] for c in C])
        dc = np.array([c["sel_def_caught"] for c in C])
        hit = np.array([c["sel_hit"] for c in C])
        tm = np.array([c["sel_time"] for c in C])
        pt = np.array([c["sel_pred_time"] for c in C])
        per_seed = {
            "recall_first": fc.sum(1) / n_first.sum() if n_first.sum() else np.full(len(ss), np.nan),
            "recall_defect": dc.sum(1) / n_def.sum() if n_def.sum() else np.full(len(ss), np.nan),
            "recall_commit": hit[:, red].mean(1),
            "time": tm.mean(1), "time_red": tm[:, red].mean(1), "time_max": tm.max(1), "pred_time": pt.mean(1),
        }
        settings = [float(policies[(policies["method"] == method) & (policies["seed"] == s) & (policies["criterion"] == criterion)
                                   & (policies["target"] == float(target)) & (policies["policy"] == policy)]["setting"].iloc[0]) for s in ss]
        block = {"method": method, "policy": policy, "criterion": criterion, "target": float(target), "seeds": ss, "settings": settings}
        for k, v in per_seed.items():
            block[k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "per_seed": [float(x) for x in v]}
        block["bootstrap_95"] = {
            "recall_first": st.percentile_ci(st.boot_ratio(counts, fc, n_first)),
            "recall_defect": st.percentile_ci(st.boot_ratio(counts, dc, n_def)),
            "recall_commit": st.percentile_ci(st.boot_mean(counts, hit, mask=red)),
            "time": st.percentile_ci(st.boot_mean(counts, tm)),
        }
        return block

    main_block = policy_block("rf", "threshold", "first", H1_TARGET)
    met = bool(main_block["recall_first"]["mean"] >= H1_RECALL and main_block["time"]["mean"] <= H1_TIME)
    h1 = {
        "range": r, "commits": [int(rng[0]), int(rng[1])], "n_commits": int(len(es.commits)), "red_commits": int(red.sum()),
        "first_exposures": int(n_first.sum()), "defect_pairs": int(n_def.sum()),
        "in_sample": r == "val",
        "note": ("val dry run: tau and B were chosen on this same val range, so these numbers only check the pipeline; "
                 "they are not a result" if r == "val" else "test part, frozen settings (notes/stage8-protocol.md, p. 2)"),
        "rule": f"H1 met iff mean over seeds of recall_first >= {H1_RECALL} and mean actual time share (all commits) <= {H1_TIME}; "
                "RF + tau (first-exposure criterion, target 0.90); the bootstrap interval is descriptive",
        "main": main_block,
        "verdict": ("met" if met else "not met") if r == "test" else ("met (in-sample, pipeline check only)" if met else "not met (in-sample, pipeline check only)"),
        "additional": {
            "defect_pairs_rf_tau": {k: main_block[k] for k in ("recall_defect",)} | {"bootstrap_95": main_block["bootstrap_95"]["recall_defect"]},
            "red_commits_rf_budget_commit_criterion": policy_block("rf", "budget", "commit", H1_TARGET),
        },
        "comparison_target_0.90": [b for b in (policy_block(m, p, "first", H1_TARGET) for m, p in H1_COMPARE) if b is not None],
        "sensitivity_target_0.95": [b for b in (policy_block(m, p, "first", "0.95") for m, p in (("rf", "threshold"),) + H1_COMPARE) if b is not None],
        "bootstrap": {"n_boot": st.BOOT_N, "seed": st.BOOT_SEED, "unit": "commit", "interval": "percentile 95 %",
                      "recall": "ratio of resampled sums per seed, then mean over seeds"},
    }

    # ---------- H2 and exploratory comparisons (protocol, p. 3) ----------
    log("H2 and exploratory comparisons")

    def per_commit_mean(method: str, metric: str) -> np.ndarray:
        arr = np.array([per_store[(method, s)][metric].to_numpy() for s in seeds_of(method)])
        return arr.mean(0)[red]

    def compare(model: str, ref: str, metric: str) -> dict:
        a, b = per_commit_mean(model, metric), per_commit_mean(ref, metric)
        w = st.wilcoxon(a, b)
        dlt = st.cliffs_delta(a, b)
        return {"model": model, "reference": ref, "metric": metric, **w, "mean_model": float(a.mean()), "mean_reference": float(b.mean()),
                "delta": dlt, "magnitude": st.romano(dlt)}

    prim = compare(*PRIMARY_PAIR, "apfd")
    h2 = {
        "range": r, "in_sample": r == "val",
        "note": ("val dry run: the combined variant and every model were chosen on this same val range; pipeline check only"
                 if r == "val" else "test part (notes/stage8-protocol.md, p. 3)"),
        "rule": "H2 confirmed iff Wilcoxon p < 0.05 and median(RF - combined) > 0; APFD over red commits; RF = mean over 10 seeds",
        "test": "scipy.stats.wilcoxon(rf, comb, zero_method='wilcox', alternative='two-sided', method='auto')",
        **prim,
        "confirmed": bool(prim["p"] == prim["p"] and prim["p"] < 0.05 and prim["median_diff"] > 0),
    }
    exp_rows = []
    for family, metric, with_primary in (("A", "apfd", False), ("B", "apfdc", True)):
        fam_rows = [compare(m, ref, metric) for m in EXPLORATORY_MODELS for ref in EXPLORATORY_REFS
                    if with_primary or (m, ref) != PRIMARY_PAIR]
        ph = st.holm([x["p"] for x in fam_rows])
        for x, p_h in zip(fam_rows, ph):
            exp_rows.append({"family": family, **x, "p_holm": p_h, "reject_holm_0.05": bool(p_h < 0.05) if p_h == p_h else False})
    exploratory = pd.DataFrame(exp_rows)
    exploratory.insert(1, "in_sample", r == "val")

    # ---------- write ----------
    log("writing results")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out / f"{r}_metrics.csv", index=False, float_format="%.6f")
    pd.concat(per_commit, ignore_index=True).to_csv(out / f"per_commit_{r}.csv", index=False, float_format="%.6f")
    policies.to_csv(out / f"policies_{r}.csv", index=False, float_format="%.6f")
    pd.concat(pol_commit, ignore_index=True).to_csv(out / f"per_commit_policies_{r}.csv", index=False, float_format="%.6f")
    diag = pd.DataFrame(diag_rows)
    diag["in_sample_calibration"] = diag["calibrated"] & (r == "val")
    diag.to_csv(out / f"diagnostics_{r}.csv", index=False, float_format="%.6f")
    write_json(out / "h1.json", h1)
    write_json(out / "h2.json", h2)
    exploratory.to_csv(out / "exploratory.csv", index=False, float_format="%.6g")
    if kp_source is None:
        write_json(out / "kp.json", kp_doc)
    elif Path(kp_source).resolve() != (out / "kp.json").resolve():
        shutil.copyfile(kp_source, out / "kp.json")  # the settings used, byte for byte
    if not args.no_figures:
        with range_titles(r, rng):
            figures.detection_curve(out / f"detection_curve_{r}.png", es, keys, score_store, "rf")
            figures.policy_tradeoff(out / f"policy_tradeoff_{r}.png", es, keys20, score_store, "rf")

    import matplotlib
    import scipy

    protocol_text = (ROOT / "notes" / "stage8-protocol.md").read_text(encoding="utf-8")
    gate = next((ln for ln in protocol_text.splitlines() if "Гейт для кроку 8B" in ln), "")
    elapsed = time.perf_counter() - t_start
    run_info = {
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "command": "ml/run_stage8.py " + " ".join(sys.argv[1:] if argv is None else argv),
        "range": r, "commits": [int(rng[0]), int(rng[1])], "allow_test": bool(ds.allow_test), "h1_approved_flag": bool(args.h1_approved),
        "protocol_gate_line": gate.strip(),
        "seeds": list(SEEDS), "n_perm": N_PERM, "n_perm_random": N_PERM_RANDOM, "combined_variant": COMBINED_VARIANT,
        "kim_porter": {"alpha": alpha, "settings": "chosen on val in this run" if kp_source is None else "read from file",
                       "file_sha256": sha256(out / "kp.json"), "commits_hist_fail_last_1_ne_previous_label": kp_mismatch},
        "frozen_check": "all match", "frozen_sha256": frozen["frozen"], "added_in_8a_sha256": frozen.get("added_in_8a", {}),
        **dataset_fingerprint(ds),
        "versions": {**library_versions(), "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
    }
    write_json(out / "run_info.json", run_info)
    mb = main_block
    log(f"H1 ({r}): recall_first {mb['recall_first']['mean']:.3f} ± {mb['recall_first']['std']:.3f}, "
        f"time {100 * mb['time']['mean']:.1f} ± {100 * mb['time']['std']:.1f} % -> {h1['verdict']}")
    log(f"H2 ({r}): APFD rf {prim['mean_model']:.3f} vs combined {prim['mean_reference']:.3f}, p = {prim['p']:.4g}, "
        f"delta = {prim['delta']:.3f} ({prim['magnitude']}) -> {'confirmed' if h2['confirmed'] else 'not confirmed'}")
    log(f"done in {elapsed:.0f} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
