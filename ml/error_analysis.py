"""Stage 8D: error analysis of the missed first exposures of RF + tau on test (notes/stage8-protocol.md, p. 6).

    ml/.venv/bin/python ml/error_analysis.py  # -> ml/results/stage8/errors/, ≈ 15 s

Nothing is trained or tuned. Scores are predictions of the frozen RF (10 seeds) with the card tau (first exposure,
target 0.90); combined uses B = 0.34. The per-defect view comes from exp_common.range_first_exposures, whose
fatal control rebuilds 8B's per-commit first_caught (main/per_commit_policies_test.csv) bit for bit.

Rules, written into the protocol (section 9, entry "8D, аналіз помилок") before the categories were counted:
  missed      share of RF seeds whose tau selection catches the first exposure <= 0.5 (not caught in >= 5 of 10);
  (a)         no test failing because of the defect has chg_cov_files > 0 on the first-exposure commit;
  (b)         the commit diff only adds lines: chg_lines_deleted == 0 and chg_lines_added > 0;
  (c)         every failing test is expensive by the prediction available before the run:
              meta_duration_mean > EXPENSIVE_MS (the "over 500 ms" band of notes/tests.md, p. 5). "Does not fit"
              is trivially true for a tau miss (no budget); actual duration / timeouts are descriptive only;
  (d)         the defect's module has no train positive (commits 0-323) caused by a defect of the same module;
  (e) other   none of (a)-(d); the description is generated from the descriptive columns.
Descriptive (not categories): empty tau selection (share of seeds), rank / score of the failing tests (mean over
seeds), margin of the best failing score to tau, hist_ of the failing tests, durations. The same category flags
are computed for the caught first exposures as a contrast (not a new category).
Shared / unique misses with combined + B: the same <= 0.5 rule over the 20 tie permutations instead of seeds.
Descriptive contrast "same ranking, other policy": per-defect catch of RF + budget B (frozen per-seed B of the
first-exposure criterion, target 0.90, from the card); fatal control against 8B's per-commit first_caught.
Timeout flag of a failing test: actual duration >= EXPECT_TIMEOUT_MS (expect timeout of tests/playwright.config.ts);
descriptive only, it is not visible before the run.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import exp_common as ec
from exp_manual_bugs import MANUAL_INDEX, module_of
from metrics import positions
from train import SEEDS

MUTANTS = ec.ROOT / "mutation" / "mutants.json"
MANUAL_DIR = ec.ROOT / "mutation" / "manual"
RUNS = ec.ROOT / "data" / "history-600" / "runs.jsonl"
MISSED_MAX_CAUGHT = 0.5  # caught in <= 5 of 10 seeds
EXPENSIVE_MS = 500.0
EXPECT_TIMEOUT_MS = 3000.0
CATS = {"a": "діф не перетинає covers жодного тесту, що падає", "b": "діф лише додає рядки",
        "c": "усі тести, що падають, дорогі за прогнозом", "d": "у модуля немає історії падінь у fit-частині",
        "e": "інше"}
EXAMPLES = ("MAN-04", "MAN-20", "MUT-1062", "MUT-593", "MUT-270")  # hand-picked before the per-example details were printed
DIFF_COLS = ["chg_files", "chg_lines_added", "chg_lines_deleted", "chg_hunks", "chg_hunks_add", "chg_hunks_modify",
             "chg_hunks_delete", "chg_modules"]


def defect_meta(ds) -> tuple[dict, dict]:
    muts = {f"MUT-{m['id']}": m for m in json.loads(MUTANTS.read_text(encoding="utf-8"))["mutants"]}
    manual = {m["id"]: m for m in json.loads(MANUAL_INDEX.read_text(encoding="utf-8"))}
    return muts, manual


def module_and_op(d: str, muts: dict, manual: dict) -> tuple[str, str, str]:
    if d.startswith("MUT-"):
        return "mutant", muts[d]["mutator"], muts[d]["module"]
    return "manual", "manual", module_of(manual[d]["file"])


def fit_modules_with_failures(ds, muts: dict, manual: dict) -> dict[str, int]:
    """module -> number of train positives (0-323) caused by a defect of that module."""
    pos = ds.positives_in((0, ds.val_range[0]))
    counts: dict[str, int] = {}
    for causes in pos["causes"]:
        for d in str(causes).split(";"):
            m = module_and_op(d, muts, manual)[2]
            counts[m] = counts.get(m, 0) + 1
    return counts


def load_runs(commits: set[int]) -> dict[int, dict]:
    out = {}
    with RUNS.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in commits:
                out[i] = json.loads(line)
            if i > max(commits):
                break
    return out


def main() -> None:
    t0, started = ec.started_now()
    out = ec.out_dir("errors")
    ds = ec.load()
    muts, manual = defect_meta(ds)
    fe, es = ec.range_first_exposures(ds, ds.test_range, "test")  # fatal control against 8B inside
    rows = ds.slice(ds.test_range)
    keys = ec.tie_keys(es, ec.N_PERM)
    scores = {s: es.matrix(ec.predict(ec.frozen_rf(s), rows)) for s in SEEDS}
    taus = {s: ec.card_tau(s) for s in SEEDS}
    pos = {s: np.mean([positions(o) for o in ec._orders(es, scores[s].ravel(), keys)], axis=0) for s in SEEDS}
    k_first = np.flatnonzero(es.def_first)
    assert (es.def_ids[k_first] == fe["defect_id"].to_numpy()).all()
    ref = pd.read_csv(ec.MAIN / "per_commit_policies_test.csv")
    rf_b = {}
    for s in SEEDS:
        b = float(ec.RF_CARD["policy"]["by_recall_criterion"]["first"]["budget_share_of_predicted_time"][str(s)]["0.90"])
        caught = ec.per_defect_caught(es, ec.budget_sels(es, scores[s].ravel(), keys, b))[k_first]
        got = np.bincount(es.def_commit[k_first], weights=caught, minlength=len(es.commits))
        r_ = ref[(ref["method"] == "rf") & (ref["seed"] == s) & (ref["policy"] == "budget") & (ref["criterion"] == "first")
                 & np.isclose(ref["target"], 0.9)].sort_values("commit_index")
        assert (ec.fmt6(got) == ec.fmt6(r_["first_caught"])).all(), f"rf budget seed {s} per-defect catch != frozen policy"
        rf_b[s] = caught
    rf_b_caught = np.mean([rf_b[s] for s in SEEDS], axis=0)
    feat = {c: es.matrix(rows[c].to_numpy(dtype=float)) for c in
            DIFF_COLS + ["chg_cov_files", "meta_duration_mean", "eval_duration_ms", "hist_fail_last_1", "hist_fail_rate",
                         "hist_never_failed", "hist_since_fail"]}
    fit_mod = fit_modules_with_failures(ds, muts, manual)
    c_index = {int(c): i for i, c in enumerate(es.commits)}
    t_index = {t: j for j, t in enumerate(es.tests)}
    commits_meta = ds.commits.set_index("commit_index")

    recs, tests_long = [], []
    for i_fe, r in enumerate(fe.itertuples(index=False)):
        ci = c_index[int(r.commit_index)]
        fails = [t_index[t] for t in r.failing_tests.split(";")]
        source, op, module = module_and_op(r.defect_id, muts, manual)
        empty = np.mean([(scores[s][ci] >= taus[s]).sum() == 0 for s in SEEDS])
        n_sel = np.mean([(scores[s][ci] >= taus[s]).sum() for s in SEEDS])
        margin_fail = np.mean([scores[s][ci, fails].max() - taus[s] for s in SEEDS])
        margin_commit = np.mean([scores[s][ci].max() - taus[s] for s in SEEDS])
        best_rank = np.mean([pos[s][ci, fails].min() for s in SEEDS])
        cov = feat["chg_cov_files"][ci, fails]
        pred = feat["meta_duration_mean"][ci, fails]
        diff = {c: float(feat[c][ci, 0]) for c in DIFF_COLS}
        flags = {
            "a": bool((cov <= 0).all()),
            "b": bool(diff["chg_lines_deleted"] == 0 and diff["chg_lines_added"] > 0),
            "c": bool((pred > EXPENSIVE_MS).all()),
            "d": fit_mod.get(module, 0) == 0,
        }
        flags["e"] = not any(flags.values())
        missed = float(r.rf_tau_caught) <= MISSED_MAX_CAUGHT + 1e-12
        cm = commits_meta.loc[int(r.commit_index)]
        recs.append({
            "defect_id": r.defect_id, "commit_index": int(r.commit_index), "source": source, "mutator": op, "module": module,
            "commit_kind": cm["kind"], "n_open_defects": int(cm["n_open"]),
            "missed_rf_tau": missed, "rf_tau_caught": float(r.rf_tau_caught),
            "combined_B_caught": float(r.combined_B_caught), "missed_combined_B": float(r.combined_B_caught) <= MISSED_MAX_CAUGHT + 1e-12,
            "covers_B_caught": float(r.covers_B_caught), "rf_B_caught": float(rf_b_caught[i_fe]),
            "n_failing_tests": int(r.n_failing_tests), "failing_tests": r.failing_tests,
            "rf_best_rank": best_rank, "combined_B_rank": float(r.combined_B_rank),
            "rf_best_score_minus_tau": margin_fail, "rf_commit_max_score_minus_tau": margin_commit,
            "tau_empty_share": empty, "tau_selected_tests": n_sel, "rf_tau_time": float(r.rf_tau_time),
            "failing_cov_files": ";".join(f"{v:g}" for v in cov),
            "failing_pred_ms": ";".join(f"{v:.0f}" for v in pred),
            "failing_actual_ms": ";".join(f"{v:.0f}" for v in feat["eval_duration_ms"][ci, fails]),
            "failing_timeout": ";".join(str(int(v >= EXPECT_TIMEOUT_MS)) for v in feat["eval_duration_ms"][ci, fails]),
            "failing_hist_fail_last_1": ";".join(f"{v:g}" for v in feat["hist_fail_last_1"][ci, fails]),
            "failing_hist_never_failed": ";".join(f"{v:g}" for v in feat["hist_never_failed"][ci, fails]),
            "module_fit_positives": int(fit_mod.get(module, 0)),
            **diff, **{f"cat_{k}": v for k, v in flags.items()},
        })
        if missed:
            for j in fails:
                tests_long.append({
                    "defect_id": r.defect_id, "commit_index": int(r.commit_index), "test_id": es.tests[j],
                    "rf_rank_mean": np.mean([pos[s][ci, j] for s in SEEDS]),
                    "rf_score_mean": np.mean([scores[s][ci, j] for s in SEEDS]),
                    "tau_mean": np.mean(list(taus.values())),
                    "chg_cov_files": feat["chg_cov_files"][ci, j], "meta_duration_mean": feat["meta_duration_mean"][ci, j],
                    "eval_duration_ms": feat["eval_duration_ms"][ci, j], "timeout": int(feat["eval_duration_ms"][ci, j] >= EXPECT_TIMEOUT_MS),
                    "hist_fail_last_1": feat["hist_fail_last_1"][ci, j],
                    "hist_fail_rate": feat["hist_fail_rate"][ci, j], "hist_never_failed": feat["hist_never_failed"][ci, j],
                    "hist_since_fail": feat["hist_since_fail"][ci, j],
                })
    allfe = pd.DataFrame(recs)
    # Sanity: uncaught mass equals (1 - recall_first) * first exposures of 8B (h1.json).
    h1 = json.loads((ec.MAIN / "h1.json").read_text(encoding="utf-8"))
    rec_main = float(h1["main"]["recall_first"]["mean"])
    uncaught = float((1 - allfe["rf_tau_caught"]).sum())
    assert len(allfe) == h1["first_exposures"] and abs(uncaught - (1 - rec_main) * len(allfe)) < 1e-6, (uncaught, rec_main)

    missed = allfe[allfe["missed_rf_tau"]].copy()
    missed["description"] = [describe(r) for r in missed.itertuples(index=False)]
    missed["categories"] = ["".join(k for k in "abcde" if getattr(r, f"cat_{k}")) for r in missed.itertuples(index=False)]

    cat_rows = []
    for k, name in CATS.items():
        col = f"cat_{k}"
        caught = allfe[~allfe["missed_rf_tau"]]
        cat_rows.append({
            "category": k, "name": name,
            "missed_n": int(missed[col].sum()), "missed_share": float(missed[col].mean()),
            "missed_commits": int(missed.loc[missed[col], "commit_index"].nunique()),
            "caught_n": int(caught[col].sum()), "caught_share": float(caught[col].mean()),
        })
    cats = pd.DataFrame(cat_rows)

    rf_m, cb_m = set(missed["defect_id"]), set(allfe.loc[allfe["missed_combined_B"], "defect_id"])
    overlap = pd.DataFrame([{"defect_id": d, "missed_rf_tau": d in rf_m, "missed_combined_B": d in cb_m,
                             "rf_tau_caught": float(allfe.set_index("defect_id").loc[d, "rf_tau_caught"]),
                             "combined_B_caught": float(allfe.set_index("defect_id").loc[d, "combined_B_caught"])}
                            for d in sorted(rf_m | cb_m)])

    ec.write_csv(allfe, out / "first_exposures_categories.csv")
    ec.write_csv(missed.drop(columns=["missed_rf_tau"]), out / "missed.csv")
    ec.write_csv(pd.DataFrame(tests_long), out / "missed_failing_tests.csv")
    ec.write_csv(cats, out / "categories.csv")
    ec.write_csv(overlap, out / "overlap_combined.csv")
    (out / "examples.md").write_text(examples_md(missed, pd.DataFrame(tests_long), muts, manual), encoding="utf-8")

    summary = {
        "experiment": "error analysis of missed first exposures (protocol p. 6); rules: protocol section 9, entry 8D",
        "rules": {"missed": f"rf_tau_caught <= {MISSED_MAX_CAUGHT}", "expensive_ms": EXPENSIVE_MS, "categories": CATS},
        "first_exposures": int(len(allfe)), "missed": int(len(missed)), "missed_commits": int(missed["commit_index"].nunique()),
        "missed_by_source": missed["source"].value_counts().sort_index().to_dict(),
        "uncaught_mass": uncaught,
        "missed_with_empty_tau_selection_majority": int((missed["tau_empty_share"] >= 0.5).sum()),
        "missed_with_failing_test_in_top5": int((missed["rf_best_rank"] <= 5).sum()),
        "missed_caught_by_rf_budget_majority": int((missed["rf_B_caught"] > MISSED_MAX_CAUGHT).sum()),
        "missed_top5_caught_by_rf_budget_majority": int(((missed["rf_best_rank"] <= 5) & (missed["rf_B_caught"] > MISSED_MAX_CAUGHT)).sum()),
        "failing_tests_of_missed_with_timeout": [int(sum(int(x) for x in v.split(";"))) for v in missed["failing_timeout"]],
        "categories": cats.to_dict(orient="records"),
        "combined_B": {"missed": int(len(cb_m)), "shared_with_rf_tau": sorted(rf_m & cb_m),
                       "only_rf_tau": sorted(rf_m - cb_m), "only_combined_B": sorted(cb_m - rf_m)},
        "module_fit_positives": fit_mod,
        "excluded": "MAN-17 (no failing test, not detectable)",
    }
    ec.write_json(out / "summary.json", summary)
    ec.write_run_info(out, t0, started, ec.standard_inputs(ds) + [MUTANTS, MANUAL_INDEX, RUNS, ds.path / "commits.csv",
                      ec.MAIN / "per_commit_policies_test.csv", ec.MAIN / "h1.json"]
                      + [ec.MODELS / "rf" / f for f in ec.RF_CARD["seed_files"].values()], SEEDS,
                      {"rules_fixed_in": "notes/stage8-protocol.md, section 9, entry '8D, аналіз помилок'"})
    ec.log(f"missed {len(missed)} of {len(allfe)} first exposures ({missed['commit_index'].nunique()} commits); "
           + ", ".join(f"({r.category}) {r.missed_n}" for r in cats.itertuples()))
    ec.log(f"combined + B misses {len(cb_m)}; shared {len(rf_m & cb_m)}")


def n(x: float, digits: int) -> str:
    """Number with a decimal comma (Ukrainian text, DSTU 3008)."""
    return f"{float(x):.{digits}f}".replace(".", ",")


def describe(r) -> str:
    """Generated description of a missed first exposure from the descriptive columns."""
    parts = []
    if r.tau_empty_share >= 0.5:
        parts.append(f"τ не відібрав жодного тесту коміту ({n(r.tau_empty_share * 10, 0)} з 10 seed)")
    else:
        parts.append(f"τ відібрав у середньому {n(r.tau_selected_tests, 1)} тестів коміту, але не тести дефекту")
    parts.append(f"найкращий тест дефекту — {n(r.rf_best_rank, 1)}-й у ранжуванні RF, оцінка нижча за τ на {n(-r.rf_best_score_minus_tau, 3)}"
                 if r.rf_best_score_minus_tau < 0 else f"найкращий тест дефекту — {n(r.rf_best_rank, 1)}-й у ранжуванні RF")
    parts.append(f"діф: +{n(r.chg_lines_added, 0)}/−{n(r.chg_lines_deleted, 0)} рядків, {n(r.chg_files, 0)} файл(и), "
                 f"hunk-и: заміна {n(r.chg_hunks_modify, 0)}, додавання {n(r.chg_hunks_add, 0)}, видалення {n(r.chg_hunks_delete, 0)}")
    if r.n_open_defects > 1:
        parts.append(f"відкритих дефектів на коміті: {r.n_open_defects}")
    return "; ".join(parts)


def examples_md(missed: pd.DataFrame, tests: pd.DataFrame, muts: dict, manual: dict) -> str:
    runs = load_runs({int(c) for c in missed["commit_index"]})
    lines = ["# Розібрані приклади пропущених перших появ (8D)", "",
             "> Згенеровано `ml/error_analysis.py`. Приклади обрано вручну до виведення деталей (з урахуванням того, що "
             "вже було відомо з 8C): ручний баг з частковим відбором (MAN-04), ручний баг з порожнім відбором на «спокійному» "
             "коміті (MAN-20), мутант із 8 тестами, що падають (MUT-1062; стільки ж у MUT-1032), і два мутанти з інших модулів "
             "(MUT-593, MUT-270). Ранги й оцінки — середнє за 10 seed RF; τ — з картки; RF + B — бюджет з картки (перша поява, 0,90).", ""]
    for d in EXAMPLES:
        r = missed[missed["defect_id"] == d]
        if r.empty:
            lines += [f"## {d}", "", "не є пропущеною першою появою за правилом 8D", ""]
            continue
        r = r.iloc[0]
        run = runs[int(r["commit_index"])]
        lines += [f"## {d} — коміт {r['commit_index']} ({r['source']}, {r['mutator']}, модуль {r['module']})", ""]
        if d.startswith("MUT-"):
            m = muts[d]
            orig = m["original"] if len(m["original"]) <= 300 else m["original"][:300] + " …"
            lines += [f"- мутант Stryker `{m['mutator']}` у `{m['file']}`, рядок {m['line']}:",
                      "", "```diff", *[f"- {x}" for x in orig.splitlines()], *[f"+ {x}" for x in m["replacement"].splitlines()], "```", ""]
        else:
            patch = (MANUAL_DIR / manual[d]["patch"]).read_text(encoding="utf-8")
            body = [x for x in patch.splitlines() if x.startswith(("@@", "+", "-")) and not x.startswith(("+++", "---"))]
            lines += [f"- ручний баг: {manual[d]['description']} (`{manual[d]['file']}`)", "", "```diff", *body[:20], "```", ""]
        files = ", ".join(f"`{f['path']}` (+{f['added']}/−{f['deleted']})" for f in run["files"])
        lines += [f"- коміт `{run['kind']}`, файли: {files}",
                  f"- категорії: {r['categories']}; {r['description']}",
                  f"- RF + τ спіймано в {n(r['rf_tau_caught'] * 10, 0)} з 10 seed; RF + B (те саме ранжування) — в {n(r['rf_B_caught'] * 10, 0)} з 10; `combined` + B — частка перестановок {n(r['combined_B_caught'], 2)} "
                  f"(ранг {n(r['combined_B_rank'], 1)}); `covers` + B — {n(r['covers_B_caught'], 2)}",
                  "", "| Тест, що падає | Ранг RF | Оцінка RF | τ | `chg_cov_files` | `hist_fail_last_1` | прогноз, мс | факт, мс | тайм-аут |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for t in tests[tests["defect_id"] == d].itertuples(index=False):
            lines.append(f"| {t.test_id} | {n(t.rf_rank_mean, 1)} | {n(t.rf_score_mean, 3)} | {n(t.tau_mean, 3)} | {t.chg_cov_files:g} | "
                         f"{t.hist_fail_last_1:g} | {n(t.meta_duration_mean, 0)} | {n(t.eval_duration_ms, 0)} | {'так' if t.timeout else 'ні'} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
