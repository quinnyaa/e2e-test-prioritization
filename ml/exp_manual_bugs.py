"""Stage 8C, experiment 7: manual bugs in the test part (notes/stage8-protocol.md, p. 5.8; descriptive, no tests).

    ml/.venv/bin/python ml/exp_manual_bugs.py   # -> ml/results/stage8/manual_bugs/, ≈ 10 s

For each of the 10 manual bugs introduced in 432-539 (defects.csv, source = manual): introduction commit; share of
RF seeds whose tau selection (first exposure, 0.90) catches its first exposure; mean rank of its first failing test
in the RF ranking; actual time share of the selection; the same for combined and covers with B.
Data source: the 8B CSVs are per commit (first_caught, first_pos of any failing test), so a per-defect catch / rank
cannot be read from them where a commit has several first exposures. The per-defect view is rebuilt from the
frozen models' predictions (no fit, no tuning) and checked against per_commit_policies_test.csv (exp_common).
A manual bug with no failing test (isolated_detected = 0) has no first-exposure pair: reported as undetectable.
"""

from __future__ import annotations

import json

import pandas as pd

import exp_common as ec
from train import SEEDS

MANUAL_INDEX = ec.ROOT / "mutation" / "manual" / "index.json"


def module_of(path: str) -> str:
    parts = path.split("/")
    return parts[2] if len(parts) > 2 and parts[:2] == ["src", "modules"] else "/".join(parts[:-1])


def main() -> None:
    t0, started = ec.started_now()
    out = ec.out_dir("manual_bugs")
    ds = ec.load()
    df, es = ec.range_first_exposures(ds, ds.test_range, "test")
    a, b = ds.test_range
    index = {m["id"]: m for m in json.loads(MANUAL_INDEX.read_text(encoding="utf-8"))}
    man = ds.defects[(ds.defects["source"] == "manual") & ds.defects["introduced_at"].between(a, b - 1)]
    rows = []
    for d in man.sort_values("introduced_at").itertuples(index=False):
        r = df[df["defect_id"] == d.defect_id]
        base = {"defect_id": d.defect_id, "introduced_at": int(d.introduced_at), "fixed_at": int(d.fixed_at),
                "file": index[d.defect_id]["file"], "module": module_of(index[d.defect_id]["file"]),
                "description": index[d.defect_id]["description"],
                "first_exposures_in_commit": int(es.first_per_commit[list(es.commits).index(d.introduced_at)])}
        if r.empty:
            rows.append({**base, "detectable": False, "n_failing_tests": 0, "failing_tests": ""})
        else:
            rows.append({**base, "detectable": True, **r.iloc[0].drop(["defect_id", "commit_index"]).to_dict()})
    res = pd.DataFrame(rows)
    det = res[res["detectable"]]
    summ = {"manual_bugs": int(len(res)), "detectable": int(len(det)),
            "undetectable": res.loc[~res["detectable"], "defect_id"].tolist()}
    for m in ("rf_tau", "combined_B", "covers_B"):
        summ[m] = {"mean_caught_share": float(det[f"{m}_caught"].mean()), "caught_all_seeds": int((det[f"{m}_caught"] == 1).sum()),
                   "caught_no_seed": int((det[f"{m}_caught"] == 0).sum()), "mean_rank": float(det[f"{m}_rank"].mean()),
                   "mean_time": float(det[f"{m}_time"].mean())}
    # Mutants of the same test part, for context.
    mut = df[df["defect_id"].str.startswith("MUT-")]
    summ["mutants_first_exposures"] = int(len(mut))
    for m in ("rf_tau", "combined_B", "covers_B"):
        summ[f"mutants_{m}_mean_caught_share"] = float(mut[f"{m}_caught"].mean())
        summ[f"mutants_{m}_mean_rank"] = float(mut[f"{m}_rank"].mean())
    ec.write_csv(res, out / "manual_bugs.csv")
    ec.write_csv(df, out / "first_exposures_test.csv")
    ec.write_json(out / "summary.json", {
        "experiment": "manual bugs (protocol p. 5.8), descriptive", "range": [a, b],
        "settings": {"rf": "frozen RF, card tau (first, 0.90), 10 seeds", "combined": f"B = {ec.COMBINED_B}", "covers": f"B = {ec.COVERS_B}"},
        "data_source": "per-defect view rebuilt from frozen predictions; per-commit sums equal 8B per_commit_policies_test.csv",
        "rank": "1-based position of the first test failing because of the defect (RF: mean over seeds; heuristics: mean over 20 tie permutations)",
        **summ,
    })
    ec.write_run_info(out, t0, started, ec.standard_inputs(ds) + [MANUAL_INDEX, ec.MAIN / "per_commit_policies_test.csv"]
                      + [ec.MODELS / "rf" / f for f in ec.RF_CARD["seed_files"].values()], SEEDS)
    ec.log(f"manual bugs: {len(det)}/{len(res)} detectable; RF+tau caught share {summ['rf_tau']['mean_caught_share']:.2f}, "
           f"combined {summ['combined_B']['mean_caught_share']:.2f}, covers {summ['covers_B']['mean_caught_share']:.2f}")


if __name__ == "__main__":
    main()
