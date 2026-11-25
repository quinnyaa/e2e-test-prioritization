"""Stage 8C, experiment 8: mutant operator, module and source of the defect (notes/stage8-protocol.md, p. 5.7; descriptive).

    ml/.venv/bin/python ml/exp_defect_types.py  # -> ml/results/stage8/defect_types/, ≈ 15 s

Grouping of first-exposure pairs: MUT-<id> -> `mutator` and `module` of mutation/mutants.json (ids are strings);
manual bugs -> module from the file of mutation/manual/index.json; source = mutant / manual.
Metric: recall_first of RF + tau (first exposure, 0.90; mean over 10 seeds) per group, with the number of first
exposures; combined and covers (B) alongside. Groups with < 5 first exposures are listed only (flag `small`),
without conclusions. The same for val 324-431 as context (in-sample there: tau and B were chosen on val).
Per-defect view: exp_common.range_first_exposures (frozen predictions; checked against the per-commit policy CSVs
of 8B / the 8A dry run).
"""

from __future__ import annotations

import json

import pandas as pd

import exp_common as ec
from exp_manual_bugs import MANUAL_INDEX, module_of
from train import SEEDS

MUTANTS = ec.ROOT / "mutation" / "mutants.json"
MIN_GROUP = 5
METHODS = ("rf_tau", "combined_B", "covers_B")


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    muts = {str(m["id"]): m for m in json.loads(MUTANTS.read_text(encoding="utf-8"))["mutants"]}
    manual = {m["id"]: m for m in json.loads(MANUAL_INDEX.read_text(encoding="utf-8"))}
    src, op, mod = [], [], []
    for d in df["defect_id"]:
        if d.startswith("MUT-"):
            m = muts[d.removeprefix("MUT-")]
            src.append("mutant"), op.append(m["mutator"]), mod.append(m["module"])
        else:
            src.append("manual"), op.append("manual"), mod.append(module_of(manual[d]["file"]))
    return df.assign(source=src, mutator=op, module=mod)


def grouped(df: pd.DataFrame, part: str) -> pd.DataFrame:
    out = []
    for by in ("mutator", "module", "source"):
        g = df.groupby(by, sort=True)
        t = g[[f"{m}_caught" for m in METHODS]].mean()
        t.columns = [f"recall_first_{m}" for m in METHODS]
        t.insert(0, "first_exposures", g.size())
        t = t.reset_index().rename(columns={by: "group"})
        t.insert(0, "grouping", by)
        out.append(t)
    res = pd.concat(out, ignore_index=True)
    res.insert(0, "range", part)
    res["small"] = res["first_exposures"] < MIN_GROUP
    return res


def main() -> None:
    t0, started = ec.started_now()
    out = ec.out_dir("defect_types")
    ds = ec.load()
    parts = {}
    for part, rng in (("test", ds.test_range), ("val", ds.val_range)):
        df, _ = ec.range_first_exposures(ds, rng, part)
        parts[part] = annotate(df).assign(range=part)
    per = pd.concat(parts.values(), ignore_index=True)
    table = pd.concat([grouped(parts["test"], "test"), grouped(parts["val"], "val (in-sample)")], ignore_index=True)
    ec.write_csv(per, out / "first_exposures.csv")
    ec.write_csv(table, out / "groups.csv")
    ec.write_json(out / "summary.json", {
        "experiment": "defect types (protocol p. 5.7), descriptive", "min_group_for_conclusions": MIN_GROUP,
        "first_exposures": {k: int(len(v)) for k, v in parts.items()},
        "overall_recall_first_rf_tau": {k: float(v["rf_tau_caught"].mean()) for k, v in parts.items()},
        "groups": table.to_dict(orient="records"),
    })
    ec.write_run_info(out, t0, started, ec.standard_inputs(ds) + [MUTANTS, MANUAL_INDEX, ec.MAIN / "per_commit_policies_test.csv",
                      ec.ROOT / "ml/results/stage8-dryrun/per_commit_policies_val.csv"]
                      + [ec.MODELS / "rf" / f for f in ec.RF_CARD["seed_files"].values()], SEEDS)
    t = table[(table["range"] == "test") & ~table["small"]]
    ec.log(f"test groups with >= {MIN_GROUP} first exposures: " + ", ".join(f"{r.group} {r.recall_first_rf_tau:.2f} ({r.first_exposures})" for r in t.itertuples()))


if __name__ == "__main__":
    main()
