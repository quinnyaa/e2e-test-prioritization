"""Post-hoc, exploratory analyses requested in the supervisor's review of 2026-11-25
(notes/review-supervisor-2026-11-25.md; logged in notes/stage8-protocol.md, section 9).

    ml/.venv/bin/python ml/exp_review.py   # -> ml/results/stage8/review/, a few seconds

Reads only the saved per-commit results of the one-time test evaluation (ml/results/stage8/main/) and, for the
worked example, the frozen RF of seed 20261117. Nothing is tuned; the H1/H2 verdicts are not revisited.

1. H2 as a paired comparison: wins / losses / ties of RF (mean over 10 seeds) vs combined per red commit,
   matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-) over non-zero differences (Kerby 2014),
   next to the Wilcoxon test and the unpaired Cliff's delta reported in 8B.
2. Sensitivity of the H2 comparison to the unit of APFD: failing test (as in 8B) vs defect (apfd_defect).
3. Worked APFDc example on one real test commit: order, durations, C_i and t_i of each failing test.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats as sst

import exp_common as ec
import stats as st
from baselines import COMBINED_VARIANTS
from metrics import N_PERM, evaluate, order_from
from train import SEEDS, predict

MAIN = ec.MAIN
OUT = ec.out_dir("review")
SEED = 20261117


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    nz = d[d != 0]
    ranks = sst.rankdata(np.abs(nz))
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    w = st.wilcoxon(a, b)
    return {
        "n": int(len(d)), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
        "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
        "wilcoxon_p": w["p"], "rank_biserial_paired": float((wp - wm) / (wp + wm)) if len(nz) else float("nan"),
        "cliffs_delta_unpaired": st.cliffs_delta(a, b), "mean_model": float(a.mean()), "mean_reference": float(b.mean()),
    }


COMBINED_VARIANT = "comb_last_cov_mod_recent"  # as in run_stage8.py (chosen on val in stage 6)


def main() -> None:
    per = pd.read_csv(MAIN / "per_commit_test.csv")  # only n_fail / n_defects (rounded metric values are not used)
    ds = ec.load()
    rows, es, keys = ec.eval_part(ds, ds.test_range)
    red_mask = es.y.any(1)
    # Exact per-commit metrics, recomputed as in run_stage8.py (the CSV is rounded and creates spurious ties).
    rf_per = [evaluate(es, predict(ec.frozen_rf(s), rows), n_perm=N_PERM, keys=keys) for s in SEEDS]
    comb_per = evaluate(es, COMBINED_VARIANTS[COMBINED_VARIANT](rows), n_perm=N_PERM, keys=keys)
    out: dict = {"note": "post-hoc, exploratory (supervisor's review 2026-11-25); H2 verdict unchanged",
                 "red_commits": "commits with at least one failing test (APFD defined)"}

    # 1-2. paired H2 description, unit = failing test and unit = defect
    for metric in ("apfd", "apfd_defect", "apfdc"):
        a = np.mean([p_[metric].to_numpy() for p_ in rf_per], axis=0)[red_mask]
        b = comb_per[metric].to_numpy()[red_mask]
        ok = ~(np.isnan(a) | np.isnan(b))
        out[metric] = paired(a[ok], b[ok])
    red = per[(per.method == "combined") & per.apfd.notna()].drop_duplicates("commit_index")
    out["units"] = {"failing_tests_per_red_commit_mean": float(red.n_fail.mean()),
                    "defects_per_red_commit_mean": float(red.n_defects.mean()),
                    "red_commits_with_more_failing_tests_than_defects": int((red.n_fail > red.n_defects).sum()),
                    "red_commits": int(len(red))}
    # Reproduce 8B (fatal if the saved per-commit file disagrees with h2.json).
    h2 = json.loads((MAIN / "h2.json").read_text(encoding="utf-8"))
    assert out["apfd"]["n"] == h2["n"] and abs(out["apfd"]["wilcoxon_p"] - h2["p"]) < 1e-12, "does not reproduce 8B"

    # 3. worked APFDc example: RF seed 20261117 on the red commit with the fewest failing tests and 2+ defects
    scores = es.matrix(predict(ec.frozen_rf(SEED), rows))
    order = order_from(scores, keys[0])
    cand = red[red.n_defects >= 2].sort_values(["n_fail", "commit_index"])
    c_idx = int(cand.commit_index.iloc[0])
    r = int(np.where(es.commits == c_idx)[0][0])
    o, dur, y = order[r], es.dur[r], es.y[r]
    T, cum, steps = float(dur.sum()), 0.0, []
    for pos, j in enumerate(o, start=1):
        cum += float(dur[j])
        if y[j]:
            steps.append({"position": pos, "test": str(es.tests[j]), "t_ms": float(dur[j]), "C_ms": cum,
                          "defects": [str(es.def_ids[k]) for k in range(len(es.def_ids))
                                      if es.def_commit[k] == r and es.def_mask[k, j]]})
    m = len(steps)
    apfdc = 1 - sum(s["C_ms"] - s["t_ms"] / 2 for s in steps) / (m * T)
    from metrics import _apfdc
    lib = float(_apfdc(o[None, :], dur[None, :], y[None, :])[0])
    assert abs(apfdc - lib) < 1e-12, (apfdc, lib)
    out["worked_example"] = {"model": f"rf seed {SEED}", "commit_index": c_idx, "n_tests": int(es.n), "T_ms": T,
                             "m_failing_tests": m, "failing_tests": steps, "apfdc": apfdc, "apfdc_library": lib}
    # 4. one commit, two units: a single defect that fails several tests (APFD by failing test vs by defect)
    one = red[(red.n_defects == 1) & (red.n_fail >= 3)].sort_values(["n_fail", "commit_index"])
    c2 = int(one.commit_index.iloc[0])
    r2 = int(np.where(es.commits == c2)[0][0])
    pos = np.empty(es.n, dtype=int)
    pos[order[r2]] = np.arange(1, es.n + 1)
    fpos = sorted(int(pos[j]) for j in np.where(es.y[r2])[0])
    n = es.n
    by_test = 1 - sum(fpos) / (n * len(fpos)) + 1 / (2 * n)
    by_defect = 1 - fpos[0] / n + 1 / (2 * n)
    lib = rf_per[0]["apfd"].to_numpy()[r2]
    assert abs(by_test - lib) < 1e-12, (by_test, lib)
    out["unit_example"] = {"model": f"rf seed {SEED}", "commit_index": c2, "n_tests": n, "defects": 1,
                           "failing_test_positions": fpos, "apfd_failing_test_unit": by_test,
                           "apfd_defect_unit": by_defect,
                           "apfd_defect_library": float(rf_per[0]["apfd_defect"].to_numpy()[r2])}
    ec.write_json(OUT / "review.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
