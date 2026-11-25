"""Stage 8 (8A): acceptance check of the dry run of ml/run_stage8.py on val (notes/stage8-protocol.md, p. 7.3).

  1. frozen files      ml/check_frozen.py passes
  2. lock              `--range test` without both --allow-test and --h1-approved, or with another kp.json, is refused
                       before any data is read;
                       a val run into ml/results/stage8/ is refused; ml/results/stage8/ does not exist at the end
  3. frozen models     every model file reproduces ml/results/stage6/val_predictions.csv exactly (np.array_equal)
  4. stage 6 parity    the dry run reproduces val_metrics.csv, per_commit_val.csv and policies.csv of stage 6
                       for every stage 6 method except `<family>@no_commit_chg` (a training control without
                       saved models; it is not part of the stage 8 protocol), as stored (%.6f strings)
  5. H1 on val         RF + tau (first exposure, 0.90): recall_first 0.914, time 26.6 ± 0.8 % (decisions.md, p. 7);
                       h1.json / h2.json are marked "range": "val", "in_sample": true
  6. Kim-Porter        the score of commit c does not change when the labels of commit c change (only commits < c);
                       alpha = 1 equals hist_fail_last_1; alpha chosen by mean APFD on val
  7. determinism       a second run (fresh process) and a run with the Kim-Porter settings read from kp.json
                       (the code path of --range test) give byte-identical CSV/JSON (run_info.json without times);
                       figure titles of a test run are rewritten to the test range

Usage:  ml/.venv/bin/python ml/check_stage8.py   (runs the dry run if it is missing; ~40 s)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import baseline_kp as kp
from check_frozen import verify
from dataset_io import ROOT, load_dataset
from train import FAMILIES

PY = sys.executable
RUN = str(ROOT / "ml" / "run_stage8.py")
DRY = ROOT / "ml" / "results" / "stage8-dryrun"
S6 = ROOT / "ml" / "results" / "stage6"
STAGE8 = ROOT / "ml" / "results" / "stage8"
EXCLUDED = "@no_commit_chg"
TIME_FIELDS = ("started", "finished", "elapsed_seconds", "command")
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([PY, RUN, *args], cwd=ROOT, capture_output=True, text=True)


def compare_csv(ref_path: Path, got_path: Path, keys: list[str], name: str) -> None:
    """Every stage 6 row (except the excluded control) must exist in the dry run with identical %.6f strings."""
    ref = pd.read_csv(ref_path, dtype=str, keep_default_na=False)
    got = pd.read_csv(got_path, dtype=str, keep_default_na=False)
    ref = ref[~ref["method"].str.contains(EXCLUDED, regex=False)]
    common = [c for c in ref.columns if c in got.columns]
    missing_cols = [c for c in ref.columns if c not in got.columns]
    merged = ref[common].merge(got[common], on=keys, how="left", suffixes=("", "__got"), indicator=True)
    missing_rows = int((merged["_merge"] != "both").sum())
    bad = []
    for c in common:
        if c in keys:
            continue
        diff = merged[c] != merged[f"{c}__got"]
        if diff.any():
            bad.append(f"{c} ({int(diff.sum())})")
    check(f"{name}: {len(ref)} rows, {len(common)} columns identical", not missing_rows and not bad and not missing_cols,
          f"missing rows {missing_rows}; missing columns {missing_cols[:5]}; differing {bad[:6]}")


def strip_times(path: Path, kp_source: bool = False) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    d = {k: v for k, v in d.items() if k not in TIME_FIELDS}
    if kp_source:
        d["kim_porter"] = {k: v for k, v in d["kim_porter"].items() if k != "settings"}  # "chosen" vs "read from file"
    return d


def same_outputs(a: Path, b: Path, name: str, kp_source: bool = False) -> None:
    files = sorted(p.name for p in a.iterdir() if p.suffix in (".csv", ".json"))
    diff = []
    for f in files:
        if f == "run_info.json":
            if strip_times(a / f, kp_source) != strip_times(b / f, kp_source):
                diff.append(f)
        elif not (b / f).exists() or (a / f).read_bytes() != (b / f).read_bytes():
            diff.append(f)
    check(f"{name}: {len(files)} CSV/JSON files byte-identical (run_info.json without times)", not diff, ", ".join(diff))


def main() -> int:
    t0 = time.time()
    print("1. frozen files")
    problems = verify(verbose=False)
    check("ml/check_frozen.py", not problems, "; ".join(problems[:3]))

    print("2. lock")
    for flags in ([], ["--allow-test"], ["--h1-approved"]):
        res = run(["--range", "test", *flags, "--out", tempfile.mkdtemp(prefix="stage8-refused-")])
        check(f"--range test {' '.join(flags) or '(no flags)'} is refused", res.returncode != 0 and "--allow-test and --h1-approved" in res.stderr,
              f"exit {res.returncode}")
    import contextlib
    import io
    import run_stage8
    refused = False
    try:  # in-process: parse_args only parses, so this can never load data
        with contextlib.redirect_stderr(io.StringIO()):
            run_stage8.parse_args(["--range", "test", "--allow-test", "--h1-approved", "--kp-settings", str(ROOT / "ml" / "other-kp.json")])
    except SystemExit as e:
        refused = e.code != 0
    check("--range test with --kp-settings other than the frozen kp.json is refused", refused)
    res = run(["--range", "val", "--out", str(STAGE8 / "main"), "--no-figures"])
    check("val run into ml/results/stage8/ is refused", res.returncode != 0 and "must not write" in (res.stdout + res.stderr) and not STAGE8.exists())

    if not (DRY / "h1.json").exists():
        print("   (dry run missing: running ml/run_stage8.py --range val)")
        res = run(["--range", "val"])
        if res.returncode != 0:
            print(res.stdout[-2000:], res.stderr[-2000:])
            check("dry run", False)
            return 1

    ds = load_dataset()  # allow_test=False
    check("dataset loaded without the test part", int(ds.rows["commit_index"].max()) < ds.test_range[0])
    rows = ds.slice(ds.val_range)

    print("3. frozen models reproduce stage 6 val predictions exactly")
    preds = pd.read_csv(S6 / "val_predictions.csv", float_precision="round_trip")
    n_ok = n_all = 0
    for fam in FAMILIES:
        card = json.loads((ROOT / "ml" / "models" / fam / "model_card.json").read_text(encoding="utf-8"))
        for s, rel in card["seed_files"].items():
            pipe = joblib.load(ROOT / "ml" / "models" / fam / rel)
            ref = preds[(preds["model"] == fam) & (preds["seed"] == int(s))].sort_values(["commit_index", "test_id"], kind="mergesort")
            got = pipe.predict_proba(rows)[:, 1]
            n_all += 1
            n_ok += int(len(ref) == len(got) and np.array_equal(ref["score"].to_numpy(), got))
    check(f"{n_ok} of {n_all} model files: np.array_equal with val_predictions.csv", n_ok == n_all == 22)

    print("4. stage 6 parity (dry run vs ml/results/stage6)")
    compare_csv(S6 / "val_metrics.csv", DRY / "val_metrics.csv", ["method", "seed"], "val_metrics.csv")
    compare_csv(S6 / "per_commit_val.csv", DRY / "per_commit_val.csv", ["method", "seed", "commit_index"], "per_commit_val.csv")
    compare_csv(S6 / "policies.csv", DRY / "policies_val.csv", ["method", "seed", "criterion", "target", "policy"], "policies.csv")
    s6m = set(pd.read_csv(S6 / "val_metrics.csv")["method"])
    check("excluded only the no_commit_chg controls", sorted(m for m in s6m if EXCLUDED in m) == [f"{f}{EXCLUDED}" for f in sorted(FAMILIES)])

    print("5. H1 / H2 on val (in-sample pipeline check)")
    h1 = json.loads((DRY / "h1.json").read_text(encoding="utf-8"))
    h2 = json.loads((DRY / "h2.json").read_text(encoding="utf-8"))
    m = h1["main"]
    check("RF + tau: recall_first 0.914", round(m["recall_first"]["mean"], 3) == 0.914, f"{m['recall_first']['mean']:.4f}")
    check("RF + tau: time 26.6 ± 0.8 %", round(100 * m["time"]["mean"], 1) == 26.6 and round(100 * m["time"]["std"], 1) == 0.8,
          f"{100 * m['time']['mean']:.2f} ± {100 * m['time']['std']:.2f} %")
    check("35 first exposures, 61 pairs, 43 red commits on val", (h1["first_exposures"], h1["defect_pairs"], h1["red_commits"]) == (35, 61, 43))
    check("h1.json / h2.json marked range val, in_sample", h1["range"] == h2["range"] == "val" and h1["in_sample"] is True and h2["in_sample"] is True)
    ci = m["bootstrap_95"]["recall_first"]
    check("bootstrap interval contains the point estimate", ci["lo"] <= m["recall_first"]["mean"] <= ci["hi"], f"[{ci['lo']:.3f}, {ci['hi']:.3f}]")
    s6 = pd.read_csv(S6 / "val_metrics.csv")
    rf_apfd = s6[s6["method"] == "rf"]["apfd"].mean()
    check("H2 means = stage 6 mean APFD (rf over seeds, combined)", abs(h2["mean_model"] - rf_apfd) < 1e-6
          and abs(h2["mean_reference"] - float(s6[s6["method"] == "combined"]["apfd"].iloc[0])) < 1e-6)

    print("6. Kim-Porter")
    hist = ds.slice((0, ds.val_range[1]))
    _, y = kp.label_matrix(hist)
    rng = np.random.default_rng(0)
    ok_past, ok_future = True, True
    for alpha in kp.ALPHAS:
        h = kp.kp_matrix(y, alpha)
        for c in rng.integers(1, len(y) - 1, size=10):
            y2 = y.copy()
            y2[c] = 1 - y2[c]  # flip every label of commit c
            h2_ = kp.kp_matrix(y2, alpha)
            ok_past &= np.array_equal(h2_[: c + 1], h[: c + 1])
            ok_future &= not np.array_equal(h2_[c + 1], h[c + 1])
    check("changing the labels of commit c leaves scores of commits <= c unchanged", bool(ok_past))
    check("... and changes the scores of commit c + 1", bool(ok_future))
    check("alpha = 1 equals hist_fail_last_1", np.array_equal(kp.kp_matrix(y, 1.0), hist["hist_fail_last_1"].to_numpy(dtype=float).reshape(y.shape)))
    kpj = json.loads((DRY / "kp.json").read_text(encoding="utf-8"))
    best = max(kpj["alpha_grid"], key=lambda r: r["apfd"])
    check("alpha = argmax mean APFD on val", kpj["alpha"] == best["alpha"], f"alpha {kpj['alpha']}, APFD {best['apfd']:.4f}")

    print("7. determinism")
    with tempfile.TemporaryDirectory(prefix="stage8-rerun-") as d1, tempfile.TemporaryDirectory(prefix="stage8-kpfile-") as d2:
        r1 = run(["--range", "val", "--out", d1, "--no-figures"])
        r2 = run(["--range", "val", "--out", d2, "--no-figures", "--kp-settings", str(DRY / "kp.json")])
        check("re-runs succeeded", r1.returncode == 0 and r2.returncode == 0, (r1.stderr + r2.stderr)[-300:])
        if r1.returncode == 0:
            same_outputs(DRY, Path(d1), "fresh re-run")
        if r2.returncode == 0:
            same_outputs(DRY, Path(d2), "Kim-Porter settings from kp.json (test code path)", kp_source=True)
            ri = json.loads((Path(d2) / "run_info.json").read_text(encoding="utf-8"))
            check("   ... run_info says the settings were read from file", ri["kim_porter"]["settings"] == "read from file")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import run_stage8
    fig, ax = plt.subplots()
    with run_stage8.range_titles("test", (432, 540)):
        ax.set_title("Val (коміти 324–431): час до першого тесту, що впав")
        t1 = ax.get_title(loc="center")
        ax.set_title("Val: політики відбору (Random Forest, seed 20261117)")
        t2 = ax.get_title(loc="center")
    ax.set_title("Val: x")
    t3 = ax.get_title(loc="center")
    plt.close(fig)
    check("figure titles of a test run name the test range (figures.py unchanged)",
          t1.startswith("Test (коміти 432–539):") and t2.startswith("Test:") and t3 == "Val: x", f"{t1!r}, {t2!r}")

    check("ml/results/stage8/ does not exist", not STAGE8.exists())
    print(f"done in {time.time() - t0:.0f} s: {'all checks passed' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
