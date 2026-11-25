"""Stage 7 (run only AFTER stage 8): the "working" model for the CLI, refit on all labelled commits.

Decision R1 (notes/stage7-plan.md, notes/decisions.md, p. 8): the CLI uses the frozen stage-6 model
(ml/models/rf, commits 0-323) until stage 8 has evaluated the hold-out test part. Only then this
script refits the same configuration (family, params, feature list from the card; constant columns
are NOT re-dropped, so the feature set stays the same) on commits 0-539 and stores it OUTSIDE ml/models,
in ml/models-prod/<family>/, so that stage 8 (which takes models with train_range [0, 324) from
ml/models) and ml/check_models.py never pick it up.

What the production refit may and may not do (enforced by ml/prioritize.py):
  * no metrics are computed here -- no data is left to evaluate it on;
  * tau and the calibration were tuned for the scores of the 0-323 model, so they are refused;
    the CLI allows the budget B and top-k settings of the frozen card (scale-free, borrowed).

Lock: a fit range that intersects the test part (432-539) needs --after-stage8 AND an existing
ml/results/stage8/ directory. A range before the test part is allowed (used by the self-check:
--fit-range 0 324 must reproduce ml/models/rf/model.joblib bit for bit).

Usage:
  ml/.venv/bin/python ml/train_production.py --after-stage8                     # 0-539 -> ml/models-prod/rf/
  ml/.venv/bin/python ml/train_production.py --fit-range 0 324 --out <dir> --check-against ml/models/rf/model.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from dataset_io import ROOT, dataset_fingerprint, load_dataset
from train import fit_pipeline, library_versions, make_pipeline

FROZEN = ROOT / "ml" / "models"
PROD = ROOT / "ml" / "models-prod"
STAGE8_RESULTS = ROOT / "ml" / "results" / "stage8"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="rf")
    ap.add_argument("--seed", type=int, help="default: the card's first seed")
    ap.add_argument("--fit-range", type=int, nargs=2, metavar=("FROM", "TO"), help="half-open; default: all commits")
    ap.add_argument("--after-stage8", action="store_true", help="confirm that stage 8 has evaluated the test part")
    ap.add_argument("--out", type=Path, help=f"output directory (default {PROD.relative_to(ROOT)}/<family>)")
    ap.add_argument("--check-against", type=Path, help="a saved model that must give identical val scores")
    args = ap.parse_args()

    card = json.loads((FROZEN / args.family / "model_card.json").read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else card["seeds"][0])
    probe = load_dataset()  # locked: only to read the split bounds
    n, (ta, tb) = probe.n_commits, probe.test_range
    fit = tuple(args.fit_range) if args.fit_range else (0, n)
    touches_test = fit[0] < tb and ta < fit[1]
    if touches_test and not (args.after_stage8 and STAGE8_RESULTS.is_dir()):
        print(f"error: fit range [{fit[0]}, {fit[1]}) uses the hold-out test part [{ta}, {tb}), which stays locked until "
              f"stage 8 has been evaluated (needs --after-stage8 and an existing {STAGE8_RESULTS.relative_to(ROOT)}/)",
              file=sys.stderr)
        return 2
    out = (args.out or PROD / args.family).resolve()
    if out.is_relative_to(FROZEN.resolve()):
        print(f"error: {out} is inside ml/models (stage-8 models only)", file=sys.stderr)
        return 2

    ds = load_dataset(allow_test=touches_test)
    rows = ds.slice(fit)
    features = list(card["features"])  # constant columns of 0-323 stay dropped, nothing new is dropped
    t0 = time.perf_counter()
    pipe = fit_pipeline(make_pipeline(args.family, card["params"], features, seed), args.family, rows, features)
    fit_s = time.perf_counter() - t0

    if args.check_against:
        a, b = ds.val_range
        val = ds.slice((a, b))
        ref = joblib.load(args.check_against).predict_proba(val)[:, 1]
        got = pipe.predict_proba(val)[:, 1]
        same = np.array_equal(ref, got)
        print(f"check against {args.check_against}: val scores {'identical' if same else 'DIFFERENT'} "
              f"(max |Δ| {np.abs(ref - got).max():.2e})")
        if not same:
            return 1

    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib", compress=3)
    prod_card = {
        "model": f"{args.family}-prod",
        "family": card["family"],
        "production": True,
        "feature_set": card["feature_set"],
        "features": features,
        "dropped_constant_features": card["dropped_constant_features"],
        "categorical": card["categorical"],
        "input": card["input"],
        "output": card["output"],
        "train_range": list(fit),
        "params": card["params"],
        "seeds": [seed],
        "seed_files": {str(seed): "model.joblib"},
        "fit_seconds": round(fit_s, 2),
        "positives_fit": int(rows["label"].sum()),
        "rows_fit": int(len(rows)),
        **dataset_fingerprint(ds),
        "versions": library_versions(),
        "created_by": "ml/train_production.py (stage 7 script)",
        "evaluation": "none: every labelled commit is in the fit range; the evaluated models are in ml/models",
        "policy": {k: card["policy"][k] for k in ("settings_per_seed", "by_recall_criterion")},
        "policy_source": {"card": f"ml/models/{args.family}/model_card.json", "seed": int(card["seeds"][0]),
                          "note": "B and k borrowed from the frozen model; tau and calibration are refused "
                                  "(tuned for another score scale)"},
    }
    (out / "model_card.json").write_text(json.dumps(prod_card, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}: {args.family}, fit commits [{fit[0]}, {fit[1]}) ({len(rows)} rows), seed {seed}, {fit_s:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
