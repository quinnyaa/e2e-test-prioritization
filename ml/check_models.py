"""Stage 6: load every saved model in a fresh process (as the stage 7 CLI will) and check that it
reproduces the stored val predictions from raw dataset.csv rows.

The project's ml/ directory is removed from sys.path before unpickling, so the check fails if a
pipeline needs a project class; only sklearn/lightgbm/numpy objects are allowed in model.joblib.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ML_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(ML_DIR)]

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--predictions", required=True)
    args = ap.parse_args()
    models_dir, ds_dir = Path(args.models_dir), Path(args.dataset)
    splits = json.loads((ds_dir / "splits.json").read_text(encoding="utf-8"))
    a, b = splits["holdout"]["val"]
    rows = pd.read_csv(ds_dir / "dataset.csv", dtype={"meta_module": "string"})
    rows = rows[(rows["commit_index"] >= a) & (rows["commit_index"] < b)]
    rows = rows.sort_values(["commit_index", "test_id"], kind="mergesort").reset_index(drop=True)
    rows["meta_module"] = rows["meta_module"].astype(object)
    preds = pd.read_csv(args.predictions)
    failed = []
    checked = 0
    for card_path in sorted(models_dir.glob("*/model_card.json")):
        card = json.loads(card_path.read_text(encoding="utf-8"))
        assert card["train_range"][1] <= a, f"{card['model']}: trained beyond the val start"
        for seed, rel in card["seed_files"].items():
            pipe = joblib.load(card_path.parent / rel)
            score = pipe.predict_proba(rows)[:, 1]
            ref = preds[(preds["model"] == card["model"]) & (preds["seed"] == int(seed))]
            ref = ref.sort_values(["commit_index", "test_id"], kind="mergesort")["score"].to_numpy()
            ok = len(ref) == len(score) and np.allclose(score, ref, rtol=1e-6, atol=1e-9)
            checked += 1
            if not ok:
                failed.append(f"{card['model']} seed {seed}")
        if (card_path.parent / "calibrated.joblib").exists():
            cal = joblib.load(card_path.parent / "calibrated.joblib")
            p = cal.predict_proba(rows)[:, 1]
            if not (np.all(p >= 0) and np.all(p <= 1)):
                failed.append(f"{card['model']} calibrated")
    print(f"fresh-process model check: {checked} model file(s) loaded without project modules; "
          f"{'all reproduce' if not failed else 'MISMATCH: ' + ', '.join(failed)} the stored val predictions")
    sys.exit(1 if failed or checked == 0 else 0)


if __name__ == "__main__":
    main()
