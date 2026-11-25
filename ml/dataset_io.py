"""Stage 6: dataset loading, named feature sets and the hold-out test lock.

The hold-out test part (`splits.json` -> holdout.test of the loaded dataset, 432-539 for the final
dataset) is physically removed from memory right after reading unless `allow_test=True`. Every
range-based function goes through `Dataset.check_range`, which refuses a range that touches the
test part when the lock is on. Stage 6 never sets `allow_test`.

Analysis-only files (`commits.csv`, `defects.csv`, `positives.csv`) are loaded for evaluation
(defect-level APFD) and reporting; their columns can never become features (asserted).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "dataset" / "history-600-540"
SEED = 20261117

# Fixed category list for `meta_module` (test ID prefixes of tests/manifest.json), so that the
# encoding never depends on which split was seen.
MODULES = ["auth", "catalog", "product", "search", "cart", "checkout", "orders", "admin", "e2e"]
CATEGORICAL = ["meta_module"]

ID_COLUMNS = ["commit_index", "sha", "test_id", "split"]
LABEL = "label"
EVAL_COLUMNS = ["eval_duration_ms", "eval_flaky"]
FEATURE_PREFIXES = ("chg_", "hist_", "meta_", "ci_")
# Pair-level `chg_` features (diff x test); the remaining `chg_` features are commit-level.
PAIR_CHG_PREFIXES = ("chg_cov_", "chg_mod_", "chg_same_")
N_COMMIT_CHG = 16

FEATURE_SETS = ("main", "main+ci", "no_commit_chg", "hist", "chg", "meta")

Range = tuple[int, int]  # half-open [from, to) commit-index range


class TestLockError(RuntimeError):
    """Raised when code touches the hold-out test part without `allow_test=True`."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Dataset:
    path: Path
    meta: dict
    splits: dict
    rows: pd.DataFrame  # dataset.csv rows; test rows removed unless allow_test
    commits: pd.DataFrame  # analysis only
    defects: pd.DataFrame  # analysis only
    positives: pd.DataFrame  # analysis only
    allow_test: bool
    n_commits: int
    tests: list[str]

    # ---------- ranges ----------

    @property
    def train_range(self) -> Range:
        return tuple(self.splits["holdout"]["train"])

    @property
    def val_range(self) -> Range:
        return tuple(self.splits["holdout"]["val"])

    @property
    def test_range(self) -> Range:
        return tuple(self.splits["holdout"]["test"])

    def check_range(self, rng: Range) -> Range:
        a, b = int(rng[0]), int(rng[1])
        if not 0 <= a < b <= self.n_commits:
            raise ValueError(f"bad commit range [{a}, {b}) for {self.n_commits} commits")
        ta, tb = self.test_range
        if a < tb and ta < b and not self.allow_test:
            raise TestLockError(
                f"range [{a}, {b}) intersects the hold-out test part [{ta}, {tb}); "
                "pass allow_test=True explicitly (stage 8 only)"
            )
        return a, b

    def slice(self, rng: Range) -> pd.DataFrame:
        """Rows of commits in [a, b), sorted by (commit_index, test_id); every commit has all tests."""
        a, b = self.check_range(rng)
        part = self.rows[(self.rows["commit_index"] >= a) & (self.rows["commit_index"] < b)]
        part = part.sort_values(["commit_index", "test_id"], kind="mergesort").reset_index(drop=True)
        if len(part) != (b - a) * len(self.tests):
            raise ValueError(f"range [{a}, {b}): expected {(b - a) * len(self.tests)} rows, got {len(part)}")
        return part

    @property
    def introduced_at(self) -> dict[str, int]:
        """defect_id -> introduction commit (analysis only; for first-exposure metrics)."""
        return {str(d): int(c) for d, c in zip(self.defects["defect_id"], self.defects["introduced_at"])}

    def positives_in(self, rng: Range) -> pd.DataFrame:
        a, b = self.check_range(rng)
        p = self.positives
        return p[(p["commit_index"] >= a) & (p["commit_index"] < b)].reset_index(drop=True)

    # ---------- features ----------

    @property
    def groups(self) -> dict[str, list[str]]:
        return self.meta["columns"]["featureGroups"]

    def commit_level_chg(self) -> list[str]:
        chg = self.groups["chg"]
        commit_level = [c for c in chg if not c.startswith(PAIR_CHG_PREFIXES)]
        if len(commit_level) != N_COMMIT_CHG or commit_level != chg[:N_COMMIT_CHG]:
            raise AssertionError(f"expected the first {N_COMMIT_CHG} chg_ features to be commit-level: {commit_level}")
        return commit_level

    def feature_set(self, name: str) -> list[str]:
        main = list(self.meta["columns"]["mainFeatureSet"])
        if name == "main":
            cols = main
        elif name == "main+ci":
            cols = main + list(self.groups["ci"])
        elif name == "no_commit_chg":
            drop = set(self.commit_level_chg())
            cols = [c for c in main if c not in drop]
        elif name in ("hist", "chg", "meta"):
            cols = list(self.groups[name])
        else:
            raise ValueError(f"unknown feature set {name!r}; known: {FEATURE_SETS}")
        self._assert_features(cols)
        return cols

    def _assert_features(self, cols: list[str]) -> None:
        forbidden = set(ID_COLUMNS) | {LABEL} | set(EVAL_COLUMNS)
        analysis = set(self.commits.columns) | set(self.defects.columns) | set(self.positives.columns)
        for c in cols:
            assert c.startswith(FEATURE_PREFIXES), f"{c}: not a feature column"
            assert c not in forbidden, f"{c}: service/label/eval column used as a feature"
            assert c not in analysis, f"{c}: analysis-only column used as a feature"
            assert c in self.rows.columns, f"{c}: missing from dataset.csv"

    def assert_commit_level_constant(self, rng: Range) -> None:
        """The commit-level chg_ features must be constant within every commit of the range."""
        part = self.slice(rng)
        nunique = part.groupby("commit_index")[self.commit_level_chg()].nunique().max()
        bad = nunique[nunique > 1]
        if len(bad):
            raise AssertionError(f"commit-level chg_ features vary within a commit: {list(bad.index)}")
        pair = [c for c in self.groups["chg"] if c.startswith(PAIR_CHG_PREFIXES)]
        varying = part.groupby("commit_index")[pair].nunique().max()
        if (varying <= 1).any():
            raise AssertionError(f"pair-level chg_ features never vary within a commit: {list(varying[varying <= 1].index)}")


def load_dataset(path: Path | str = DEFAULT_DATASET, allow_test: bool = False) -> Dataset:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    splits = json.loads((path / "splits.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(path / "dataset.csv", dtype={"meta_module": "string"})
    commits = pd.read_csv(path / "commits.csv")
    defects = pd.read_csv(path / "defects.csv")
    positives = pd.read_csv(path / "positives.csv")

    # Integrity of the split column against the index ranges.
    n_commits = int(meta["counts"]["commits"])
    for name, (a, b) in splits["holdout"].items():
        in_range = rows["commit_index"].between(a, b - 1)
        if not (rows.loc[in_range, "split"] == name).all() or not (rows.loc[rows["split"] == name, "commit_index"].between(a, b - 1)).all():
            raise AssertionError(f"split column disagrees with holdout.{name} = [{a}, {b})")
    tests = sorted(rows["test_id"].unique())
    if len(rows) != n_commits * len(tests):
        raise AssertionError("dataset.csv is not a full commit x test grid")
    unknown = set(rows["meta_module"].unique()) - set(MODULES)
    if unknown:
        raise AssertionError(f"unknown meta_module values: {sorted(unknown)}")
    rows["meta_module"] = rows["meta_module"].astype(object)

    ta, tb = splits["holdout"]["test"]
    if not allow_test:
        keep = ~rows["commit_index"].between(ta, tb - 1)
        rows = rows[keep].reset_index(drop=True)
        commits = commits[~commits["commit_index"].between(ta, tb - 1)].reset_index(drop=True)
        defects = defects[~defects["introduced_at"].between(ta, tb - 1)].reset_index(drop=True)
        positives = positives[~positives["commit_index"].between(ta, tb - 1)].reset_index(drop=True)

    ds = Dataset(path, meta, splits, rows, commits, defects, positives, allow_test, n_commits, tests)
    ds.feature_set("main+ci")  # asserts the column whitelist
    return ds


def drop_constant(rows: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """Split features into (kept, dropped) by whether they are constant on `rows` (the fit part)."""
    nunique = rows[features].nunique(dropna=False)
    dropped = [c for c in features if nunique[c] <= 1]
    return [c for c in features if c not in dropped], dropped


def dataset_fingerprint(ds: Dataset) -> dict:
    return {
        "dataset": str(ds.path.relative_to(ROOT)) if ds.path.is_relative_to(ROOT) else str(ds.path),
        "dataset_csv_sha256": sha256(ds.path / "dataset.csv"),
        "meta_json_sha256": sha256(ds.path / "meta.json"),
        "splits_json_sha256": sha256(ds.path / "splits.json"),
    }


def assert_non_negative(rows: pd.DataFrame, features: list[str]) -> None:
    num = [c for c in features if c not in CATEGORICAL]
    mins = rows[num].min()
    if (mins < 0).any():
        raise AssertionError(f"negative feature values: {list(mins[mins < 0].index)}")
    if rows[num].isna().any().any():
        raise AssertionError("missing feature values")


__all__ = [
    "CATEGORICAL", "DEFAULT_DATASET", "Dataset", "FEATURE_SETS", "MODULES", "ROOT", "Range", "SEED",
    "TestLockError", "assert_non_negative", "dataset_fingerprint", "drop_constant", "load_dataset", "sha256",
]

if __name__ == "__main__":
    ds = load_dataset()
    print("rows in memory:", len(ds.rows), "commits:", ds.rows["commit_index"].nunique())
    for fs in FEATURE_SETS:
        print(fs, len(ds.feature_set(fs)))
    ds.assert_commit_level_constant(ds.train_range)
    try:
        ds.slice((400, 440))
    except TestLockError as e:
        print("lock ok:", e)
    print(np.unique(ds.rows["split"]))
