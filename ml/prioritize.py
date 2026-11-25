"""Stage 7: score, rank and select the E2E tests of ONE commit (library behind ml/cli.py).

Methods:
  * a saved model of stage 6 (`ml/models/<rf|lgbm|hgb|lr>/`, trained on commits 0-323) -- the default is
    Random Forest, seed 20261117 (notes/decisions.md, p. 7);
  * the production refit (`ml/models/rf-prod/`, `ml/train_production.py`; only after stage 8);
  * the heuristics of stage 6 (`covers` = covers ∩ diff, `combined`) for comparison in the demo.

Ranking and ties follow ml/metrics.py: descending uncalibrated score, ties broken by the random key of
permutation 0, `default_rng([20261117, commit_index])` (never by test_id). For a new commit
commit_index = number of commits in the history before it.

Policies (ml/policy.py semantics, settings from model_card.json; nothing is tuned here):
  threshold  score >= tau            (default: first-exposure criterion, target 0.90 -- the H1 setting)
  budget     rank order while cumulative predicted duration <= B * predicted full run
  topk       first k tests
  all        every test in rank order (prioritization only)
tau and the calibration were fitted to the scores of the 0-323 model on val; they are refused for the
production refit, whose score scale differs (notes/stage7-plan.md, R1).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from baselines import BASELINES, COMBINED_VARIANTS
from metrics import TIE_SEED, order_from

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "ml" / "models"
PROD_MODELS_DIR = ROOT / "ml" / "models-prod"
STAGE6_RESULTS = ROOT / "ml" / "results" / "stage6"
DEFAULT_MODEL = "rf"
HEURISTICS = {"covers": BASELINES["covers"], "combined": COMBINED_VARIANTS["comb_last_cov_mod_recent"]}
POLICIES = ("threshold", "budget", "topk", "all")
CRITERIA = ("first", "commit")  # recall criterion the setting was tuned for (policy.py)
TARGETS = ("0.90", "0.95")


class PolicyError(ValueError):
    pass


@dataclass
class Method:
    name: str  # rf / lgbm / hgb / lr / rf-prod / covers / combined
    kind: str  # "model" | "heuristic"
    seed: int | None = None
    card: dict = field(default_factory=dict)
    pipe: object = None
    calibrated: object = None
    production: bool = False

    @property
    def label(self) -> str:
        if self.kind == "heuristic":
            return f"heuristic:{self.name}"
        a, b = self.card['train_range']
        return f"{self.name} (seed {self.seed}, train commits [{a}, {b}))"


def load_method(name: str = DEFAULT_MODEL, seed: int | None = None, models_dir: Path = MODELS_DIR) -> Method:
    if name in HEURISTICS:
        return Method(name, "heuristic")
    # "<family>-prod" = the production refit of ml/train_production.py, stored outside ml/models
    mdir = PROD_MODELS_DIR / name.removesuffix("-prod") if name.endswith("-prod") else models_dir / name
    card_path = mdir / "model_card.json"
    if not card_path.exists():
        hint = " (run ml/train_production.py after stage 8)" if name.endswith("-prod") else ""
        raise FileNotFoundError(f"{card_path} not found{hint}")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    seed = int(seed if seed is not None else card["seeds"][0])
    rel = card["seed_files"].get(str(seed))
    if rel is None:
        raise ValueError(f"{name}: no saved model for seed {seed}; saved: {sorted(card['seed_files'])}")
    pipe = joblib.load(mdir / rel)
    production = bool(card.get("production", False))
    calibrated = None
    # The calibrator wraps the seed-0 model of stage 6; it is valid only for that exact model.
    if not production and seed == int(card["seeds"][0]) and (mdir / "calibrated.joblib").exists():
        calibrated = joblib.load(mdir / "calibrated.joblib")
    return Method(name, "model", seed, card, pipe, calibrated, production)


# ---------------------------------------------------------------------- scores and ranking

def scores(method: Method, rows: pd.DataFrame) -> np.ndarray:
    if method.kind == "heuristic":
        return np.asarray(HEURISTICS[method.name](rows), dtype=float)
    return method.pipe.predict_proba(rows)[:, 1]


def calibrated_p(method: Method, rows: pd.DataFrame) -> np.ndarray | None:
    return None if method.calibrated is None else method.calibrated.predict_proba(rows)[:, 1]


def tie_key(commit_index: int, n: int, seed: int = TIE_SEED) -> np.ndarray:
    """Permutation-0 key of ml/metrics.py::tie_keys for this commit."""
    return np.random.default_rng([seed, int(commit_index)]).random((1, n))[0]


def rank_order(score: np.ndarray, commit_index: int) -> np.ndarray:
    """Test indices in run order (descending score, random ties)."""
    return order_from(score[None, :], tie_key(commit_index, len(score))[None, :])[0]


# ---------------------------------------------------------------------- policies

@dataclass
class Policy:
    name: str
    setting: float | None  # tau / B (share of predicted time) / k (number of tests)
    source: str  # where the setting comes from

    def describe(self) -> str:
        if self.name == "threshold":
            return f"поріг τ = {self.setting:.4f} ({self.source})"
        if self.name == "budget":
            return f"бюджет B = {100 * self.setting:.0f} % прогнозного часу ({self.source})"
        if self.name == "topk":
            k = f"{int(self.setting)}" if self.setting >= 1 else f"{100 * self.setting:.1f} %"
            return f"top-k: {k} тестів ({self.source})"
        return "усі тести в порядку пріоритету (лише пріоритизація)"


def heuristic_budget(method: str, criterion: str, target: str) -> float:
    """B of a heuristic chosen on val by the same procedure as for the models (policies.csv, stage 6)."""
    p = pd.read_csv(STAGE6_RESULTS / "policies.csv")
    row = p[(p["method"] == method) & (p["criterion"] == criterion) & (p["policy"] == "budget")
            & (np.isclose(p["target"], float(target)))]
    if row.empty:
        raise PolicyError(f"no budget setting for {method} / {criterion} / {target} in policies.csv")
    return float(row["setting"].iloc[0])


def resolve_policy(method: Method, name: str = "threshold", criterion: str = "first", target: str = "0.90",
                   value: float | None = None) -> Policy:
    """Policy setting from the model card (or an explicit `value`)."""
    if name not in POLICIES:
        raise PolicyError(f"unknown policy {name!r}; known: {POLICIES}")
    if criterion not in CRITERIA or target not in TARGETS:
        raise PolicyError(f"criterion must be one of {CRITERIA}, target one of {TARGETS}")
    if name == "all":
        return Policy("all", None, "без відбору")
    if name == "threshold" and method.kind == "heuristic":
        raise PolicyError("a heuristic score is a rank, not a probability: use --policy budget or topk")
    if name == "threshold" and method.production and value is None:
        raise PolicyError("tau was tuned for the 0-323 model; the production refit has another score scale "
                          "(use --policy budget or topk, or pass an explicit --value)")
    if value is not None:
        return Policy(name, float(value), "задано вручну")
    if method.kind == "heuristic":
        if name == "topk":
            raise PolicyError("pass --value K for top-k with a heuristic")
        return Policy("budget", heuristic_budget(method.name, criterion, target),
                      f"policies.csv етапу 6: {method.name}, критерій {criterion}, ціль {target}")
    # The production refit stores the policy block of the frozen card it borrows B / k from.
    pol = method.card["policy"]
    seed = str(method.seed if not method.production else method.card["policy_source"]["seed"])
    table = pol["by_recall_criterion"]["first"] if criterion == "first" else pol["settings_per_seed"]
    key = {"threshold": "threshold_tau", "budget": "budget_share_of_predicted_time",
           "topk": "topk_share_of_tests"}[name]
    setting = float(table[key][seed][target])  # top-k: share of tests (< 1)
    src = f"model_card.json: критерій {criterion}, ціль {target}, seed {seed}"
    return Policy(name, setting, src)


def select(policy: Policy, score: np.ndarray, order: np.ndarray, pred_dur: np.ndarray) -> np.ndarray:
    """Boolean mask of selected tests (column order = rows order)."""
    n = len(score)
    if policy.name == "all":
        return np.ones(n, dtype=bool)
    if policy.name == "threshold":
        return score >= policy.setting
    pos = np.empty(n, dtype=int)
    pos[order] = np.arange(1, n + 1)
    if policy.name == "topk":
        # setting >= 1: number of tests; < 1: share of tests from the card (stored as s / n, so round)
        k = int(policy.setting) if policy.setting >= 1 else int(round(policy.setting * n))
        return pos <= k
    cum = np.empty(n)
    cum[order] = np.cumsum(pred_dur[order])
    return cum <= policy.setting * pred_dur.sum() + 1e-9  # ml/metrics.py: stops at the first overflow


# ---------------------------------------------------------------------- one commit

def reasons(row: pd.Series) -> str:
    out = []
    if row["chg_cov_files"] > 0:
        out.append(f"covers ∩ діф: {int(row['chg_cov_files'])} ф.")
    elif row["chg_mod_files"] > 0:
        out.append(f"модуль ∩ діф: {int(row['chg_mod_files'])} ф.")
    if row["hist_fail_last_1"] > 0:
        out.append("впав на попередньому коміті")
    elif row["hist_since_fail"] < 100:
        out.append(f"падав {int(row['hist_since_fail'])} ком. тому")
    if row["hist_file_fail_rate"] > 0:
        out.append(f"асоціація файл→тест {row['hist_file_fail_rate']:.2f}")
    return "; ".join(out)


def prioritize(method: Method, rows: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    """Ranked table of the commit's tests: rank, score, p (calibrated), predicted time, selected, reasons."""
    rows = rows.sort_values("test_id", kind="mergesort").reset_index(drop=True)
    commit = int(rows["commit_index"].iloc[0])
    s = scores(method, rows)
    order = rank_order(s, commit)
    pred = rows["meta_duration_mean"].to_numpy(dtype=float)
    sel = select(policy, s, order, pred)
    p = calibrated_p(method, rows)
    rank = np.empty(len(s), dtype=int)
    rank[order] = np.arange(1, len(s) + 1)
    out = pd.DataFrame({
        "rank": rank,
        "test_id": rows["test_id"],
        "module": rows["meta_module"],
        "score": s,
        "p": p if p is not None else np.nan,
        "pred_ms": pred,
        "selected": sel,
        "reasons": [reasons(r) for _, r in rows.iterrows()],
    })
    return out.sort_values("rank").reset_index(drop=True)
