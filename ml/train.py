"""Stage 6: pointwise ML models (LR, RF, HistGradientBoosting, LightGBM) and range-parametrised tuning.

`tune(ds, family, feature_set, fit_range, val_range)` fits every grid configuration on `fit_range`
and scores it on `val_range` by the mean APFD over red commits (metrics.py). Nothing here is tied
to the hold-out ranges, so stage 8 (rolling windows) calls the same functions with other ranges.
Every fitted object is a full sklearn Pipeline that takes raw dataset.csv columns (feature
selection, constant-column drop, log1p/scaling/encoding inside), contains only sklearn/lightgbm
classes and can be loaded with joblib in a fresh process.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from dataset_io import CATEGORICAL, MODULES, SEED, Dataset, Range, assert_non_negative, drop_constant
from metrics import N_PERM, EvalSet, build_eval_set, classification_diagnostics, evaluate, summarize, tie_keys

FAMILIES = ("lr", "rf", "hgb", "lgbm")
FAMILY_LABELS = {
    "lr": "логістична регресія",
    "rf": "Random Forest",
    "hgb": "HistGradientBoosting",
    "lgbm": "LightGBM",
}
SEEDS = tuple(SEED + i for i in range(10))
N_JOBS = 4  # fixed thread count (RF and LightGBM results do not depend on it, but timings do)
ITERATIONS = (100, 200, 400)  # boosting rounds evaluated from one fit (staged / num_iteration)


def grid(family: str, spw: float) -> list[dict]:
    """Hyper-parameter grid (<= 18 configurations per family; val is small)."""
    if family == "lr":
        return [{"C": c, "class_weight": cw} for cw in (None, "balanced") for c in (0.01, 0.1, 1.0, 10.0)]
    if family == "rf":
        return [
            {"max_depth": d, "min_samples_leaf": leaf, "class_weight": cw}
            for cw in (None, "balanced_subsample") for d in (None, 8, 16) for leaf in (1, 5, 20)
        ]
    if family == "hgb":
        return [
            {"max_leaf_nodes": nl, "class_weight": cw, "max_iter": it}
            for cw in (None, "balanced") for nl in (7, 15, 31) for it in ITERATIONS
        ]
    if family == "lgbm":
        return [
            {"num_leaves": nl, "scale_pos_weight": w, "n_estimators": it}
            for w in (1.0, spw) for nl in (7, 15, 31) for it in ITERATIONS
        ]
    raise ValueError(family)


def log_columns(features: list[str]) -> list[str]:
    """Count-like columns that get log1p in the linear model (rates and fractions stay as they are)."""
    return [c for c in features if c not in CATEGORICAL and "_rate" not in c and "_frac" not in c]


def make_pipeline(family: str, params: dict, features: list[str], seed: int, n_iter: int | None = None) -> Pipeline:
    cat = [c for c in features if c in CATEGORICAL]
    num = [c for c in features if c not in CATEGORICAL]
    if family == "lr":
        logc = log_columns(num)
        lin = [c for c in num if c not in logc]
        parts = [
            ("log", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())]), logc),
            ("lin", StandardScaler(), lin),
        ]
        if cat:
            parts.append(("cat", OneHotEncoder(categories=[MODULES], handle_unknown="ignore", sparse_output=False), cat))
        prep = ColumnTransformer(parts, remainder="drop")
        model = LogisticRegression(C=params["C"], class_weight=params["class_weight"], max_iter=5000)
    elif family == "rf":
        parts = [("num", "passthrough", num)]
        if cat:
            parts.append(("cat", OneHotEncoder(categories=[MODULES], handle_unknown="ignore", sparse_output=False), cat))
        prep = ColumnTransformer(parts, remainder="drop")
        model = RandomForestClassifier(
            n_estimators=500, max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
            class_weight=params["class_weight"], max_features="sqrt", n_jobs=N_JOBS, random_state=seed,
        )
    elif family in ("hgb", "lgbm"):
        parts = [("num", "passthrough", num)]
        if cat:
            parts.append(("cat", OrdinalEncoder(categories=[MODULES], handle_unknown="use_encoded_value", unknown_value=-1), cat))
        prep = ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False).set_output(transform="pandas")
        if family == "hgb":
            model = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=n_iter or params["max_iter"], max_leaf_nodes=params["max_leaf_nodes"],
                min_samples_leaf=20, class_weight=params["class_weight"], early_stopping=False,
                categorical_features=cat or None, random_state=seed,
            )
        else:
            model = LGBMClassifier(
                objective="binary", learning_rate=0.05, n_estimators=n_iter or params["n_estimators"],
                num_leaves=params["num_leaves"], min_child_samples=20, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, scale_pos_weight=params["scale_pos_weight"], random_state=seed,
                deterministic=True, force_row_wise=True, n_jobs=N_JOBS, verbose=-1,
            )
    else:
        raise ValueError(family)
    return Pipeline([("prep", prep), ("model", model)])


def fit_pipeline(pipe: Pipeline, family: str, rows: pd.DataFrame, features: list[str]) -> Pipeline:
    X, y = rows, rows["label"].to_numpy()
    cat = [c for c in features if c in CATEGORICAL]
    if family == "lgbm" and cat:
        pipe.fit(X, y, model__categorical_feature=cat)
    else:
        pipe.fit(X, y)
    if family == "rf":
        # Parallel predict_proba sums tree outputs in thread order (1e-16 jitter that can flip exact
        # ties); fitting is parallel, prediction of the saved model is single-threaded and exact.
        pipe.set_params(model__n_jobs=1)
    return pipe


def predict(pipe: Pipeline, rows: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(rows)[:, 1]


def staged_scores(pipe: Pipeline, family: str, rows: pd.DataFrame, iterations: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Scores after k boosting rounds for every k in `iterations`, from one fitted model."""
    Xt = pipe[:-1].transform(rows)
    model = pipe[-1]
    if family == "lgbm":
        return {k: model.predict_proba(Xt, num_iteration=k)[:, 1] for k in iterations}
    out = {}
    for i, proba in enumerate(model.staged_predict_proba(Xt), start=1):
        if i in iterations:
            out[i] = proba[:, 1]
    return out


@dataclass
class FitContext:
    """Fit/val rows of one (fit_range, val_range, feature_set) setting."""

    ds: Dataset
    fit_range: Range
    val_range: Range
    feature_set: str
    fit_rows: pd.DataFrame = field(init=False)
    val_rows: pd.DataFrame = field(init=False)
    features: list[str] = field(init=False)
    dropped: list[str] = field(init=False)
    es: EvalSet = field(init=False)
    keys: np.ndarray = field(init=False)
    spw: float = field(init=False)

    def __post_init__(self) -> None:
        self.fit_rows = self.ds.slice(self.fit_range)
        self.val_rows = self.ds.slice(self.val_range)
        if self.fit_range[1] > self.val_range[0]:
            raise ValueError(f"fit range {self.fit_range} must end before val range {self.val_range} starts")
        all_features = self.ds.feature_set(self.feature_set)
        self.features, self.dropped = drop_constant(self.fit_rows, all_features)
        assert_non_negative(self.fit_rows, self.features)
        self.es = build_eval_set(self.val_rows, self.ds.positives_in(self.val_range), self.ds.introduced_at)
        self.keys = tie_keys(self.es, N_PERM)
        pos = int(self.fit_rows["label"].sum())
        self.spw = round((len(self.fit_rows) - pos) / pos, 2)

    def score_metrics(self, scores: np.ndarray, n_perm: int = N_PERM) -> tuple[dict, pd.DataFrame]:
        per = evaluate(self.es, scores, n_perm=n_perm, keys=self.keys)
        summ = summarize(per)
        summ.update(classification_diagnostics(self.val_rows["label"].to_numpy(), scores))
        return summ, per


def tune(ds: Dataset, family: str, feature_set: str, fit_range: Range, val_range: Range, seed: int = SEED,
         ctx: FitContext | None = None, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Fit every grid configuration on fit_range, evaluate on val_range; select by mean APFD."""
    ctx = ctx or FitContext(ds, fit_range, val_range, feature_set)
    configs = grid(family, ctx.spw)
    rows = []
    if family in ("hgb", "lgbm"):
        it_key = "max_iter" if family == "hgb" else "n_estimators"
        structures = []
        for c in configs:
            s = {k: v for k, v in c.items() if k != it_key}
            if s not in structures:
                structures.append(s)
        for s in structures:
            t0 = time.perf_counter()
            pipe = fit_pipeline(make_pipeline(family, {**s, it_key: max(ITERATIONS)}, ctx.features, seed), family, ctx.fit_rows, ctx.features)
            fit_s = time.perf_counter() - t0
            for k, scores in staged_scores(pipe, family, ctx.val_rows, ITERATIONS).items():
                summ, _ = ctx.score_metrics(scores)
                rows.append({"params": {**s, it_key: k}, "fit_seconds": fit_s, **summ})
    else:
        for c in configs:
            t0 = time.perf_counter()
            pipe = fit_pipeline(make_pipeline(family, c, ctx.features, seed), family, ctx.fit_rows, ctx.features)
            fit_s = time.perf_counter() - t0
            summ, _ = ctx.score_metrics(predict(pipe, ctx.val_rows))
            rows.append({"params": c, "fit_seconds": fit_s, **summ})
    table = pd.DataFrame(rows)
    table.insert(0, "family", family)
    table.insert(1, "feature_set", feature_set)
    table.insert(2, "fit_range", f"{fit_range[0]}-{fit_range[1] - 1}")
    table.insert(3, "val_range", f"{val_range[0]}-{val_range[1] - 1}")
    table.insert(4, "config", range(len(table)))
    table["params_json"] = table["params"].map(lambda p: json.dumps(p, sort_keys=True))
    # Selection: highest mean APFD; on an exact tie the earlier (simpler) grid entry wins.
    best_i = int(table["apfd"].to_numpy().argmax())
    table["selected"] = False
    table.loc[best_i, "selected"] = True
    best = dict(table.loc[best_i, "params"])
    if verbose:
        print(f"  {family}/{feature_set}: best {best} APFD {table.loc[best_i, 'apfd']:.4f} "
              f"(grid {len(table)}, range {table['apfd'].min():.3f}-{table['apfd'].max():.3f})")
    return table.drop(columns=["params"]), best


def fit_final(ctx: FitContext, family: str, params: dict, seed: int) -> Pipeline:
    return fit_pipeline(make_pipeline(family, params, ctx.features, seed), family, ctx.fit_rows, ctx.features)


def library_versions() -> dict[str, str]:
    import platform

    import joblib
    import lightgbm
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "joblib": joblib.__version__,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
    }
