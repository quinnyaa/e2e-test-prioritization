"""Stage 4 pilot analysis: kill matrix, detection rates and non-ML prioritization baselines.

Input:  data/pilot/runs.jsonl (mutation/pilot-run.ts), tests/manifest.json (test metadata).
Output: data/pilot/{summary.csv, per_defect.csv, kill_matrix.csv, tables.md}, data/pilot/figures/*.png.

Definitions (one injected defect per run, so the defect is the fault):
- a test detects a defect if it is `confirmedFail` (failed in the run and in both re-runs);
- a defect is detected if at least one test detects it; build failures are excluded;
- APFD  = 1 - p/n + 1/(2n), p = 1-based position of the first detecting test, n = number of tests;
- APFDc = (sum_{j>=p} t_j - t_p/2) / sum_j t_j  (Elbaum et al., one fault, equal severity),
  t_j = actual duration of the j-th test in that run;
- recall@k% tests: share of detected defects whose first detecting test is among the first ceil(k*n) tests;
- recall@k% time: share of detected defects found once the cumulative run time reaches k% of the full run.
Orderings use only a-priori information (manifest durations, criticality, covers), except the oracle.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SEED = 20261021
RANDOM_PERMUTATIONS = 100
BUDGETS = (0.1, 0.2, 0.3)

ROOT = Path(__file__).resolve().parent.parent
RUNS_FILE = ROOT / "data" / "pilot" / "runs.jsonl"
MANIFEST_FILE = ROOT / "tests" / "manifest.json"
META_FILE = ROOT / "data" / "pilot" / "meta.json"
EQUIVALENT_FILE = ROOT / "mutation" / "equivalent.json"
OUT_DIR = ROOT / "data" / "pilot"
FIG_DIR = OUT_DIR / "figures"

MODULES = ["auth", "catalog", "product", "search", "cart", "checkout", "orders", "admin"]
CRIT_RANK = {"high": 0, "medium": 1, "low": 2}

BASELINES = {
    "random": "випадковий (100 перестановок)",
    "shortest": "найкоротші першими",
    "criticality": "за критичністю",
    "covers": "covers-збіг зі зміненим файлом",
    "oracle": "оптимальний (оракул)",
}

# Light-surface palette (reference dataviz palette): three categorical slots + neutrals.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
COLORS = {
    "random": "#8f8e88",
    "shortest": "#2a78d6",
    "criticality": "#eb6834",
    "covers": "#1baf7a",
    "oracle": INK,
}


# ---------- loading ----------


def load_runs() -> tuple[list[dict], list[str]]:
    """Runs of the current pilot defect set; runs of mutants later judged equivalent are dropped (ids returned)."""
    with RUNS_FILE.open(encoding="utf-8") as f:
        runs = [json.loads(line) for line in f if line.strip()]
    current = set(json.loads(META_FILE.read_text(encoding="utf-8"))["defects"])
    equivalent = {f"MUT-{e['id']}" for e in json.loads(EQUIVALENT_FILE.read_text(encoding="utf-8"))}
    dropped = [r["defectId"] for r in runs if r["defectId"] not in current]
    unknown = set(dropped) - equivalent
    if unknown:
        raise ValueError(f"runs.jsonl has defects outside the pilot set that are not marked equivalent: {sorted(unknown)}")
    missing = current - {r["defectId"] for r in runs}
    if missing:
        raise ValueError(f"defects without a run: {sorted(missing)}")
    return [r for r in runs if r["defectId"] in current], dropped


def load_manifest() -> pd.DataFrame:
    tests = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))["tests"]
    df = pd.DataFrame(tests).set_index("id")
    return df[["modules", "criticality", "coversPaths", "avgDurationMs"]]


def layer_of(file: str) -> str:
    if file.endswith(".ejs"):
        return "view"
    return Path(file).stem  # logic | service | routes


# ---------- orderings ----------


def static_orders(manifest: pd.DataFrame) -> dict[str, list[str]]:
    by_duration = sorted(manifest.index, key=lambda t: (manifest.at[t, "avgDurationMs"], t))
    by_crit = sorted(
        manifest.index,
        key=lambda t: (CRIT_RANK[manifest.at[t, "criticality"]], manifest.at[t, "avgDurationMs"], t),
    )
    return {"shortest": by_duration, "criticality": by_crit}


def covers_order(manifest: pd.DataFrame, changed_file: str) -> list[str]:
    def key(t: str) -> tuple:
        return (changed_file not in manifest.at[t, "coversPaths"], manifest.at[t, "avgDurationMs"], t)

    return sorted(manifest.index, key=key)


def oracle_order(tests: list[str], failing: set[str], durations: dict[str, float]) -> list[str]:
    return sorted(tests, key=lambda t: (t not in failing, durations[t], t))


# ---------- metrics ----------


def metrics(order: list[str], failing: set[str], durations: dict[str, float]) -> dict[str, float]:
    n = len(order)
    p = next(i for i, t in enumerate(order, start=1) if t in failing)
    times = np.array([durations[t] for t in order], dtype=float)
    total = times.sum()
    cum_before = times[: p - 1].sum()
    time_to_detect = cum_before + times[p - 1]
    result = {
        "first_pos": p,
        "apfd": 1 - p / n + 1 / (2 * n),
        "apfdc": (times[p - 1 :].sum() - times[p - 1] / 2) / total,
        "time_share": time_to_detect / total,
    }
    for k in BUDGETS:
        result[f"recall_tests_{int(k * 100)}"] = float(p <= math.ceil(k * n))
        result[f"recall_time_{int(k * 100)}"] = float(time_to_detect <= k * total)
    return result


# ---------- analysis ----------


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    runs, dropped = load_runs()
    manifest = load_manifest()
    test_ids = list(manifest.index)

    clean = [r for r in runs if r["type"] == "clean"]
    defects = [r for r in runs if r["type"] != "clean"]
    build_failed = [r for r in defects if not r["buildOk"]]
    defects = [r for r in defects if r["buildOk"]]

    # Kill matrix (defect x test) and per-defect info.
    rows, matrix = [], []
    for r in defects:
        by_id = {t["id"]: t for t in r["tests"]}
        missing = set(test_ids) - set(by_id)
        if missing:
            raise ValueError(f"{r['defectId']}: tests missing from the run: {sorted(missing)}")
        failing = {t for t in test_ids if by_id[t]["confirmedFail"]}
        matrix.append([int(t in failing) for t in test_ids])
        rows.append(
            {
                "defect": r["defectId"],
                "type": r["type"],
                "file": r["file"],
                "module": r["module"],
                "layer": layer_of(r["file"]),
                "mutator": r["mutator"] or "manual",
                "unit_status": r["unitStatus"],
                "detected": bool(failing),
                "n_failing": len(failing),
                "n_flaky": sum(by_id[t]["flaky"] for t in test_ids),
                "n_timed_out": sum(by_id[t]["status"] == "timedOut" for t in test_ids),
                "run_ms": r["totalDurationMs"],
                "wall_ms": r["testWallMs"] + r["rerunWallMs"],
            }
        )
    info = pd.DataFrame(rows).set_index("defect")
    kill = pd.DataFrame(matrix, index=info.index, columns=test_ids)
    kill.to_csv(OUT_DIR / "kill_matrix.csv")

    # Baseline metrics over detected defects.
    rng = np.random.default_rng(SEED)
    fixed = static_orders(manifest)
    per_defect, random_shares = [], []
    for r in defects:
        d = r["defectId"]
        if not info.at[d, "detected"]:
            continue
        failing = {t for t in test_ids if kill.at[d, t]}
        durations = {t["id"]: float(t["durationMs"]) for t in r["tests"]}
        orders = {
            "shortest": fixed["shortest"],
            "criticality": fixed["criticality"],
            "covers": covers_order(manifest, r["file"]),
            "oracle": oracle_order(test_ids, failing, durations),
        }
        for name, order in orders.items():
            per_defect.append({"defect": d, "baseline": name, **metrics(order, failing, durations)})
        perms = [metrics(list(rng.permutation(test_ids)), failing, durations) for _ in range(RANDOM_PERMUTATIONS)]
        random_shares.append([m["time_share"] for m in perms])
        per_defect.append({"defect": d, "baseline": "random", **pd.DataFrame(perms).mean().to_dict()})
    per = pd.DataFrame(per_defect)
    per = per.merge(info[["type", "module", "mutator"]], left_on="defect", right_index=True)
    per.to_csv(OUT_DIR / "per_defect.csv", index=False, float_format="%.4f")

    metric_cols = ["apfd", "apfdc", "time_share"] + [
        f"recall_{kind}_{int(k * 100)}" for kind in ("tests", "time") for k in BUDGETS
    ]
    order_names = list(BASELINES)
    summary = per.groupby("baseline")[metric_cols].mean().reindex(order_names)
    summary.insert(0, "apfd_median", per.groupby("baseline")["apfd"].median().reindex(order_names))
    summary.insert(0, "defects", per.groupby("baseline").size().reindex(order_names))
    summary.to_csv(OUT_DIR / "summary.csv", float_format="%.4f")

    write_tables(info, kill, summary, per, clean, build_failed, dropped, manifest)
    plot_apfd_box(per)
    plot_detection_curve(per, np.array(random_shares))
    plot_kill_heatmap(info, kill, manifest)
    print(summary.round(3).to_string())


# ---------- tables ----------


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def detection_table(info: pd.DataFrame, by: str, label: str) -> str:
    g = info.groupby(by)["detected"].agg(["size", "sum"])
    g["rate"] = g["sum"] / g["size"]
    g = g.rename(columns={"size": "дефектів", "sum": "виявлено", "rate": "частка, %"})
    g["частка, %"] = g["частка, %"].map(pct)
    g.index.name = label
    return g.to_markdown()


def write_tables(info, kill, summary, per, clean, build_failed, dropped, manifest) -> None:
    mutants = info[info["type"] == "mutant"]
    out = ["# Пілот етапу 4: таблиці (згенеровано ml/pilot_analysis.py)", ""]

    total = {
        "дефектів оброблено": len(info),
        "мутантів": int((info["type"] == "mutant").sum()),
        "ручних багів": int((info["type"] == "manual").sum()),
        "не зібралися (виключено)": len(build_failed),
        "виключено після прогону як еквівалентні": ", ".join(dropped) or "—",
        "виявлено": int(info["detected"].sum()),
        "вижило": int((~info["detected"]).sum()),
        "контрольних прогонів без змін": len(clean),
        "хибних падінь у контрольних прогонах": sum(t["status"] != "passed" for r in clean for t in r["tests"]),
        "пар (дефект × тест) із flaky": int(info["n_flaky"].sum()),
        "падінь (дефект × тест), confirmedFail": int(kill.values.sum()),
        "подій (дефект × тест)": int(kill.size),
        "сумарний час прогонів (без виключених), хв": round(
            (info["wall_ms"].sum() + sum(r["testWallMs"] for r in clean)) / 60000, 1
        ),
        "середній повний прогін Playwright, с": round(info["run_ms"].mean() / 1000, 2),
    }
    out += ["## Загальні числа", "", pd.Series(total, name="значення").to_markdown(), ""]

    out += ["## Виявлення за типом дефекту", "", detection_table(info, "type", "тип"), ""]
    out += ["## Виявлення за модулем", "", detection_table(info, "module", "модуль"), ""]
    out += ["## Виявлення за шаром файлу", "", detection_table(info, "layer", "шар"), ""]
    out += ["## Виявлення мутантів за оператором Stryker", "", detection_table(mutants, "mutator", "оператор"), ""]
    out += [
        "## Виявлення мутантів залежно від статусу в unit-тестах",
        "",
        detection_table(mutants, "unit_status", "статус Stryker (Vitest)"),
        "",
    ]

    survived = mutants[~mutants["detected"]]
    surv = survived.groupby(["mutator", "layer"]).size().unstack(fill_value=0)
    out += ["## Мутанти, що вижили: оператор × шар", "", surv.to_markdown(), ""]

    failing_counts = info.loc[info["detected"], "n_failing"]
    out += [
        "## Скільки тестів виявляє один дефект (серед виявлених)",
        "",
        failing_counts.describe().round(2).rename("тестів").to_markdown(),
        "",
    ]

    per_test = kill.sum().rename("виявлено дефектів").to_frame()
    per_test["критичність"] = manifest["criticality"]
    per_test["тривалість, мс"] = manifest["avgDurationMs"].round().astype(int)
    per_test = per_test.sort_values("виявлено дефектів", ascending=False)
    out += ["## Внесок тестів (кількість виявлених дефектів)", "", per_test.to_markdown(), ""]

    t = summary.copy()
    show = pd.DataFrame(index=[BASELINES[b] for b in t.index])
    show["APFD (сер.)"] = t["apfd"].round(3).values
    show["APFD (медіана)"] = t["apfd_median"].round(3).values
    show["APFDc"] = t["apfdc"].round(3).values
    for k in BUDGETS:
        show[f"recall@{int(k * 100)}% тестів"] = t[f"recall_tests_{int(k * 100)}"].map(pct).values
    for k in BUDGETS:
        show[f"recall@{int(k * 100)}% часу"] = t[f"recall_time_{int(k * 100)}"].map(pct).values
    show["час до виявлення, % (сер.)"] = t["time_share"].map(pct).values
    show.index.name = "baseline"
    out += [f"## Baseline-пріоритизація (виявлені дефекти, n = {int(t['defects'].iloc[0])})", "", show.to_markdown(), ""]

    by_type = per.pivot_table(index="baseline", columns="type", values=["apfd", "apfdc"], aggfunc="mean")
    by_type = by_type.reindex(list(BASELINES)).round(3)
    by_type.index = [BASELINES[b] for b in by_type.index]
    by_type.columns = [f"{m.upper()} ({'мутанти' if ty == 'mutant' else 'ручні'})" for m, ty in by_type.columns]
    out += ["## APFD / APFDc окремо для мутантів і ручних багів", "", by_type.to_markdown(), ""]

    (OUT_DIR / "tables.md").write_text("\n".join(out), encoding="utf-8")


# ---------- figures ----------


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def plot_apfd_box(per: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    names = list(BASELINES)
    for ax, metric, title in zip(axes, ("apfd", "apfdc"), ("APFD", "APFDc")):
        style_axes(ax)
        ax.grid(axis="x", visible=False)
        data = [per.loc[per["baseline"] == b, metric].values for b in names]
        box = ax.boxplot(data, widths=0.5, patch_artist=True, showmeans=True,
                         medianprops={"color": INK, "linewidth": 1.5},
                         meanprops={"marker": "D", "markerfacecolor": SURFACE, "markeredgecolor": INK, "markersize": 5},
                         whiskerprops={"color": INK_2}, capprops={"color": INK_2},
                         flierprops={"marker": "o", "markersize": 3, "markerfacecolor": INK_2, "markeredgecolor": "none"})
        for patch, b in zip(box["boxes"], names):
            patch.set_facecolor(COLORS[b])
            patch.set_alpha(0.35 if b in ("random", "oracle") else 0.55)
            patch.set_edgecolor(COLORS[b])
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(["випадковий", "найкоротші", "критичність", "covers", "оракул"], color=INK, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("значення на дефект", color=INK_2)
    fig.text(0.01, 0.01, "◆ — середнє, лінія — медіана; виявлені дефекти пілоту", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG_DIR / "apfd_boxplot.png", facecolor=SURFACE)
    plt.close(fig)


def plot_detection_curve(per: pd.DataFrame, random_shares: np.ndarray) -> None:
    """random_shares: (defects x permutations) time shares; the random curve is their expected CDF."""
    fig, ax = new_figure(7.5, 4.6)
    grid = np.linspace(0, 1, 201)
    for b, label in BASELINES.items():
        if b == "random":
            curve = np.array([(random_shares <= x).mean() for x in grid])
        else:
            shares = np.sort(per.loc[per["baseline"] == b, "time_share"].values)
            curve = np.searchsorted(shares, grid, side="right") / len(shares)
        style = {"linestyle": "--", "linewidth": 1.5} if b in ("random", "oracle") else {"linewidth": 2}
        ax.plot(grid * 100, curve * 100, color=COLORS[b], label=label, **style)
    for k in BUDGETS:
        ax.axvline(k * 100, color=GRID, linewidth=1, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel("частка часу повного прогону, %", color=INK_2)
    ax.set_ylabel("виявлені дефекти, %", color=INK_2)
    ax.set_title("Крива виявлення дефектів залежно від витраченого часу", color=INK, fontsize=11, loc="left")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "detection_curve.png", facecolor=SURFACE)
    plt.close(fig)


def plot_kill_heatmap(info: pd.DataFrame, kill: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Share of the defects of each module (rows) detected by each test (columns)."""
    share = kill.groupby(info["module"]).mean().reindex(MODULES)
    counts = info.groupby("module").size().reindex(MODULES).fillna(0).astype(int)
    prefixes = ["AUTH", "CAT", "PRD", "SRCH", "CART", "CHK", "ORD", "ADM", "E2E"]  # same order as MODULES
    cols = sorted(kill.columns, key=lambda t: (prefixes.index(t.split("-")[0]), t))
    share = share[cols]
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f3f6fb", "#2a78d6", "#0d3a73"])
    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(share.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7, color=INK)
    ax.set_yticks(range(len(MODULES)))
    ax.set_yticklabels([f"{m} (n={counts[m]})" for m in MODULES], fontsize=8, color=INK)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODULES)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Частка дефектів модуля (рядок), які виявляє тест (стовпець)", color=INK, fontsize=11, loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=INK_2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "kill_heatmap.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
