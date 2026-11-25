"""Stage 8D: figures and Markdown tables for chapter 4, built only from stored result files.

    ml/.venv/bin/python ml/figures_stage8.py  # -> ml/results/stage8/figures/ (PNG 300 dpi + PDF), ml/results/stage8/tables.md

Style of ml/figures.py (same palette, neutral references), Ukrainian axis labels, no titles on the figures
(captions go into the text, DSTU 3008), width ≈ 16 cm. Deterministic: no clock in the output (PDF CreationDate is
removed), the only random part (APFD bootstrap of figure 1 / table 2) uses default_rng(20261117).
Every number of tables.md comes from a result file named under the table; the analogues table is the only one
written by hand (numbers from notes/literature.md only).
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dataset_io import ROOT  # noqa: E402
from figures import GRID, INK, INK_2, NEUTRAL, SLOTS, SURFACE  # noqa: E402

R = ROOT / "ml" / "results" / "stage8"
DRY = ROOT / "ml" / "results" / "stage8-dryrun"
OUT = R / "figures"
WIDTH = 16 / 2.54
N_BOOT, BOOT_SEED = 10_000, 20261117
METHOD_UK = {"oracle": "оракул", "rf": "Random Forest", "lgbm": "LightGBM", "hgb": "HistGradientBoosting",
             "lr": "логістична регресія", "combined": "комбінована евристика", "recent": "недавні невдалі виконання",
             "kp": "Кім–Портер (α = 0,5)", "covers": "перетин змінених файлів", "fail_rate": "історія невдалих виконань",
             "criticality": "критичність", "random": "випадковий порядок", "shortest": "найкоротші першими"}
POLICY_UK = {"threshold": "поріг τ", "budget": "бюджет B", "topk": "top-k"}
SET_UK = {"main": "main (8B)", "hist": "hist", "chg": "chg", "meta": "meta", "main-hist": "main − hist",
          "main-chg": "main − chg", "main-meta": "main − meta", "no_commit_chg": "no_commit_chg", "main+ci": "main + ci"}


# ---------------------------------------------------------------------- formatting


def f(x, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{float(x):,.{d}f}" if d == 0 and abs(float(x)) >= 10_000 else f"{float(x):.{d}f}"
    return s.replace(",", "\u202f").replace(".", ",").replace("-", "−")


def pct(x, d: int = 1) -> str:
    return f(100 * float(x), d) + " %" if x is not None and np.isfinite(float(x)) else "—"


def pm(m, s, d: int = 3, as_pct: bool = False) -> str:
    if as_pct:
        return (f(100 * m, 1) + (" ± " + f(100 * s, 1) if s and s > 0 else "")) + " %"
    return f(m, d) + (" ± " + f(s, d) if s and s > 0 else "")


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def src(*files: str) -> str:
    return "Джерело: " + ", ".join(f"`{x}`" for x in files) + "."


# ---------------------------------------------------------------------- figure helpers


def axes(ncols: int = 1, height_cm: float = 8.0):
    fig, axs = plt.subplots(1, ncols, figsize=(WIDTH, height_cm / 2.54), dpi=300, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axs[0]:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return fig, axs[0]


def legend(ax, loc="lower right", **kw) -> None:
    leg = ax.legend(loc=loc, frameon=False, fontsize=7.5, **kw)
    for t in leg.get_texts():
        t.set_color(INK)


def label(ax, x: str, y: str) -> None:
    ax.set_xlabel(x, color=INK_2, fontsize=8.5)
    ax.set_ylabel(y, color=INK_2, fontsize=8.5)


def save(fig, name: str, rect=(0, 0, 1, 1)) -> list[str]:
    fig.tight_layout(rect=rect)
    files = []
    for ext, kw in (("png", {"metadata": {"Software": None}}), ("pdf", {"metadata": {"CreationDate": None, "Producer": None, "Creator": None}})):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, facecolor=SURFACE, **kw)
        files.append(p.name)
    plt.close(fig)
    return files


def comma_ticks(ax, axis: str = "y", digits: int = 2) -> None:
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.{digits}f}".replace(".", ","))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------- data


def red_apfd() -> tuple[pd.DataFrame, np.ndarray]:
    """APFD per red test commit (columns = methods; RF / LightGBM: mean over seeds) and the commit indices."""
    p = pd.read_csv(R / "main" / "per_commit_test.csv")
    p = p[(p["n_fail"] > 0) & ~p["method"].str.contains("/")]
    t = p.groupby(["commit_index", "method"])["apfd"].mean().unstack("method")
    return t, t.index.to_numpy()


def bootstrap_means(t: pd.DataFrame) -> pd.DataFrame:
    """Percentile 95 % interval of the mean APFD over red commits; one resampling matrix shared by all methods."""
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(t), size=(N_BOOT, len(t)))
    rows = []
    for m in t.columns:
        v = t[m].to_numpy(dtype=float)
        bm = v[idx].mean(1)
        rows.append({"method": m, "apfd": v.mean(), "lo": np.percentile(bm, 2.5), "hi": np.percentile(bm, 97.5)})
    return pd.DataFrame(rows).sort_values("apfd", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------- figures


def fig_apfd(ci: pd.DataFrame) -> list[str]:
    fig, (ax,) = axes(1, 8.5)
    y = np.arange(len(ci))[::-1]
    for yi, r in zip(y, ci.itertuples(index=False)):
        model = r.method in ("rf", "lgbm", "hgb", "lr")
        ref = r.method in ("oracle", "random")
        color = SLOTS[0] if model else (NEUTRAL if ref else SLOTS[3] if r.method == "combined" else INK_2)
        ax.plot([r.lo, r.hi], [yi, yi], color=color, linewidth=1.6, solid_capstyle="butt")
        ax.plot([r.apfd], [yi], "o", color=color, markersize=4.5, markerfacecolor=SURFACE if ref else color)
        ax.text(r.hi + 0.008, yi, f(r.apfd), va="center", fontsize=7, color=INK_2)
    ax.set_yticks(y, [METHOD_UK[m] for m in ci["method"]], fontsize=8, color=INK)
    ax.set_xlim(0.38, 1.0)
    comma_ticks(ax, "x")
    label(ax, "APFD (точка – середнє, відрізок – 95 % бутстреп-інтервал за комітами)", "")
    ax.grid(axis="y", visible=False)
    return save(fig, "fig1_apfd_test")


def fig_cost_curves() -> list[str]:
    mc = pd.read_csv(R / "cost_curve" / "mean_curves.csv")
    series = [("random_B", "випадковий порядок, B", NEUTRAL, "--"), ("kp_B", "Кім–Портер, B", SLOTS[1], "-"),
              ("combined_B", "комбінована евристика, B", SLOTS[3], "-"), ("covers_B", "перетин змінених файлів, B", SLOTS[2], "-"),
              ("rf_B", "Random Forest, B", SLOTS[0], "--"), ("rf_tau", "Random Forest, τ", SLOTS[0], "-"),
              ("oracle_B", "оракул, B", INK, "--")]
    fig, (ax,) = axes(1, 9.0)
    for m, lab, color, ls in series:
        g = mc[mc["method"] == m]
        ax.plot(g["time"] * 100, g["mean"] * 100, color=color, linestyle=ls, linewidth=1.8 if ls == "-" else 1.3, label=lab)
    ax.axvline(30, color=INK_2, linewidth=0.8, linestyle=":", zorder=0)
    ax.axhline(90, color=INK_2, linewidth=0.8, linestyle=":", zorder=0)
    ax.text(31, 3, "30 % часу", color=INK_2, fontsize=7)
    ax.text(1.5, 91.5, "90 % перших появ", color=INK_2, fontsize=7)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    label(ax, "середня частка фактичного часу повного прогону, %", "виявлені перші появи дефектів, %")
    legend(ax)
    return save(fig, "fig2_recall_first_vs_time")


def fig_rolling() -> list[str]:
    pf = pd.read_csv(R / "rolling" / "per_fold.csv")
    pf = pf[(pf["scheme"] == "expanding") & pf["method"].isin(["rf", "combined"])]
    fig, (a1, a2) = axes(2, 7.5)
    for m, color, lab in (("rf", SLOTS[0], "RF + τ"), ("combined", SLOTS[3], "комбінована + B")):
        g = pf[pf["method"] == m].sort_values("fold")
        off = -0.08 if m == "rf" else 0.08
        for ax, v, t in ((a1, "val_recall_first", "test_recall_first"), (a2, "val_time", "test_time")):
            x = g["fold"].to_numpy() + off
            ax.vlines(x, g[v] * 100, g[t] * 100, color=color, linewidth=1.0, alpha=0.7)
            ax.plot(x, g[v] * 100, "o", color=color, markerfacecolor=SURFACE, markersize=4.5, label=f"{lab}: val вікна")
            ax.plot(x, g[t] * 100, "o", color=color, markersize=4.5, label=f"{lab}: test вікна")
    a1.axhline(90, color=INK_2, linewidth=0.8, linestyle=":", zorder=0)
    a2.axhline(30, color=INK_2, linewidth=0.8, linestyle=":", zorder=0)
    for ax in (a1, a2):
        ax.set_xticks(range(5))
    label(a1, "часове вікно (expanding, по 54 коміти)", "виявлені перші появи, %")
    label(a2, "часове вікно (expanding, по 54 коміти)", "фактичний час, %")
    a1.set_ylim(55, 102)
    h, l = a1.get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5)
    for t in leg.get_texts():
        t.set_color(INK)
    return save(fig, "fig3_rolling_transfer", rect=(0, 0.14, 1, 1))


def fig_ablation() -> list[str]:
    ab = pd.read_csv(R / "ablation" / "summary.csv")
    ab = ab[ab["set"].isin(SET_UK)].copy()
    ab["order"] = [list(SET_UK).index(s) for s in ab["set"]]
    ab = ab.sort_values("order")
    fig, (a1, a2) = axes(2, 8.0)
    y = np.arange(len(ab))[::-1]
    for yi, r in zip(y, ab.itertuples(index=False)):
        color = SLOTS[0] if r.set == "main" else (SLOTS[2] if r.set == "no_commit_chg" else INK_2)
        a1.errorbar(r.apfd, yi, xerr=r.apfd_std, fmt="o", color=color, markersize=4.5, elinewidth=1, capsize=0)
    a1.set_yticks(y, [SET_UK[s] for s in ab["set"]], fontsize=8, color=INK)
    a1.grid(axis="y", visible=False)
    comma_ticks(a1, "x")
    label(a1, "APFD на тестовій частині (± стандартне відхилення за запусками)", "")
    a2.fill_between([0, 30], 90, 100, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    a2.text(16, 96.5, "ціль H1", color=INK_2, fontsize=7)
    for r in ab.itertuples(index=False):
        color = SLOTS[0] if r.set == "main" else (SLOTS[2] if r.set == "no_commit_chg" else INK_2)
        a2.plot(r.time * 100, r.recall_first * 100, "o", color=color, markersize=4.5)
        a2.annotate(SET_UK[r.set], (r.time * 100, r.recall_first * 100), xytext=(4, -9) if r.set == "chg" else (4, 2),
                    textcoords="offset points", fontsize=6.5, color=INK_2)
    a2.set_xlim(15, 95)
    a2.set_ylim(55, 100)
    label(a2, "фактичний час за τ, %", "виявлені перші появи за τ, %")
    return save(fig, "fig4_ablation")


def fig_closed_loop() -> list[str]:
    cl = pd.read_csv(R / "closed_loop" / "summary.csv")
    cl["x"] = [0 if l == "open" else {5: 1, 10: 2}.get(int(n), 3) if np.isfinite(n) else 3 for l, n in zip(cl["loop"], cl["n_full"])]
    names = ["відкрита\nпетля", "N = 5", "N = 10", "N = ∞"]
    fig, (a1, a2) = axes(2, 7.5)
    for m, color, lab, off in (("rf", SLOTS[0], "RF + τ", -0.12), ("combined", SLOTS[3], "комбінована + B", 0.12)):
        g = cl[cl["method"] == m].sort_values("x")
        a1.bar(g["x"] + off, g["share_undetected_before_fix"] * 100, width=0.24, color=color, label=lab,
               yerr=g["share_undetected_before_fix_std"].fillna(0) * 100, error_kw={"elinewidth": 0.8, "ecolor": INK_2})
        a2.bar(g["x"] + off, g["delay_mean"], width=0.24, color=color, label=lab)
    for ax in (a1, a2):
        ax.set_xticks(range(4), names, fontsize=7.5)
        ax.grid(axis="x", visible=False)
    label(a1, "повний прогін кожні N комітів", "не виявлено до виправлення, %")
    label(a2, "повний прогін кожні N комітів", "середня затримка, комітів")
    comma_ticks(a2, "y")
    legend(a1, loc="upper left")
    return save(fig, "fig5_closed_loop")


def fig_learning_curve() -> list[str]:
    lc = pd.read_csv(R / "learning_curve" / "summary.csv")
    fig, (a1, a2) = axes(2, 7.5)
    for hist, mk, ls, lab in (("history-600-540", "o", "-", "основна історія"), ("history (300, auxiliary)", "s", "--", "допоміжна історія")):
        g = lc[lc["history"] == hist]
        a1.plot(g["fit_commits"], g["apfd"], marker=mk, linestyle=ls, color=SLOTS[0], markersize=4, linewidth=1.4, label=lab)
        a2.plot(g["fit_commits"], g["recall_first"] * 100, marker=mk, linestyle=ls, color=SLOTS[0], markersize=4, linewidth=1.4,
                label=f"{lab}: спіймані перші появи")
        a2.plot(g["fit_commits"], g["time"] * 100, marker=mk, linestyle=ls, color=SLOTS[1], markersize=4, linewidth=1.4,
                label=f"{lab}: фактичний час")
    a1.set_ylim(0.85, 0.90)
    comma_ticks(a1, "y")
    a2.axhline(90, color=INK_2, linewidth=0.8, linestyle=":", zorder=0)
    a2.axhline(30, color=INK_2, linewidth=0.8, linestyle=":", zorder=0)
    a2.set_ylim(20, 102)
    label(a1, "комітів у навчальній частині", "APFD на тестовій частині")
    label(a2, "комітів у навчальній частині", "частка за τ, %")
    legend(a1, loc="lower right")
    h, l = a2.get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7)
    for t in leg.get_texts():
        t.set_color(INK)
    return save(fig, "fig6_learning_curve", rect=(0, 0.14, 1, 1))


def fig_error_categories() -> list[str]:
    cats = pd.read_csv(R / "errors" / "categories.csv")
    s = json.loads((R / "errors" / "summary.json").read_text(encoding="utf-8"))
    n_m, n_c = s["missed"], s["first_exposures"] - s["missed"]
    fig, (ax,) = axes(1, 6.5)
    y = np.arange(len(cats))[::-1]
    ax.barh(y + 0.18, cats["missed_share"] * 100, height=0.34, color=SLOTS[1], label=f"пропущені перші появи (n = {n_m})")
    ax.barh(y - 0.18, cats["caught_share"] * 100, height=0.34, color=NEUTRAL, label=f"спіймані перші появи (n = {n_c}), контраст")
    for yi, r in zip(y, cats.itertuples(index=False)):
        ax.text(r.missed_share * 100 + 1, yi + 0.18, str(r.missed_n), va="center", fontsize=7, color=INK_2)
        ax.text(r.caught_share * 100 + 1, yi - 0.18, str(r.caught_n), va="center", fontsize=7, color=INK_2)
    ax.set_yticks(y, [f"({'абвгд'['abcde'.index(c)]}) {n}" for c, n in zip(cats["category"], cats["name"])], fontsize=7.5, color=INK)
    ax.set_xlim(0, 110)
    ax.grid(axis="y", visible=False)
    label(ax, "частка перших появ, % (числа — кількість)", "")
    legend(ax, loc="upper right")
    return save(fig, "fig7_error_categories")


# ---------------------------------------------------------------------- tables


def tables(ci: pd.DataFrame) -> str:
    h1 = json.loads((R / "main" / "h1.json").read_text(encoding="utf-8"))
    h2 = json.loads((R / "main" / "h2.json").read_text(encoding="utf-8"))
    h1v = json.loads((DRY / "h1.json").read_text(encoding="utf-8"))
    m, b = h1["main"], h1["main"]["bootstrap_95"]
    parts = ["# Таблиці для розділу 4 (етап 8)", "",
             "> Згенеровано `ml/figures_stage8.py` з файлів результатів; числа не редагувати вручну (крім табл. 10). "
             "Десятковий роздільник — кома. Test — коміти 432–539 (108 комітів, 61 «червоний», 39 перших появ дефектів).", ""]

    parts += ["## Таблиця 1. Вердикти H1 і H2 (попередньо зареєстровані правила)", "", md_table(
        ["Гіпотеза", "Правило", "Результат", "Вердикт"],
        [["H1", "RF + τ: recall перших появ ≥ 0,90 і фактичний час ≤ 30 % (середнє за 10 seed)",
          f"recall {pm(m['recall_first']['mean'], m['recall_first']['std'])} (95 %: {f(b['recall_first']['lo'])}–{f(b['recall_first']['hi'])}); "
          f"час {pm(m['time']['mean'], m['time']['std'], as_pct=True)} (95 %: {pct(b['time']['lo'])}–{pct(b['time']['hi'])})",
          "не виконано" if h1["verdict"] == "not met" else h1["verdict"]],
         ["H2", "APFD RF проти комбінованої евристики: Вілкоксон p < 0,05 і медіана різниць > 0",
          f"{f(h2['mean_model'])} проти {f(h2['mean_reference'])}; середня різниця {f(h2['mean_diff'])}, медіана {f(h2['median_diff'])} "
          f"({h2['n_zero']} нульових з {h2['n']}); p = {f(h2['p'])}; δ = {f(h2['delta'])}",
          "підтверджено" if h2["confirmed"] else "не підтверджено"]]), "", src("ml/results/stage8/main/h1.json", "ml/results/stage8/main/h2.json"), ""]

    tm = pd.read_csv(R / "main" / "test_metrics.csv")
    tm = tm[~tm["method"].str.contains("/")]
    g = tm.groupby("method")
    mean, std = g[["apfd", "apfdc", "apfd_new", "apfd_repeat", "apfd_defect"]].mean(), g["apfd"].std(ddof=1)
    rows = []
    for r in ci.itertuples(index=False):
        mm = mean.loc[r.method]
        s = std.loc[r.method] if g.size()[r.method] > 1 else 0
        rows.append([METHOD_UK[r.method], pm(mm["apfd"], s), f"{f(r.lo)}–{f(r.hi)}", f(mm["apfdc"]), f(mm["apfd_new"]), f(mm["apfd_repeat"]), f(mm["apfd_defect"])])
    parts += ["## Таблиця 2. Пріоритизація на test (APFD по «червоних» комітах; RF і LightGBM — середнє ± std за 10 seed)", "",
              md_table(["Метод", "APFD", "95 % bootstrap за комітами", "APFDc", "APFD нових", "APFD повторних", "APFD дефектів"], rows), "",
              "Bootstrap — 10 000 вибірок 61 «червоного» коміту з поверненням, `default_rng(20261117)`, одна матриця для всіх методів "
              "(для RF і LightGBM — APFD коміту, усереднений за seed); описово, розд. 9 протоколу.", "",
              src("ml/results/stage8/main/test_metrics.csv", "ml/results/stage8/main/per_commit_test.csv"), ""]

    pv = pd.read_csv(DRY / "policies_val.csv")
    pt = pd.read_csv(R / "main" / "policies_test.csv")
    sel = lambda p: p[(p["criterion"] == "first") & np.isclose(p["target"], 0.9)]
    pv, pt = sel(pv), sel(pt)
    plan = [("rf", "threshold"), ("rf", "budget"), ("lgbm", "threshold"), ("hgb", "threshold"), ("lr", "threshold"),
            ("combined", "budget"), ("covers", "budget"), ("kp", "budget"), ("oracle", "budget")]
    rows3, rows4 = [], []
    for meth, pol in plan:
        a = pv[(pv["method"] == meth) & (pv["policy"] == pol)]
        t = pt[(pt["method"] == meth) & (pt["policy"] == pol)]
        rows3.append([METHOD_UK[meth], POLICY_UK[pol], f(a["recall_first"].mean()), pct(a["time"].mean()),
                      f(t["recall_first"].mean()), pct(t["time"].mean()), f(t["recall_first"].mean() - a["recall_first"].mean())])
        setting = pm(t["setting"].mean(), t["setting"].std(ddof=1) if len(t) > 1 else 0, 3)
        rows4.append([METHOD_UK[meth], POLICY_UK[pol], setting, pm(t["recall_first"].mean(), t["recall_first"].std(ddof=1) if len(t) > 1 else 0),
                      pct(t["time"].mean()), pct(t["pred_time"].mean()), pct(t["time_red"].mean()), pct(t["empty_selections"].mean())])
    parts += ["## Таблиця 3. Перенесення налаштувань з val на test (перша поява, ціль 0,90; val — in-sample)", "",
              md_table(["Метод", "Політика", "Recall перших появ, val", "Час, val", "Recall перших появ, test", "Час, test", "Різниця recall"], rows3), "",
              f"Val: перших появ — {h1v['first_exposures']}, «червоних» комітів — {h1v['red_commits']}; налаштування підібрано на тій самій val.", "",
              src("ml/results/stage8-dryrun/policies_val.csv", "ml/results/stage8/main/policies_test.csv"), ""]
    parts += ["## Таблиця 4. Заморожені політики на test (перша поява, ціль 0,90; середнє за seed)", "",
              md_table(["Метод", "Політика", "Налаштування (τ або B)", "Recall перших появ", "Фактичний час", "Прогнозний час", "Час на «червоних»", "Порожні відбори"], rows4), "",
              src("ml/results/stage8/main/policies_test.csv"), ""]

    ab = pd.read_csv(R / "ablation" / "summary.csv")
    rows = []
    for r in ab.itertuples(index=False):
        name = SET_UK.get(r.set, SET_UK.get(r.set.split("@")[0], r.set) + ", `tune()` на val")
        rows.append([name, str(int(r.n_features)), pm(r.apfd, r.apfd_std), f(r.apfd_new), f(r.apfd_repeat), f(r.roc_auc),
                     pm(r.recall_first, r.recall_first_std), pct(r.time), pct(r.val_time)])
    parts += ["## Таблиця 5. Ablation за групами ознак (RF, 10 seed; τ кожного набору — на val)", "",
              md_table(["Набір", "Ознак", "APFD", "APFD нових", "APFD повторних", "ROC-AUC", "Recall перших появ (τ)", "Час (τ)", "Час на val"], rows), "",
              src("ml/results/stage8/ablation/summary.csv"), ""]

    rs = pd.read_csv(R / "rolling" / "summary.csv")
    rs = rs[rs["test_recall_first"].notna()]
    rows = []
    for _, r in rs.iterrows():
        pol = "τ" if r["method"] in ("rf", "lgbm", "hgb", "lr") else "B"
        rows.append([r["scheme"], f"{METHOD_UK[r['method']]} + {pol}", pm(r["apfd"], r["apfd_std"]), pm(r["test_recall_first"], r["test_recall_first_std"]),
                     pm(r["test_time"], r["test_time_std"], as_pct=True), f(r["gap_recall_first"]),
                     f"{int(r['test_recall_first_ge_0.90'])} / {int(r['test_time_le_0.30'])}"])
    parts += ["## Таблиця 6. Rolling-window: перенесення порогу між сусідніми вікнами (середнє ± std за 5 folds)", "",
              md_table(["Схема", "Метод", "APFD", "Recall перших появ (test вікна)", "Фактичний час", "Розрив recall test − val", "Folds з recall ≥ 0,90 / час ≤ 30 %"], rows), "",
              "Евристики від `fit` не залежать, тож для обох схем їхні рядки однакові. Folds 3–4 оцінюють на 432–539, fold 4 підбирає на 432–485.", "",
              src("ml/results/stage8/rolling/summary.csv"), ""]

    cl = pd.read_csv(R / "closed_loop" / "summary.csv")
    comb8b = pt[(pt["method"] == "combined") & (pt["policy"] == "budget")].iloc[0]
    rows = []
    for r in cl.itertuples(index=False):
        loop = ("відкрита (8B)" if r.method == "rf" else "відкрита (перестановка 0)") if r.loop == "open" else f"замкнена, N = {'∞' if not np.isfinite(r.n_full) else int(r.n_full)}"
        rows.append([("RF + τ" if r.method == "rf" else "комбінована + B"), loop, str(int(r.full_run_commits)),
                     pm(r.recall_first_all, r.recall_first_all_std), pct(r.time_all), f"{f(r.recall_first_selected)} / {pct(r.time_selected)}",
                     f"{f(r.delay_mean, 2)} / {int(r.delay_max)}", f"{f(r.undetected_before_fix, 1)} ({pct(r.share_undetected_before_fix)})"])
    parts += ["## Таблиця 7. Замкнена петля відбору (39 дефектів із першою появою в test)", "",
              md_table(["Метод", "Петля", "Повних прогонів", "Recall перших появ", "Час", "Recall / час без повних прогонів", "Затримка: середня / макс.", "Не виявлено до виправлення"], rows), "",
              "Комбінована евристика в петлі — одна реалізація відбору (ключ перестановки 0); у 8B (середнє за 20 перестановками) — "
              f"{f(comb8b['recall_first'])} і {pct(comb8b['time'])}.", "",
              src("ml/results/stage8/closed_loop/summary.csv", "ml/results/stage8/main/policies_test.csv"), ""]

    mb = pd.read_csv(R / "manual_bugs" / "manual_bugs.csv")
    rows = []
    for r in mb.itertuples(index=False):
        if not r.detectable:
            rows.append([r.defect_id, str(r.introduced_at), r.module, "—", "не виявний", "—", "—"])
            continue
        rows.append([r.defect_id, str(r.introduced_at), r.module, r.failing_tests.replace(";", ", "),
                     f"{f(r.rf_tau_caught, 1)} / {f(r.rf_tau_rank, 1)} / {pct(r.rf_tau_time, 0)}",
                     f"{f(r.combined_B_caught, 2)} / {f(r.combined_B_rank, 1)} / {pct(r.combined_B_time, 0)}",
                     f"{f(r.covers_B_caught, 2)} / {f(r.covers_B_rank, 1)} / {pct(r.covers_B_time, 0)}"])
    parts += ["## Таблиця 8. Ручні баги в test (описово, n = 10)", "",
              md_table(["Баг", "Коміт", "Модуль", "Тести, що падають", "RF + τ: спіймано / ранг / час", "комбінована + B", "covers + B"], rows), "",
              "Спіймано — частка seed (RF) або перестановок нічиїх (евристики); ранг — позиція першого тесту, що падає через баг; час — частка фактичного часу відбору на коміті.", "",
              src("ml/results/stage8/manual_bugs/manual_bugs.csv"), ""]

    ov = json.loads((R / "overhead" / "summary.json").read_text(encoding="utf-8"))
    rk = ov["rank_ms"]
    rows = [[n, f"{f(rk[k]['median_over_commits_of_median'], 2)} ({f(rk[k]['p95_over_commits_of_p95'], 2)})"] for n, k in
            (("ознаки", "features"), ("прогноз RF (500 дерев)", "score"), ("ранжування й відбір", "rank_select"), ("`rank` загалом", "total"))]
    rows += [["завантаження моделі", f"{f(ov['model_load_ms']['median'], 0)} ({f(ov['model_load_ms']['p95'], 0)})"],
             ["повний прогін, послідовна сума тривалостей (медіана / середнє)", f"{f(ov['full_run_sequential_ms']['median'], 0)} / {f(ov['full_run_sequential_ms']['mean'], 0)}"],
             ["повний прогін на 4 worker'ах (`notes/cli.md`)", f"≈ {f(ov['full_run_4_workers_s'] * 1000, 0)}"]]
    parts += ["## Таблиця 9. Накладні витрати ML-відбору (мс; медіана за комітами з медіан 30 повторів, у дужках p95)", "",
              md_table(["Складова", "Час, мс"], rows), "",
              f"Частка `rank` від послідовного прогону — {pct(ov['rank_share_of_sequential_run_median'], 2)}, від прогону на 4 worker'ах — "
              f"{pct(ov['rank_share_of_4_worker_run'], 2)}. Платформа: {ov['platform']['system']}, {ov['platform']['cpu_count']} логічних ядер, "
              f"Python {ov['platform']['python']}; числа залежать від машини.", "", src("ml/results/stage8/overhead/summary.json"), ""]

    parts += ["## Таблиця 10. Порівняння з аналогами (непряме, крім Кіма–Портера)", "", analogues(), "",
              "Числа аналогів звірено з повними текстами джерел 2026-11-22 (`notes/literature.md`).", "",
              src("notes/literature.md", "thesis/references.md", "ml/results/stage8/main/test_metrics.csv", "ml/results/stage8/cost_curve/readouts.csv"), ""]

    # extra tables of 8D
    cats = pd.read_csv(R / "errors" / "categories.csv")
    es = json.loads((R / "errors" / "summary.json").read_text(encoding="utf-8"))
    rows = [[f"({'абвгд'['abcde'.index(r.category)]}) {r.name}", f"{r.missed_n} ({pct(r.missed_share, 0)})", str(r.missed_commits),
             f"{r.caught_n} ({pct(r.caught_share, 0)})"] for r in cats.itertuples(index=False)]
    parts += ["## Таблиця 11. Категорії пропущених перших появ RF + τ (правила — розд. 9 протоколу, запис 8D)", "",
              md_table(["Категорія", f"Пропущені (n = {es['missed']})", "Різних комітів", f"Спіймані (n = {es['first_exposures'] - es['missed']}), контраст"], rows), "",
              "Одна поява може мати кілька категорій; (д) — жодна з (а)–(г).", "", src("ml/results/stage8/errors/categories.csv", "ml/results/stage8/errors/missed.csv"), ""]
    ms = pd.read_csv(R / "errors" / "missed.csv")
    rows = [[r.defect_id, str(r.commit_index), r.mutator if r.source == "mutant" else "ручний", r.module, f(r.rf_tau_caught, 1),
             f(r.rf_best_rank, 1), f(r.rf_best_score_minus_tau), f(r.tau_selected_tests, 1), "так" if r.tau_empty_share >= 0.5 else "ні",
             f"+{int(r.chg_lines_added)} / −{int(r.chg_lines_deleted)}", f(r.rf_B_caught, 1), f(r.combined_B_caught, 2)] for r in ms.itertuples(index=False)]
    parts += ["## Таблиця 12. Пропущені перші появи RF + τ (поіменно)", "",
              md_table(["Дефект", "Коміт", "Оператор", "Модуль", "Спіймано (частка seed)", "Ранг найкращого тесту", "Оцінка − τ", "Відібрано тестів", "Порожній відбір", "Діф: рядки", "RF + B (те саме ранжування)", "комбінована + B"], rows), "",
              "RF + B — бюджет із картки (перша поява, 0,90), частка seed; комбінована + B — частка перестановок нічиїх.", "",
              src("ml/results/stage8/errors/missed.csv"), ""]
    rd = pd.read_csv(R / "cost_curve" / "readouts.csv")
    rows = []
    for _, d in rd.iterrows():
        cell = lambda c: pm(d[c], d[f"{c}_std"])
        rows.append([d["label"], *[cell(f"recall_at_{t}") for t in (20, 30, 40, 50)], pm(d["min_time_recall_0.90"], d["min_time_recall_0.90_std"], as_pct=True)])
    parts += ["## Таблиця 13. Recall перших появ за однакового фактичного часу (розвідувально; не перевірка H1)", "",
              md_table(["Метод і політика", "20 %", "30 %", "40 %", "50 %", "Найменший час для recall ≥ 0,90"], rows), "",
              "Налаштування розгортаються на тих самих даних test, на яких зчитуються, тож це опис компромісу, а не заморожена політика. "
              "Лінійна інтерполяція між точками кривої; моделі з 10 seed — середнє ± std за seed.", "",
              src("ml/results/stage8/cost_curve/readouts.csv"), ""]
    return "\n".join(parts)


def analogues() -> str:
    """Hand-built table of notes/literature.md numbers; our own numbers (Kim-Porter row, RF misses) read from files."""
    tm = pd.read_csv(R / "main" / "test_metrics.csv")
    kp_apfd = float(tm.loc[tm["method"] == "kp", "apfd"].iloc[0])
    pt = pd.read_csv(R / "main" / "policies_test.csv")
    kp_b = pt[(pt["method"] == "kp") & (pt["policy"] == "budget") & (pt["criterion"] == "first") & np.isclose(pt["target"], 0.9)].iloc[0]
    rf_miss = 1 - json.loads((R / "main" / "h1.json").read_text(encoding="utf-8"))["main"]["recall_first"]["mean"]
    return _ANALOGUES.format(kp_apfd=f(kp_apfd), kp_b=f(kp_b["setting"], 2), kp_rec=f(kp_b["recall_first"]), kp_time=pct(kp_b["time"]),
                             rf_miss=pct(rf_miss, 0))


_ANALOGUES = md_table(
    ["Підхід (джерело)", "Дані", "Метрика", "Результат, як подають автори", "Що порівнюване з нашою роботою", "Що не порівнюване"],
    [["Кім і Портер [9], відтворено в роботі (α = 0,5)", "наша test-частина", "APFD; recall перших появ за B",
      "APFD {kp_apfd}; за B = {kp_b} — recall перших появ {kp_rec} за {kp_time} часу", "пряма точка порівняння: ті самі дані, метрики й протокол", "—"],
     ["Facebook PTS, Machalica et al. [3]", "промислова історія прогонів", "частка спійманих падінь і дефектних змін за вартості",
      "витрати ×0,5 за збереження > 95 % падінь і > 99,9 % дефектних змін", "постановка (вибір за ймовірністю падіння, компроміс recall/вартість)",
      "масштаб, реальні дефекти, flaky; одиниця — зміна, а не перша поява дефекту"],
     ["Ericsson, Najafi et al. [12]", "промислова історія виконань", "час до першого падіння; частка пропусків",
      "пріоритизація за історією (Prio) — медіанний час до першого падіння −60,05 % проти типового порядку; вибір за аналізом вартості (TheoSelect) — −41,78 % часу виконання, але 34,65 % пропущених падінь", "історичні ознаки; ризик пропусків за агресивного вибору (пор. {rf_miss} пропущених перших появ RF + τ)",
      "інші дані й метрики (не APFD)"],
     ["Fallahzadeh et al. [26]", "285 млн результатів тестів (Chrome та ін.)", "економія часу; пропущені падіння",
      "вибір −96 % часу, але до 55 % пропущених падінь; пакетне тестування — до −99 % без пропусків", "той самий компроміс «час ↔ пропуски»",
      "масштаб, інший тип тестів, пакетне тестування в нас не досліджено"],
     ["RETECS, Spieker et al. [20]", "ABB Paint Control (114 тестів, 312 циклів CI), IOF/ROL (2 086 / 320), Google GSDTSR (5 555 / 336)", "NAPFD", "таблиці значень немає (лише рисунки); після ≈ 60 циклів CI — на рівні або краще за детерміновані методи (сортування за останніми вердиктами, зважування); на GSDTSR різниця NAPFD — до 0,2", "ознаки (тривалість, останній запуск, історія падінь); потреба в «розігріві» історії (пор. холодний старт)", "NAPFD ≠ APFD; RL-агент у нас не відтворено (подальша робота)"],
     ["Yaraghi et al. [24]", "25 відкритих Java-проєктів, 21,5 тис. збірок (2,5 тис. невдалих)", "APFDc", "RF на повному наборі ознак — середній APFDc 0,82; найкраща евристика (частота падінь тесту) — 0,71 (p < 0,01); найбільший внесок — ознаки історії виконань; без перенавчання APFDc спадає ≈ на 0,005 за збірку (до ≈ 11 збірок)", "RF кращий за евристику частоти падінь (у нас APFD 0,886 проти 0,651), але не за комбіновану евристику; деградація з часом (пор. rolling)", "інші проєкти й ознаки; APFDc за реальними падіннями збірок"],
     ["Zhao et al. [25]", "11 Java-проєктів, по 800 комітів", "rAPFD (нормований APFD), час навчання й прогнозу", "на проєктах із частими падіннями найкращий ACER-PA (RL), з рідкісними — MART; ефективність змінюється між циклами CI переважно через обсяг навчальних даних; MART економить > 95 % часу тестування на більшості проєктів", "залежність від обсягу навчальних даних (у нас крива навчання пласка вже від 81 коміту)", "rAPFD ≠ APFD; простих евристик автори не порівнювали"],
     ["Bertolino et al. [22]", "6 Java-проєктів Apache Commons (111–627 комітів), реальні падіння (0,004–0,035 % тестів)", "RPA (близькість до ідеального ранжування), час навчання й ранжування; APFD свідомо не використано", "найкраща RPA — у MART і L-MART; середня RPA всіх 10 алгоритмів > 0,8; RL-методи потребують значно більшого часу навчання", "ансамблі дерев — серед найкращих (у нас RF, HGB, LightGBM практично рівні)", "RPA ≠ APFD; одиниця — тестовий клас"],
     ["DeepOrder, Sharif et al. [23]", "Cisco (320 тестів), ABB Paint Control (85), IOF/ROL (1 488), Google GSDTSR (5 507); невдалих виконань 0,0025–0,43 %", "APFD, NAPFD, час", "APFD 0,52 / 0,76 / 0,67 / 0,94, NAPFD 0,52 / 0,52 / 0,56 / 0,79 (Cisco / Paint Control / IOF/ROL / GSDTSR); кращий за RETECS, крім Paint Control; ≈ 0,5 год проти ≈ 25 год у RETECS", "пріоритет за історією виконань і тривалістю", "глибинна модель у нас не відтворена; інші дані"]])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t, _ = red_apfd()
    ci = bootstrap_means(t)
    files = []
    files += fig_apfd(ci)
    files += fig_cost_curves()
    files += fig_rolling()
    files += fig_ablation()
    files += fig_closed_loop()
    files += fig_learning_curve()
    files += fig_error_categories()
    ci.to_csv(OUT / "apfd_bootstrap.csv", index=False, float_format="%.6f")
    (R / "tables.md").write_text(tables(ci), encoding="utf-8")
    (OUT / "README.md").write_text(FIG_README, encoding="utf-8")
    print("written:", ", ".join(files), "+ apfd_bootstrap.csv, README.md, ../tables.md")


FIG_README = """# Рисунки для розділу 4 (етап 8)

Згенеровано `ml/figures_stage8.py`; PNG 300 dpi і PDF, ширина ≈ 16 см, без заголовків (підпис — у тексті).

| Файл | Що показує | Джерело |
|---|---|---|
| `fig1_apfd_test` | APFD за методами на test, точка + 95 % bootstrap-інтервал за 61 «червоним» комітом | `main/per_commit_test.csv` (+ `apfd_bootstrap.csv`) |
| `fig2_recall_first_vs_time` | криві recall перших появ від фактичного часу (розвідувально, не перевірка H1) | `cost_curve/mean_curves.csv` |
| `fig3_rolling_transfer` | перенесення τ (RF) і B (комбінована) між val і test вікон rolling (expanding) | `rolling/per_fold.csv` |
| `fig4_ablation` | ablation: APFD і recall перших появ / час за τ за наборами ознак | `ablation/summary.csv` |
| `fig5_closed_loop` | замкнена петля: частка невиявлених до виправлення дефектів і затримка залежно від N | `closed_loop/summary.csv` |
| `fig6_learning_curve` | learning curve: APFD і recall / час за τ залежно від кількості навчальних комітів | `learning_curve/summary.csv` |
| `fig7_error_categories` | категорії пропущених перших появ (а)–(д) і контраст зі спійманими | `errors/categories.csv` |
"""


if __name__ == "__main__":
    main()
