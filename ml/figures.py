"""Stage 6 figures (static PNG for the thesis). Palette: reference categorical slots 1-4 (validated
with the dataviz validator: CVD and normal-vision separation pass; contrast WARN is relieved by the
legend and by the numbers in tables.md); random and oracle are neutral dashed references."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from metrics import N_PERM, N_PERM_RANDOM, EvalSet, has_ties, order_from, perm_metrics  # noqa: E402
from policy import budget_curve, threshold_curve, topk_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#8f8e88"
FAMILY_UK = {"lr": "логістична регресія", "rf": "Random Forest", "hgb": "HistGradientBoosting", "lgbm": "LightGBM"}


def _axes(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _legend(ax, loc="lower right") -> None:
    leg = ax.legend(loc=loc, frameon=False, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(INK)


def _ttd_shares(es: EvalSet, scores: np.ndarray, keys: np.ndarray, n_perm: int) -> np.ndarray:
    scores = es.matrix(scores)
    p = n_perm if has_ties(scores) else 1
    red = es.y.any(1)
    return np.concatenate([perm_metrics(es, order_from(scores, keys[i]))["ttd_share"][red] for i in range(p)])


def detection_curve(path: Path, es: EvalSet, keys: np.ndarray, scores: dict, main_family: str) -> None:
    """Share of red val commits whose first failed test has finished vs the actual share of run time."""
    fig, ax = _axes(7.5, 4.6)
    grid = np.linspace(0, 1, 401)
    series = [
        ("random", "випадковий порядок", NEUTRAL, "--", N_PERM_RANDOM),
        ("covers", "covers ∩ діф", SLOTS[2], "-", N_PERM),
        ("recent", "недавні падіння", SLOTS[1], "-", N_PERM),
        ("combined", "комбінована евристика", SLOTS[3], "-", N_PERM),
        (main_family, f"{FAMILY_UK[main_family]} (модель)", SLOTS[0], "-", N_PERM),
        ("oracle", "оракул", INK, "--", N_PERM),
    ]
    for method, label, color, ls, n_perm in series:
        shares = np.sort(_ttd_shares(es, scores[(method, scores_seed(scores, method))], keys, n_perm))
        curve = np.searchsorted(shares, grid, side="right") / len(shares)
        ax.plot(grid * 100, curve * 100, color=color, linestyle=ls, linewidth=2 if ls == "-" else 1.5, label=label)
    ax.axvline(30, color=INK_2, linewidth=0.8, linestyle=":", zorder=0)
    ax.axhline(90, color=INK_2, linewidth=0.8, linestyle=":", zorder=0)
    ax.text(31, 4, "30 % часу", color=INK_2, fontsize=8)
    ax.text(74, 85.5, "90 % комітів (H1)", color=INK_2, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel("частка фактичного часу повного прогону, %", color=INK_2)
    ax.set_ylabel("«червоні» коміти з виявленим падінням, %", color=INK_2)
    ax.set_title("Val (коміти 324–431): час до першого тесту, що впав", color=INK, fontsize=11, loc="left")
    _legend(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def scores_seed(scores: dict, method: str) -> int:
    return min(s for (m, s) in scores if m == method)


def policy_tradeoff(path: Path, es: EvalSet, keys: np.ndarray, scores: dict, main_family: str) -> None:
    """Commit-level recall vs mean actual time share of the three policies (main model, seed 0)
    and of the combined heuristic (top-k and budget only)."""
    fig, ax = _axes(7.5, 4.6)
    sc = scores[(main_family, scores_seed(scores, main_family))]
    heur = scores[("combined", -1)]
    curves = [
        (topk_curve(es, sc, keys), f"модель: top-k % тестів", SLOTS[0], "-"),
        (budget_curve(es, sc, keys), "модель: бюджет часу", SLOTS[1], "-"),
        (threshold_curve(es, sc), "модель: адаптивний поріг τ", SLOTS[2], "-"),
        (budget_curve(es, heur, keys), "евристика: бюджет часу", SLOTS[3], "--"),
    ]
    for df, label, color, ls in curves:
        df = df.sort_values("time")
        ax.plot(df["time"] * 100, df["recall_commit"] * 100, color=color, linestyle=ls, linewidth=2, label=label)
    ax.axvline(30, color=INK_2, linewidth=0.8, linestyle=":", zorder=0)
    ax.axhline(90, color=INK_2, linewidth=0.8, linestyle=":", zorder=0)
    ax.fill_between([0, 30], 90, 102, color=GRID, alpha=0.5, zorder=0, linewidth=0)
    ax.text(2, 96, "ціль H1", color=INK_2, fontsize=8)
    ax.set_xlim(0, 60)
    ax.set_ylim(40, 102)
    ax.set_xlabel("середня частка фактичного часу повного прогону, %", color=INK_2)
    ax.set_ylabel("виявлені «червоні» коміти, %", color=INK_2)
    ax.set_title(f"Val: політики відбору ({FAMILY_UK[main_family]}, seed {scores_seed(scores, main_family)})", color=INK, fontsize=11, loc="left")
    _legend(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
