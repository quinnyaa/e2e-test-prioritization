"""Stage 7: reproducible demo of the CLI for the defense (notes/stage7-plan.md, step 7.6; notes/cli.md).

Base: commit 429 of data/history-600 (the last commit of train+val without open defects); the history
state is commits 0-429, so the demo never reads the locked test part 432-539 (decision R3).
Every scenario is a NEW sibling commit on top of the base in a clone data/demo-stage7/repo (fixed author
and dates, so the SHAs are reproducible). For every scenario:

  1. `cli.py rank` with the main method (RF + tau, the H1 setting) and, for comparison, RF + budget B and
     the heuristics covers ∩ diff and combined (budget B chosen on val by the same procedure);
  2. `cli.py run` of the tests selected by RF + tau (the actual tool);
  3. `cli.py run --policy all` = full run of the suite = ground truth;
  4. which method's selection contains a failing test, and at what share of the actual time.

Defects: 6 manual bugs that never entered data/history-600 (MAN-03, 06, 09, 13, 16, 18). Their patches
no longer apply to commit 429 (benign commits renamed variables and added comments: the anchor drift
that exhausted the pool, notes/simulator.md, p. 12), so they are PORTED by hand: the same semantic change
on the current text. The ported diffs are saved in data/demo-stage7/patches/.

Also measured (input for the stage-8 overhead experiment): CLI time per step, fixed Playwright cost
(1 test) and full-run wall time, 3 repetitions each.

Usage:  ml/.venv/bin/python ml/demo_stage7.py            (~5-10 min, most of it Playwright)
        ml/.venv/bin/python ml/demo_stage7.py --skip-overhead
Output: data/demo-stage7/{report.md, report.json, patches/, runs/}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "data" / "history-600"
OUT = ROOT / "data" / "demo-stage7"
REPO = OUT / "repo"
CLI = [sys.executable, str(ROOT / "ml" / "cli.py")]
BASE_INDEX = 429
AUTHOR = {"GIT_AUTHOR_NAME": "Shop Developer", "GIT_AUTHOR_EMAIL": "dev@demo-shop.test",
          "GIT_COMMITTER_NAME": "Shop Developer", "GIT_COMMITTER_EMAIL": "dev@demo-shop.test"}
DATE = "2027-01-01T10:00:00+00:00"

M = "src/modules"
SCENARIOS = [
    {
        "name": "S1-style-docs", "kind": "доброякісний", "defect": None,
        "title": "Стиль і документація: відступ у шапці сайту, рядок у README",
        "edits": [
            ("src/public/style.css", "padding: 12px 16px; background: #fff;", "padding: 14px 16px; background: #fff;"),
            ("README.md", None, "\nDemo build notes: header spacing adjusted.\n"),  # append
        ],
    },
    {
        "name": "S2-MAN-03-cart", "kind": "ручний баг", "defect": "MAN-03",
        "title": "Кошик більше не перевіряє залишок на складі (лише ліміт 10 на рядок)",
        "edits": [(f"{M}/cart/logic.ts",
                   "  if (total > stock) {\n"
                   "    const noteValue = inCart > 0 ? ` (${inCart} already in your cart)` : '';\n"
                   "    return { ok: false, error: `Only ${stock} item${stock === 1 ? '' : 's'} available${noteValue}` };\n"
                   "  }\n", "")],
    },
    {
        "name": "S3-MAN-09-checkout", "kind": "ручний баг", "defect": "MAN-09",
        "title": "Промокод уже прострочений у момент закінчення терміну (>= замість >)",
        "edits": [(f"{M}/checkout/logic.ts", "now.getTime() > Date.parse(promo.expiresAt)",
                   "now.getTime() >= Date.parse(promo.expiresAt)")],
    },
    {
        "name": "S4-MAN-13-catalog", "kind": "ручний баг", "defect": "MAN-13",
        "title": "Фільтр ціни виключає товари з ціною, рівною максимуму (>= замість >)",
        "edits": [(f"{M}/catalog/logic.ts", "p.priceCents > filters.maxPriceCents) return false;",
                   "p.priceCents >= filters.maxPriceCents) return false;")],
    },
    {
        "name": "S5-MAN-16-orders", "kind": "ручний баг", "defect": "MAN-16",
        "title": "Замовлення з однаковим часом сортуються від старшого id (a.id − b.id)",
        "edits": [(f"{M}/orders/logic.ts", "|| b.id - a.id);", "|| a.id - b.id);")],
    },
    {
        "name": "S6-MAN-18-admin", "kind": "ручний баг", "defect": "MAN-18",
        "title": "Форма адміністратора приймає нульову ціну (< замість <=)",
        "edits": [(f"{M}/admin/logic.ts", "else if (priceCentsLocalLocal <= 0)", "else if (priceCentsLocalLocal < 0)")],
    },
    {
        "name": "S7-MAN-06-search-hang", "kind": "ручний баг (зависання)", "defect": "MAN-06",
        "title": "Ранжування пошуку переписано циклом без інкремента: запит зависає",
        "edits": [(f"{M}/search/logic.ts",
                   "  if (lowerNameAmountAmount.split(' ').some((word) => word.startsWith(lowerQueryLocalAmount))) return 1;\n",
                   "  const words = lowerNameAmountAmount.split(' ');\n"
                   "  for (let i = 0; i < words.length; ) {\n"
                   "    if (words[i].startsWith(lowerQueryLocalAmount)) return 1;\n"
                   "  }\n")],
    },
    {
        "name": "S8-MAN-16-plus-style", "kind": "ручний баг + доброякісні правки", "defect": "MAN-16",
        "title": "Комбінований коміт: баг сортування замовлень разом зі стилем і документацією",
        "edits": [
            (f"{M}/orders/logic.ts", "|| b.id - a.id);", "|| a.id - b.id);"),
            ("src/public/style.css", "padding: 12px 16px; background: #fff;", "padding: 14px 16px; background: #fff;"),
            ("README.md", None, "\nDemo build notes: header spacing adjusted.\n"),
        ],
    },
]

COMPARE = [  # (label, cli args) -- rank only; caught/time are computed from the full run
    ("RF + τ (основна, H1)", ["--model", "rf", "--policy", "threshold"]),
    ("RF + бюджет B", ["--model", "rf", "--policy", "budget"]),
    ("covers ∩ діф + бюджет B", ["--model", "covers", "--policy", "budget"]),
    ("комбінована евристика + бюджет B", ["--model", "combined", "--policy", "budget"]),
]


def git(*args: str, env: dict | None = None) -> str:
    return subprocess.run(["git", "-C", str(REPO), "-c", "core.quotepath=off", *args], check=True,
                          capture_output=True, text=True, env={**os.environ, **(env or {})}).stdout.strip()


def base_sha() -> str:
    runs = [json.loads(line) for line in (HISTORY / "runs.jsonl").read_text().splitlines() if line.strip()]
    run = runs[BASE_INDEX]
    assert run["openAfter"] == [], f"commit {BASE_INDEX} has open defects {run['openAfter']}"
    return run["sha"]


def make_commits(base: str) -> dict[str, str]:
    if REPO.exists():
        shutil.rmtree(REPO)
    REPO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(HISTORY / "repo"), str(REPO)], check=True)
    (REPO / ".git" / "info" / "exclude").write_text("node_modules\ndist\ndata/*.db*\n")
    patches = OUT / "patches"
    patches.mkdir(parents=True, exist_ok=True)
    shas = {}
    for sc in SCENARIOS:
        git("checkout", "--quiet", "--force", "-B", f"demo/{sc['name']}", base)
        for path, find, replace in sc["edits"]:
            file = REPO / path
            text = file.read_text()
            if find is None:
                text = text + replace
            else:
                n = text.count(find)
                if n != 1:
                    raise RuntimeError(f"{sc['name']}: anchor occurs {n} times in {path}")
                text = text.replace(find, replace)
            file.write_text(text)
        git("add", "-A")
        msg = f"demo: {sc['title']}" + (f" ({sc['defect']}, ported)" if sc["defect"] else "")
        git("-c", "commit.gpgsign=false", "commit", "--quiet", "--no-verify", "-m", msg,
            env={**AUTHOR, "GIT_AUTHOR_DATE": DATE, "GIT_COMMITTER_DATE": DATE})
        shas[sc["name"]] = git("rev-parse", "HEAD")
        (patches / f"{sc['name']}.patch").write_text(git("diff", base, shas[sc["name"]]) + "\n")
    git("checkout", "--quiet", "--detach", base)
    return shas


def cli(args: list[str], expect: tuple[int, ...] = (0, 1)) -> tuple[dict, float, int]:
    t0 = time.perf_counter()
    res = subprocess.run(CLI + args + ["--json"], capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if res.returncode not in expect:
        raise RuntimeError(f"cli {' '.join(args[:3])} exited {res.returncode}: {res.stderr[-800:]}")
    return json.loads(res.stdout), wall, res.returncode


def common(head: str, upto: int = BASE_INDEX + 1) -> list[str]:
    return ["--repo", str(REPO), "--head", head, "--history", str(HISTORY), "--upto", str(upto)]


def evaluate_selection(rank: dict, truth: dict) -> dict:
    dur = {t["test_id"]: t["run"]["durationMs"] for t in truth["tests"]}
    failing = set(truth["run"]["confirmed_fail"])
    sel = [t["test_id"] for t in rank["tests"] if t["selected"]]
    first = next((t["rank"] for t in rank["tests"] if t["test_id"] in failing), None)
    return {
        "selected": len(sel), "selected_ids": sel,
        "caught": sorted(failing & set(sel)), "detected": bool(failing & set(sel)) if failing else None,
        "pred_share": rank["selected"]["pred_share"],
        "actual_share": sum(dur[i] for i in sel) / sum(dur.values()),
        "first_failing_rank": first,
    }


def scenario(sc: dict, head: str) -> dict:
    print(f"\n== {sc['name']}: {sc['title']}")
    out = OUT / "runs" / sc["name"]
    if out.exists():
        shutil.rmtree(out)
    ranks = {}
    for label, args in COMPARE:
        r, wall, _ = cli(["rank", *common(head), *args], expect=(0,))
        r["process_wall_s"] = wall
        ranks[label] = r
    sel, sel_wall, sel_code = cli(["run", *common(head), "--out", str(out / "selected")], expect=(0, 1, 3))
    full, full_wall, full_code = cli(["run", *common(head), "--policy", "all", "--out", str(out / "full")], expect=(0, 1, 3))
    truth = full["run"]
    print(f"   full run: confirmed failures {truth['confirmed_fail'] or 'none'}; wall {truth['wall_s']:.1f} s "
          f"(+ re-runs {truth['rerun_s']:.1f} s)")
    methods = {}
    for label, r in ranks.items():
        methods[label] = evaluate_selection(r, full)
        m = methods[label]
        print(f"   {label:34s} selected {m['selected']:2d}/36, pred {100 * m['pred_share']:5.1f} %, "
              f"actual {100 * m['actual_share']:5.1f} %, caught {m['caught'] or '-'}")
    main = ranks[COMPARE[0][0]]
    return {
        "name": sc["name"], "kind": sc["kind"], "defect": sc["defect"], "title": sc["title"], "head": head,
        "diff": main["diff"], "truth": {"confirmed_fail": truth["confirmed_fail"], "flaky": truth["flaky"],
                                        "build_ok": truth["build_ok"], "wall_s": truth["wall_s"],
                                        "rerun_s": truth["rerun_s"], "tests_ms": truth["tests_ms"]},
        "methods": methods,
        "tool_run": {"executed": sel["run"]["executed"], "confirmed_fail": sel["run"]["confirmed_fail"],
                     "wall_s": sel["run"]["wall_s"], "rerun_s": sel["run"]["rerun_s"], "tests_ms": sel["run"]["tests_ms"],
                     "process_wall_s": round(sel_wall, 3), "exit_code": sel_code},
        "full_process_wall_s": round(full_wall, 3),
        "top10": [{k: t[k] for k in ("rank", "test_id", "score", "p", "selected", "reasons")} for t in main["tests"][:10]],
        "cli_timings": main["timings"], "rank_process_wall_s": round(main["process_wall_s"], 3),
    }


def waves_demo(head: str) -> dict:
    """Strict priority order with fail-fast: waves of 3 tests from the RF ranking, stop at the first failure."""
    out = OUT / "runs" / "waves-S4"
    if out.exists():
        shutil.rmtree(out)
    r, wall, code = cli(["run", *common(head), "--policy", "all", "--waves", "3", "--fail-fast", "--out", str(out)],
                        expect=(0, 1, 3))
    run = r["run"]
    return {"scenario": "S4-MAN-13-catalog", "waves_run": len(run["waves"]), "executed": run["executed"],
            "confirmed_fail": run["confirmed_fail"], "wall_s": run["wall_s"], "rerun_s": run["rerun_s"],
            "process_wall_s": round(wall, 3), "exit_code": code,
            "wave_ids": [w["ids"] for w in run["waves"]]}


def overhead(base: str, reps: int = 3) -> dict:
    """Fixed Playwright cost (1 test) vs the full run, on the base commit itself (history 0-428)."""
    one, full, rank_wall, timings = [], [], [], []
    for i in range(reps):
        r, w, _ = cli(["rank", *common(base, BASE_INDEX), "--policy", "threshold"], expect=(0,))
        rank_wall.append(w)
        timings.append(r["timings"])
        r1, _, _ = cli(["run", *common(base, BASE_INDEX), "--policy", "topk", "--value", "1",
                        "--out", str(OUT / "runs" / "overhead" / f"one-{i}")])
        one.append({"wall_s": r1["run"]["wall_s"], "tests_ms": r1["run"]["tests_ms"],
                    "test": r1["selected"]["ids"][0]})
        rf, _, _ = cli(["run", *common(base, BASE_INDEX), "--policy", "all", "--out", str(OUT / "runs" / "overhead" / f"full-{i}")])
        full.append({"wall_s": rf["run"]["wall_s"], "tests_ms": rf["run"]["tests_ms"]})
    med = lambda xs: round(statistics.median(xs), 3)  # noqa: E731
    keys = timings[0].keys()
    return {
        "reps": reps,
        "rank_process_wall_s_median": med(rank_wall),
        "cli_timings_median": {k: med([t[k] for t in timings]) for k in keys},
        "one_test_wall_s_median": med([x["wall_s"] for x in one]), "one_test_ms_median": med([x["tests_ms"] for x in one]),
        "one_test": one[0]["test"],
        "full_wall_s_median": med([x["wall_s"] for x in full]), "full_tests_ms_median": med([x["tests_ms"] for x in full]),
        "raw": {"one": one, "full": full, "rank_wall_s": rank_wall},
    }


# ---------------------------------------------------------------------- report

def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f} %".replace(".", ",")


def fmt_s(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def waves_word(n: int) -> str:
    return "хвилю" if n % 10 == 1 and n % 100 != 11 else ("хвилі" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "хвиль")


def write_report(res: dict) -> None:
    L = []
    L.append("# Демо етапу 7: CLI пріоритизації тестів\n")
    L.append(f"> Згенеровано `ml/demo_stage7.py`. База — коміт {BASE_INDEX} `{res['base'][:10]}` історії "
             f"`data/history-600` (стан історії — коміти 0–{BASE_INDEX}; test-частину 432–539 не використано). "
             "Кожен сценарій — новий коміт поверх бази. Модель — заморожена Random Forest етапу 6 "
             "(seed 20261117, навчена на 0–323). Ручні баги ніколи не потрапляли в історію; їхні патчі портовано "
             "вручну через дрейф якорів (`data/demo-stage7/patches/`). Час — фактичний, з однієї машини; "
             "ранжування й відбір детерміновані, час — ні.\n")
    L.append("## 1. Результати за сценаріями\n")
    L.append("Частка часу — сума фактичних тривалостей відібраних тестів (з повного прогону) / сума всіх 36; "
             "«спіймано» — серед відібраних є тест, що впав у повному прогоні.\n")
    labels = [c[0] for c in COMPARE]
    L.append("| Сценарій | Тип | Падіння в повному прогоні | " + " | ".join(labels) + " |")
    L.append("|---|---|---|" + "---|" * len(labels))
    for s in res["scenarios"]:
        fails = ", ".join(s["truth"]["confirmed_fail"]) or "немає"
        cells = []
        for lab in labels:
            m = s["methods"][lab]
            mark = "—" if m["detected"] is None else ("✔" if m["detected"] else "✘")
            cells.append(f"{mark} {m['selected']} т., {pct(m['actual_share'])}")
        L.append(f"| {s['name']} | {s['kind']} | {fails} | " + " | ".join(cells) + " |")
    det = {lab: [s["methods"][lab]["detected"] for s in res["scenarios"] if s["methods"][lab]["detected"] is not None]
           for lab in labels}
    L.append("")
    L.append("| Метод | Виявлено червоних сценаріїв | Середня частка часу (усі сценарії) |")
    L.append("|---|---|---|")
    for lab in labels:
        shares = [s["methods"][lab]["actual_share"] for s in res["scenarios"]]
        L.append(f"| {lab} | {sum(det[lab])} з {len(det[lab])} | {pct(sum(shares) / len(shares))} |")
    L.append("")
    L.append("## 2. Прогін інструмента (RF + τ) проти повного прогону\n")
    o = res.get("overhead")
    cli_s = o["rank_process_wall_s_median"] if o else statistics.median(s["rank_process_wall_s"] for s in res["scenarios"])
    L.append(f"Наскрізний час інструмента = процес `cli.py rank` ({fmt_s(cli_s)} с, медіана) + wall відібраних + їхні повторні "
             "запуски; повного прогону = wall + повторні запуски. Час `git clone` копії SUT не враховано (одноразовий для коміту).\n")
    L.append("| Сценарій | Відібрано / виконано | Падіння серед відібраних | Wall відібраних, с | Wall повного, с | Повторні запуски (відібр. / повний), с | Наскрізно: інструмент / повний, с |")
    L.append("|---|---|---|---|---|---|---|")
    for s in res["scenarios"]:
        t, f = s["tool_run"], s["truth"]
        tool_total = cli_s + t["wall_s"] + t["rerun_s"]
        full_total = f["wall_s"] + f["rerun_s"]
        L.append(f"| {s['name']} | {s['methods'][labels[0]]['selected']} / {t['executed']} | "
                 f"{', '.join(t['confirmed_fail']) or 'немає'} | {fmt_s(t['wall_s'])} | {fmt_s(f['wall_s'])} | "
                 f"{fmt_s(t['rerun_s'])} / {fmt_s(f['rerun_s'])} | {fmt_s(tool_total)} / {fmt_s(full_total)} |")
    L.append("")
    w = res.get("waves")
    if w:
        L.append("## 3. Строгий порядок: хвилі по 3 тести з fail-fast (S4)\n")
        L.append(f"Виконано {w['waves_run']} {waves_word(w['waves_run'])} ({w['executed']} тестів), падіння: {', '.join(w['confirmed_fail']) or 'немає'}; "
                 f"wall {fmt_s(w['wall_s'])} с + повторні запуски {fmt_s(w['rerun_s'])} с. Хвилі: "
                 + "; ".join(", ".join(x) for x in w["wave_ids"]) + ".\n")
    o = res.get("overhead")
    if o:
        L.append("## 4. Накладні витрати (медіана з 3 повторів, базовий коміт)\n")
        tm = o["cli_timings_median"]
        L.append("| Складова | Час |")
        L.append("|---|---|")
        L.append(f"| процес `cli.py rank` загалом (Python, імпорт, усе нижче) | {fmt_s(o['rank_process_wall_s_median'])} с |")
        L.append(f"| імпорт бібліотек (numpy, pandas, sklearn) | {fmt_s(tm['import_s'], 2)} с |")
        L.append(f"| відтворення стану історії (430 комітів, діфи з кешу) | {fmt_s(tm['history_replay_s'], 2)} с |")
        L.append(f"| діф коміту (git) | {fmt_s(tm['diff_s'], 3)} с |")
        L.append(f"| ознаки | {fmt_s(tm['features_s'], 3)} с |")
        L.append(f"| завантаження моделі | {fmt_s(tm['model_load_s'], 2)} с |")
        L.append(f"| прогноз і відбір | {fmt_s(tm['predict_s'], 3)} с |")
        L.append(f"| Playwright, 1 тест ({o['one_test']}): wall (фіксовані витрати: збирання SUT, браузер, worker) | "
                 f"{fmt_s(o['one_test_wall_s_median'])} с (сам тест {fmt_s(o['one_test_ms_median'] / 1000, 2)} с) |")
        L.append(f"| Playwright, повний прогін 36 тестів, 4 worker'и: wall | {fmt_s(o['full_wall_s_median'])} с "
                 f"(сума тривалостей {fmt_s(o['full_tests_ms_median'] / 1000)} с) |")
        L.append("")
    (OUT / "report.md").write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-overhead", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these scenario names")
    ap.add_argument("--report-only", action="store_true", help="rewrite report.md from report.json")
    args = ap.parse_args()
    if args.report_only:
        write_report(json.loads((OUT / "report.json").read_text()))
        return 0
    t0 = time.time()
    base = base_sha()
    shas = make_commits(base)
    res = {"base": base, "base_index": BASE_INDEX, "history": "data/history-600", "history_commits": BASE_INDEX + 1,
           "model": "ml/models/rf (seed 20261117, train 0-323)", "shas": shas, "scenarios": []}
    for sc in SCENARIOS:
        if args.only and sc["name"] not in args.only:
            continue
        res["scenarios"].append(scenario(sc, shas[sc["name"]]))
    if not args.only or "S4-MAN-13-catalog" in args.only:
        print("\n== waves demo (S4)")
        res["waves"] = waves_demo(shas["S4-MAN-13-catalog"])
        print(f"   {res['waves']}")
    if not args.skip_overhead:
        print("\n== overhead")
        res["overhead"] = overhead(base)
        print(f"   one test {res['overhead']['one_test_wall_s_median']} s, full {res['overhead']['full_wall_s_median']} s")
    res["demo_seconds"] = round(time.time() - t0, 1)
    (OUT / "report.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    write_report(res)
    print(f"\nwrote {OUT.relative_to(ROOT)}/report.md and report.json in {res['demo_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
