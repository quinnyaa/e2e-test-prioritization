"""Stage 7: ML-based selection and prioritization of the Playwright E2E tests for one commit.

  rank    diff (base..head of a git repository of the SUT) -> features -> ranked list + selection
  run     rank, then run the selected tests with Playwright (+ normalize + rerun-failed) on a checkout of head
  record  append a FULL run (from `run --policy all`) to an extra-runs file that later `rank` calls replay

History features need the results of the previous commits: `--history` is a history directory of the
simulator (runs.jsonl + repo/), `--upto N` keeps its first N commits, `--extra-runs` adds full runs
recorded with `record`. The base commit must be the last commit of that history (checked). Only full
runs update the history (selection feedback, notes/stage7-plan.md, R4).

Stage-8 lock: the commits 432-539 of data/history-600 are the hold-out test part. Using them even as
history input needs `--allow-test-history` (notes/stage7-plan.md, R3).

Examples:
  ml/.venv/bin/python ml/cli.py rank --repo data/demo-stage7/repo --base <sha> --head <sha> \\
      --history data/history-600 --upto 430
  ml/.venv/bin/python ml/cli.py run  ... --policy threshold --out data/demo-stage7/runs/<name>
  ml/.venv/bin/python ml/cli.py run  ... --policy all --out <dir>        # full run (ground truth)
  ml/.venv/bin/python ml/cli.py record --from <dir>/result.json --to <extra_runs.jsonl>

Exit codes: 0 ok (rank) / no confirmed failure (run); 1 confirmed failure among the run tests;
2 usage or history error; 3 the head commit does not build (from tests/global-setup.ts).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))

T_IMPORT = time.perf_counter()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_dataset import HistoryError, _git  # noqa: E402
from features_online import ROOT, diff_between, load_extra_runs, load_runs, replay  # noqa: E402
from prioritize import (  # noqa: E402
    CRITERIA, DEFAULT_MODEL, HEURISTICS, POLICIES, TARGETS, Policy, PolicyError, load_method, prioritize, resolve_policy,
)

T_IMPORTED = time.perf_counter()
TESTS_DIR = ROOT / "tests"
SUT_SRC = ROOT / "sut"
MAIN_HISTORY = ROOT / "data" / "history-600"
TEST_PART_START = 432  # data/dataset/history-600-540/splits.json -> holdout.test
CACHE = ROOT / "data" / "cache"
BUILD_FAILED_EXIT_CODE = 3
RESULT_FILES = ("run.json", "raw.json", "rerun-raw.json", "build-status.json")


# ---------------------------------------------------------------------- rank

def resolve_upto(args, runs_all: list[dict]) -> int:
    if args.upto is not None:
        return args.upto
    by_sha = {r["sha"]: r["index"] for r in runs_all}
    if args.base in by_sha:
        return by_sha[args.base] + 1
    raise HistoryError("--base is not a commit of --history; pass --upto N (and --extra-runs) explicitly")


def guard_test_part(history: Path, upto: int, allow: bool) -> None:
    if history.resolve() == MAIN_HISTORY.resolve() and upto > TEST_PART_START and not allow:
        raise HistoryError(
            f"--upto {upto} would read results of commits {TEST_PART_START}..{upto - 1} of data/history-600 "
            "(the hold-out test part, locked until stage 8); pass --allow-test-history to do it on purpose"
        )


def rev(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()


def do_rank(args) -> dict:
    t = {"import_s": T_IMPORTED - T_IMPORT}
    repo = Path(args.repo).resolve()
    head = rev(repo, args.head)
    base = rev(repo, args.base or f"{head}~1")
    history = Path(args.history).resolve()

    t0 = time.perf_counter()
    runs_all = load_runs(history)
    args.base = base
    upto = resolve_upto(args, runs_all)
    guard_test_part(history, upto, args.allow_test_history)
    extra = load_extra_runs(Path(args.extra_runs) if args.extra_runs else None)
    state, _ = replay(history, upto, extra)
    t["history_replay_s"] = time.perf_counter() - t0
    if state.last_sha != base:
        raise HistoryError(f"base {base[:10]} is not the last commit of the history state ({str(state.last_sha)[:10]}): "
                           "rank needs the history up to and including the parent of head")

    t0 = time.perf_counter()
    diff = diff_between(repo, base, head)
    t["diff_s"] = time.perf_counter() - t0
    commit_index = state.commits_seen
    t0 = time.perf_counter()
    rows = state.features(diff, commit_index, head)
    t["features_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    method = load_method(args.model, args.seed)
    t["model_load_s"] = time.perf_counter() - t0
    policy = resolve_policy(method, args.policy, args.criterion, args.target, args.value)
    t0 = time.perf_counter()
    table = prioritize(method, rows, policy)
    t["predict_s"] = time.perf_counter() - t0

    pred_total = float(table["pred_ms"].sum())
    sel = table[table["selected"]]
    return {
        "repo": str(repo), "base": base, "head": head, "commit_index": commit_index,
        "history": str(history), "history_commits": upto, "extra_runs": len(extra),
        "method": method.label, "model_dir": None if method.kind == "heuristic" else f"ml/models/{method.name}",
        "train_range": method.card.get("train_range"), "production_model": method.production,
        "policy": {"name": policy.name, "setting": policy.setting, "source": policy.source, "text": policy.describe()},
        "diff": {"files": len(diff), "lines_added": sum(r["added"] for r in diff.values()),
                 "lines_deleted": sum(r["deleted"] for r in diff.values()), "paths": sorted(diff)},
        "selected": {"tests": int(len(sel)), "of": int(len(table)),
                     "pred_ms": float(sel["pred_ms"].sum()), "pred_full_ms": pred_total,
                     "pred_share": float(sel["pred_ms"].sum() / pred_total) if pred_total else 0.0,
                     "ids": sel["test_id"].tolist()},
        "tests": [
            {"rank": int(r.rank), "test_id": r.test_id, "module": r.module, "score": float(r.score),
             "p": None if pd.isna(r.p) else float(r.p), "pred_ms": float(r.pred_ms), "selected": bool(r.selected),
             "reasons": r.reasons}
            for r in table.itertuples(index=False)
        ],
        "timings": {k: round(v, 4) for k, v in t.items()},
    }


def print_rank(res: dict, top: int | None = None) -> None:
    s = res["selected"]
    print(f"commit {res['head'][:10]} (index {res['commit_index']}), base {res['base'][:10]}; "
          f"history: {res['history_commits']} commits + {res['extra_runs']} recorded")
    d = res["diff"]
    print(f"diff: {d['files']} file(s), +{d['lines_added']} −{d['lines_deleted']}: {', '.join(d['paths'][:6])}"
          f"{' …' if len(d['paths']) > 6 else ''}")
    print(f"method: {res['method']}; policy: {res['policy']['text']}")
    print(f"selected {s['tests']} of {s['of']} tests, predicted {s['pred_ms'] / 1000:.1f} of {s['pred_full_ms'] / 1000:.1f} s "
          f"({100 * s['pred_share']:.1f} %)\n")
    has_p = any(t["p"] is not None for t in res["tests"])
    head = f"{'#':>3} {'sel':3} {'test':8} {'module':9} {'score':>7} " + (f"{'p':>6} " if has_p else "") + f"{'pred,s':>6}  reasons"
    print(head)
    print("-" * len(head))
    for t in res["tests"][: top or None]:
        p = f"{t['p']:6.3f} " if has_p and t["p"] is not None else ""
        print(f"{t['rank']:3d} {'●' if t['selected'] else ' ':3} {t['test_id']:8} {t['module']:9} {t['score']:7.4f} "
              f"{p}{t['pred_ms'] / 1000:6.2f}  {t['reasons']}")
    tm = res["timings"]
    print(f"\ntimings: history {tm['history_replay_s']:.2f} s, diff {tm['diff_s']:.3f} s, features {tm['features_s']:.3f} s, "
          f"model load {tm['model_load_s']:.2f} s, predict {tm['predict_s']:.3f} s")


# ---------------------------------------------------------------------- run

def checkout(repo: Path, head: str, sut_dir: Path | None) -> Path:
    """A working copy of `head` with the SUT node_modules (cached by SHA under data/cache/sut/)."""
    if sut_dir is not None:
        actual = _git(sut_dir, "rev-parse", "HEAD").strip()
        if actual != head:
            raise HistoryError(f"--sut-dir is at {actual[:10]}, not at head {head[:10]}")
        return sut_dir
    work = CACHE / "sut" / head[:12]
    if not (work / ".git").exists():
        work.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout", str(repo), str(work)], check=True)
        subprocess.run(["git", "-C", str(work), "checkout", "--quiet", "--detach", head], check=True)
        (work / ".git" / "info" / "exclude").write_text("node_modules\ndist\ndata/*.db*\n")
    link = work / "node_modules"
    if not link.exists():
        link.symlink_to(SUT_SRC / "node_modules", target_is_directory=True)
    status = _git(work, "status", "--porcelain").strip()
    if status:
        raise HistoryError(f"checkout {work} is dirty: {status[:200]}")
    return work


def grep_for(ids: list[str]) -> str:
    return "|".join(f"\\[{i}\\]" for i in ids)  # tests/scripts/lib.ts::grepForIds


def run_wave(ids: list[str], sut: Path, out: Path, name: str, workers: int, fail_fast: bool) -> dict:
    raw = out / f"{name}-raw.json"
    env = {**os.environ, "SUT_DIR": str(sut), "RAW_REPORT": str(raw), "PW_REPORTER": "null",
           "WORKERS": str(workers), "RUN_ID": name}
    cmd = ["npx", "playwright", "test", "--grep", grep_for(ids)] + (["-x"] if fail_fast else [])
    t0 = time.perf_counter()
    pw = subprocess.run(cmd, cwd=TESTS_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    wall = time.perf_counter() - t0
    if pw.returncode == BUILD_FAILED_EXIT_CODE:
        return {"name": name, "ids": ids, "build_ok": False, "wall_s": wall, "rerun_s": 0.0, "tests": []}
    subprocess.run(["npm", "run", "--silent", "normalize"], cwd=TESTS_DIR, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # warns about the tests not selected
    rerun = 0.0
    if pw.returncode == 1:
        t1 = time.perf_counter()
        subprocess.run(["npm", "run", "--silent", "rerun-failed"], cwd=TESTS_DIR, env=env, check=True, stdout=subprocess.DEVNULL)
        rerun = time.perf_counter() - t1
    elif pw.returncode != 0:
        raise RuntimeError(f"playwright exited with {pw.returncode}: {pw.stderr[-500:]}")
    run = json.loads((TESTS_DIR / "results" / "run.json").read_text())
    shutil.copy(TESTS_DIR / "results" / "run.json", out / f"{name}-run.json")
    return {"name": name, "ids": ids, "build_ok": run["buildOk"], "wall_s": wall, "rerun_s": rerun,
            "tests": [t for t in run["tests"] if t["status"] != "skipped"]}


def do_run(args) -> tuple[dict, int]:
    res = do_rank(args)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(res["repo"])
    sut = checkout(repo, res["head"], Path(args.sut_dir).resolve() if args.sut_dir else None)
    ranked = [t["test_id"] for t in res["tests"] if t["selected"]]
    if args.waves and args.waves > 0:
        waves = [ranked[i:i + args.waves] for i in range(0, len(ranked), args.waves)]
    else:
        waves = [ranked] if ranked else []

    backup = CACHE / "results-backup"
    backup.mkdir(parents=True, exist_ok=True)
    saved = [f for f in RESULT_FILES if (TESTS_DIR / "results" / f).exists()]
    for f in saved:  # the Playwright helper scripts write fixed paths under tests/results/: keep the user's files
        shutil.copy(TESTS_DIR / "results" / f, backup / f)
    done: list[dict] = []
    build_ok = True
    try:
        for i, ids in enumerate(waves):
            w = run_wave(ids, sut, out, f"wave{i + 1}", args.workers, args.fail_fast)
            done.append(w)
            if not w["build_ok"]:
                build_ok = False
                break
            if args.fail_fast and any(t["confirmedFail"] for t in w["tests"]):
                break
    finally:
        for f in RESULT_FILES:
            if f in saved:
                shutil.copy(backup / f, TESTS_DIR / "results" / f)
            else:
                (TESTS_DIR / "results" / f).unlink(missing_ok=True)

    results = {t["id"]: t for w in done for t in w["tests"]}
    for t in res["tests"]:
        r = results.get(t["test_id"])
        t["run"] = None if r is None else {k: r[k] for k in ("status", "durationMs", "flaky", "confirmedFail", "errorSnippet")}
    failed = [t["test_id"] for t in res["tests"] if t["run"] and t["run"]["confirmedFail"]]
    executed = [t for t in res["tests"] if t["run"]]
    res["run"] = {
        "sut_dir": str(sut), "build_ok": build_ok, "waves": [{"ids": w["ids"], "wall_s": round(w["wall_s"], 3),
                                                              "rerun_s": round(w["rerun_s"], 3)} for w in done],
        "executed": len(executed), "skipped_by_fail_fast": len(ranked) - len(executed) if build_ok else None,
        "confirmed_fail": failed, "flaky": [t["test_id"] for t in executed if t["run"]["flaky"]],
        "tests_ms": float(sum(t["run"]["durationMs"] for t in executed)),
        "wall_s": round(sum(w["wall_s"] for w in done), 3), "rerun_s": round(sum(w["rerun_s"] for w in done), 3),
        "full_run": len(executed) == len(res["tests"]) and build_ok,
        "workers": args.workers, "fail_fast": args.fail_fast,
    }
    (out / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    code = BUILD_FAILED_EXIT_CODE if not build_ok else (1 if failed else 0)
    return res, code


def print_run(res: dict) -> None:
    r = res["run"]
    if not r["build_ok"]:
        print("\nBUILD FAILED: the head commit does not build (exit code 3)")
        return
    print(f"\nPlaywright: {r['executed']} test(s) in {len(r['waves'])} invocation(s), wall {r['wall_s']:.1f} s "
          f"(+ re-runs {r['rerun_s']:.1f} s), sum of test durations {r['tests_ms'] / 1000:.1f} s")
    print(f"confirmed failures: {', '.join(r['confirmed_fail']) or 'none'}"
          + (f"; flaky: {', '.join(r['flaky'])}" if r["flaky"] else ""))


# ---------------------------------------------------------------------- record

def do_record(args) -> int:
    res = json.loads(Path(args.from_).read_text())
    if not res.get("run", {}).get("full_run"):
        raise HistoryError("only a full run (`run --policy all`, no fail-fast) may update the history "
                           "(selection feedback, notes/stage7-plan.md, R4)")
    repo = Path(res["repo"])
    rec = {
        "index": res["commit_index"], "sha": res["head"], "parentSha": res["base"], "buildOk": res["run"]["build_ok"],
        "diff": diff_between(repo, res["base"], res["head"]),
        "tests": [{"id": t["test_id"], **{k: t["run"][k] for k in ("status", "durationMs", "flaky", "confirmedFail")}}
                  for t in res["tests"]],
        "recordedFrom": str(Path(args.from_).resolve()),
    }
    to = Path(args.to)
    existing = load_extra_runs(to)
    if existing and existing[-1]["sha"] != rec["parentSha"]:
        raise HistoryError(f"{to}: the last recorded commit {existing[-1]['sha'][:10]} is not the parent of {rec['sha'][:10]}")
    to.parent.mkdir(parents=True, exist_ok=True)
    with to.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"recorded full run of {rec['sha'][:10]} (index {rec['index']}) -> {to}")
    return 0


# ---------------------------------------------------------------------- main

def add_rank_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", required=True, help="git repository of the SUT that contains base and head")
    p.add_argument("--head", default="HEAD", help="commit to prioritize (default HEAD)")
    p.add_argument("--base", help="parent commit (default head~1); must be the last commit of the history state")
    p.add_argument("--history", default=str(MAIN_HISTORY), help="history directory (runs.jsonl + repo/)")
    p.add_argument("--upto", type=int, help="use the first N commits of --history (default: up to --base)")
    p.add_argument("--extra-runs", help="JSONL file of full runs recorded with `record`")
    p.add_argument("--allow-test-history", action="store_true", help="allow results of the locked commits 432+")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"ml/models/<name> or a heuristic {sorted(HEURISTICS)}")
    p.add_argument("--seed", type=int, help="model seed (default: the card's first seed, 20261117)")
    p.add_argument("--policy", default="threshold", choices=POLICIES)
    p.add_argument("--criterion", default="first", choices=CRITERIA, help="recall criterion the setting was tuned for")
    p.add_argument("--target", default="0.90", choices=TARGETS)
    p.add_argument("--value", type=float, help="explicit setting: tau, B (share) or k (number of tests)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_rank = sub.add_parser("rank", help="ranked list and selection for one commit")
    add_rank_args(p_rank)
    p_rank.add_argument("--top", type=int, help="print only the first N rows")
    p_run = sub.add_parser("run", help="rank + run the selected tests with Playwright")
    add_rank_args(p_run)
    p_run.add_argument("--out", required=True, help="output directory (result.json, Playwright reports)")
    p_run.add_argument("--sut-dir", help="existing working copy of head (default: clone to data/cache/sut/<sha>)")
    p_run.add_argument("--workers", type=int, default=4)
    p_run.add_argument("--fail-fast", action="store_true", help="stop at the first failure (-x; stops later waves)")
    p_run.add_argument("--waves", type=int, default=0,
                       help="run the ranking in invocations of N tests (strict priority order between waves); "
                            "0 = one invocation (Playwright orders tests by file inside it)")
    p_rec = sub.add_parser("record", help="append a full run to the extra-runs history file")
    p_rec.add_argument("--from", dest="from_", required=True, help="result.json of `run --policy all`")
    p_rec.add_argument("--to", required=True, help="extra-runs JSONL file")
    args = ap.parse_args()
    try:
        if args.cmd == "rank":
            res = do_rank(args)
            print(json.dumps(res, ensure_ascii=False, indent=1)) if args.json else print_rank(res, args.top)
            return 0
        if args.cmd == "run":
            res, code = do_run(args)
            if args.json:
                print(json.dumps(res, ensure_ascii=False, indent=1))
            else:
                print_rank(res)
                print_run(res)
            return code
        return do_record(args)
    except (HistoryError, PolicyError, FileNotFoundError, subprocess.CalledProcessError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
