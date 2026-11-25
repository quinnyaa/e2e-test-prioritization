# Mutation: defects for the E2E suite

Generates defects of the Demo Shop SUT (`../sut`) and runs the frozen Playwright suite (`../tests`) on each.
The `sut/` repository is never modified: every change is applied to a copy.

## Reproduce the stage 4 pilot

```bash
npm ci
npm run stryker      # Stryker 10.0.0 + Vitest runner → reports/mutation.json, mutants.json (≈30 s)
npm run sample       # 100 mutants, seed 20261021, stratified by module → pilot/sample.json, pilot/excluded.json
npm run pilot        # one full E2E run per defect → ../data/pilot/runs.jsonl (resumable, ≈25 min)
cd ../ml && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python pilot_analysis.py   # tables, CSV and figures in ../data/pilot/
```

Requirements: `../sut` and `../tests` dependencies installed, Chromium for Playwright, `patch` in PATH.

## Files

| Path | What it is |
|---|---|
| `stryker.config.json` | Stryker config: mutates `src/modules/*/{logic,service,routes}.ts`, Vitest unit tests as runner |
| `scripts/stryker.ts` | Copies the SUT to `work/stryker-sut`, runs Stryker there, flattens the report into `mutants.json` |
| `mutants.json` | All mutants: id, file, module, line, mutator, original → replacement, unit-test status |
| `scripts/sample.ts` | Stratified seeded sample (proportional, ≥ 5 per module); skips `equivalent.json` and mutants that break the esbuild build |
| `equivalent.json` | Mutants judged equivalent by inspection, with the reason |
| `pilot/sample.json`, `pilot/excluded.json` | The sampled mutants and the skipped candidates (with reasons) |
| `manual/*.patch`, `manual/index.json` | Hand-made "realistic" bugs (unified diffs, applied with `patch -p1`) |
| `pilot-run.ts` | Pilot runner: clean control run + every defect → copy SUT, apply, `playwright test`, `normalize`, `rerun-failed` |
| `sim/plan.ts`, `sim/execute.ts` | Commit-history simulator (planner + executor), see `../notes/simulator.md` |
| `sim/benign.ts`, `sim/defects.ts`, `sim/tree.ts` | Benign-edit generators, defects as anchored edits, in-memory SUT snapshot |
| `work/` | Temporary copies and the run log (safe to delete) |

`pilot-run.ts` overwrites `../tests/results/{raw,run,build-status}.json` during the run and restores the previous
files at the end. Defect ids: `MUT-<stryker id>`, `MAN-NN`, `CLEAN-00`. Pass `--only ID,ID` to run a subset.

## Commit-history simulator (stage 4, done)

Extra defects: `npm run sim:isolate` (199 more mutants, each run alone, ≈ 1 h → `../data/history/isolation.jsonl`; `--batch 2 --seed 20261117 --size 220` adds a second batch, pinned in plans via `--isolation-batches 1,2`) and `npm run sim:tsc-check`. Plan options: `--pool extended|typed|sample|none`, `--defect-rate R`, `--commits N`, `--seed S`.

```bash
npm run sim:plan -- --preset smoke                      # 12-commit scenario → ../data/history-smoke/plan.json
npm run sim:run -- --dir ../data/history-smoke --reset  # replay into ../data/history-smoke/repo + run the suite per commit

# full history (300 commits, ≈ 90 min): data/history/{plan.json,runs.jsonl,repo/}; 600 commits: see ../notes/simulator.md §12
npm run sim:plan -- --commits 300 --seed 20261103 --pool typed --defect-rate 0.45
npm run sim:run -- --dir ../data/history --reset
```

The history is committed to a clone of `sut/` (never to `sut/` itself). `sim:run` halts (exit 2, `halt.json`) if a commit without open defects is not green.
