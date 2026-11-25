# ML-based selection and prioritization of E2E tests for a web application

Reproducibility package of a master's thesis (2026): a controlled change history of a web application,
a Playwright E2E suite, mutation-based defect injection, and ML models that rank and select tests per commit.

The thesis text itself is not part of this repository.

## Layout

| Path | Contents |
|---|---|
| `sut/` | System under test: Node.js, TypeScript, Express, EJS, SQLite; Vitest unit tests |
| `tests/` | 36 Playwright E2E tests, test metadata `manifest.json`, result normalization and rerun scripts |
| `mutation/` | StrykerJS setup and mutant list, manual defects `manual/*.patch`, pilot, commit-history simulator `sim/` |
| `data/history-600/` | Main history (600 commits): `plan.json`, `runs.jsonl` (real Playwright runs on every commit), `repo.bundle` (the generated git history) |
| `data/history/` | Auxiliary 300-commit history, same structure |
| `data/dataset/` | Datasets; the final one is `history-600-540/` (first 540 commits) |
| `data/demo-stage7/` | CLI demo outputs |
| `ml/` | Dataset building and checks, metrics, baselines, models, selection policies, CLI, evaluation; frozen models `models/`, results `results/` |
| `notes/stage8-protocol.md` | Preregistration protocol of the final evaluation (rules for H1/H2, sha256 of frozen files, deviation log) |

Restore a generated history repository from its bundle:

```bash
git clone data/history-600/repo.bundle data/history-600/repo
```

## Environment

Node.js 22.18.0, Playwright 1.63.0 (Chromium), Python 3 with `ml/requirements.txt`.

```bash
(cd sut && npm ci) && (cd tests && npm ci && npx playwright install chromium) && (cd mutation && npm ci)
python3 -m venv ml/.venv && ml/.venv/bin/pip install -r ml/requirements.txt
```

## Reproduction

`npm` commands run in `mutation/`, Python commands in the repository root.

```bash
# 1. Mutants and pilot (~30 min)
npm run stryker && npm run sample && npm run pilot

# 2. Main 600-commit history (~4 h, runs the full Playwright suite on every commit)
npm run sim:plan -- --commits 600 --seed 20261117 --pool typed --defect-rate 0.45 \
  --isolation-batches 1,2 --out ../data/history-600
npm run sim:run -- --dir ../data/history-600 --reset

# 3. Dataset on the first 540 commits and leakage checks
ml/.venv/bin/python ml/build_dataset.py --history data/history-600 --max-commits 540
ml/.venv/bin/python ml/check_dataset.py --history data/history-600 --max-commits 540

# 4. Metrics, baselines, models, selection on the validation part (~75 s)
ml/.venv/bin/python ml/check_metrics.py
ml/.venv/bin/python ml/run_stage6.py

# 5. Final evaluation: hash check of frozen files, dry run on validation, test part
ml/.venv/bin/python ml/check_frozen.py
ml/.venv/bin/python ml/run_stage8.py --range val
ml/.venv/bin/python ml/run_stage8.py --range test --allow-test --h1-approved

# 6. Additional experiments, error analysis, figures
for e in ablation closed_loop rolling learning_curve cold_start label_noise \
         manual_bugs defect_types overhead cost_curve; do
  ml/.venv/bin/python ml/exp_$e.py
done
ml/.venv/bin/python ml/error_analysis.py
ml/.venv/bin/python ml/figures_stage8.py
```

Steps 1–2 are only needed to regenerate the raw data; the logged runs are already in `data/`.
The test-part evaluation refuses to run again if its results already exist.

Fixed seeds: history plan 20261117 (auxiliary history 20261103), pilot sample 20261021,
hyperparameter search 20261117, seed-dependent models 20261117…20261126.
