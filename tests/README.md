# Demo Shop E2E suite

A frozen Playwright (TypeScript) suite of 36 end-to-end tests for the Demo Shop SUT (`../sut`).
It is the subject of research on ML-based test prioritization: in stage 4 the suite is run in full on
hundreds of simulated commits of the SUT, and every test result becomes a dataset label. Therefore the
tests, their IDs and their number do not change between commits.

## Quick start

```bash
npm ci
npx playwright install chromium
npm test                 # build the SUT, run all tests (4 workers), write results/raw.json
npm run normalize        # results/raw.json → results/run.json
```

Requirements: Node.js 20+, the SUT dependencies installed (`cd ../sut && npm install`).

## How a run works

1. **Global setup** runs `npm run build` in `SUT_DIR` once. If the build fails, the run stops with
   **exit code 3** ("commit does not build"); test failures give exit code 1. The build status is saved
   to `results/build-status.json`.
2. **One SUT server per worker** (worker-scoped fixture `sutServer`): `node dist/server.js` with
   `NODE_ENV=test`, `PORT = BASE_PORT + workerIndex` and its own SQLite file in a temp dir. The fixture waits
   for `/health`, and kills the server (`SIGKILL`) on teardown. A browser page is loaded once per worker
   (`warmUp`) so that the first test of a worker is not charged with the browser's first-page cost.
3. **Before every test** the auto fixture `resetDb` calls `POST /__test__/reset`, so tests share no state
   and can run in any order or alone. A worker that sees a failure is replaced by Playwright, so a broken SUT
   state never leaks into the next test.
4. Logging in: `loginAs(user)` posts to `/login` through the browser context's request client, so the session
   cookie lands directly in the context. Only the auth tests log in through the UI. `apiAs(user)` and
   `pageAs(user)` create a second actor (e.g. an admin) with its own session.

Limits: test 15 s, `expect` 3 s, action/navigation 5 s, retries 0. A hanging mutant therefore turns into a
failed or timed-out test within seconds instead of blocking the run.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SUT_DIR` | `../sut` | SUT checkout to build and test (stage 4 points it to a modified copy) |
| `WORKERS` | `4` | Playwright workers, each with its own SUT server |
| `BASE_PORT` | `4100` | Port of worker 0; worker *i* uses `BASE_PORT + i` |
| `RAW_REPORT` | `results/raw.json` | Output of the Playwright JSON reporter |
| `PW_REPORTER` | `list` | Console reporter (`dot`, `line`, `null`, …); the JSON reporter is always on |
| `PW_TRACE` | `off` | Set to `retain-on-failure` to keep Playwright traces for debugging |
| `RUN_ID` | derived | Overrides `runId` in `run.json` (`<startedAt>-<short commit>` by default) |
| `STABILITY_RUNS` | `10` | Number of runs for `npm run stability` |

## Scripts

| Script | What it does |
|---|---|
| `npm test` | Full run, raw results in `results/raw.json` |
| `npm run test:headed` | Same with a visible browser and one worker (debugging) |
| `npm run manifest` | Lists the suite, validates the metadata, writes `manifest.json` (keeps measured durations) |
| `npm run normalize` | Converts `raw.json` (+ build status, SUT commit) into `results/run.json` |
| `npm run rerun-failed` | Re-runs the failed tests of `run.json` two more times and sets `flaky` / `confirmedFail` |
| `npm run stability` | Runs the suite N times (`--runs N`), prints failures and duration stats, updates `manifest.json` (skip with `--no-manifest`) |
| `npm run typecheck` | `tsc --noEmit` |

Typical stage-4 loop for one simulated commit:

```bash
SUT_DIR=/path/to/sut-commit RUN_ID=c0042 npm test; code=$?
SUT_DIR=/path/to/sut-commit RUN_ID=c0042 npm run normalize
[ $code -eq 1 ] && SUT_DIR=/path/to/sut-commit npm run rerun-failed
```

## Test metadata

Every test is declared as

```ts
test('[CHK-06] ships free from $50.00 after discount and always charges express',
  meta({ modules: ['checkout'], crit: 'high', covers: ['checkout/logic.ts', 'checkout/routes.ts'] }),
  async ({ page, loginAs }) => { … });
```

- **ID** in the title: `[PREFIX-NN]`, prefixes `AUTH, CAT, PRD, SRCH, CART, CHK, ORD, ADM, E2E`.
- **Tags**: `@module:<name>` (the first one is the module of the prefix; E2E tests have several) and
  exactly one `@crit:high|medium|low`.
- **`covers` annotations**: SUT files the test intentionally exercises, relative to `src/modules/`
  (`shared/…` and `seed/…` are relative to `src/`). This is the declared test ↔ file mapping.

Selectors are `data-testid` only (plus ARIA roles for `<option>`); there are no CSS/XPath selectors and no
fixed waits. Assertions use exact values from the business rules (amounts, error texts, `data-status`).

## `manifest.json`

```jsonc
{
  "generatedAt": "2026-10-21T09:10:00.000Z",
  "testCount": 36,
  "durations": { "runs": 10, "workers": 4, "measuredAt": "…" },   // null until `npm run stability`
  "tests": [{
    "id": "CHK-06",
    "title": "[CHK-06] ships free from $50.00 after discount and always charges express",
    "file": "specs/checkout/checkout.spec.ts",
    "modules": ["checkout"],
    "criticality": "high",
    "covers": ["checkout/logic.ts", "checkout/routes.ts", "checkout/views/checkout.ejs"],
    "coversPaths": ["src/modules/checkout/logic.ts", "…"],   // SUT repo paths, to match commit diffs
    "avgDurationMs": 352, "stdDurationMs": 30                 // from the stability runs
  }]
}
```

Validation (the script fails otherwise): unique IDs, known prefix, the first module tag matches the prefix,
E2E tests have ≥ 2 modules, exactly one criticality tag, at least one `covers` entry, and every covered file
exists in `SUT_DIR`.

## `results/run.json`

```jsonc
{
  "runId": "20261021T090331Z-1ea882e",
  "sutCommit": "1ea882e…",          // git -C $SUT_DIR rev-parse HEAD (null if not a git repo)
  "sutDir": "/…/sut",
  "startedAt": "2026-10-21T09:03:31.237Z",
  "buildOk": true,
  "buildError": null,               // first 1000 chars of the build output when buildOk = false
  "totalDurationMs": 4246,
  "tests": [{
    "id": "CHK-06",
    "status": "passed",             // passed | failed | timedOut | skipped ("interrupted" counts as failed)
    "durationMs": 331,
    "errorSnippet": null,           // first error message without ANSI codes, ≤ 400 chars
    "flaky": false,                 // failed, then passed at least once in rerun-failed
    "confirmedFail": false,         // failed in the run and in both re-runs
    "rerunStatuses": []             // statuses of the re-runs, empty if not re-run
  }]
}
```

When the commit does not build, `buildOk` is `false` and `tests` is empty.

## Layout

```
playwright.config.ts   global-setup.ts
fixtures/              sut-server.ts (per-worker server), test.ts (extended `test` with fixtures)
helpers/               env.ts, meta.ts, data.ts (seed data), api.ts (HTTP preconditions), ui.ts
specs/<module>/        auth, catalog, product, search, cart, checkout, orders, admin, e2e
scripts/               lib.ts, manifest.ts, normalize.ts, rerun-failed.ts, stability.ts
results/               run outputs (not versioned)
```
