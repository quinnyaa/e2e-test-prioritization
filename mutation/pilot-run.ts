// npm run pilot [-- --only MAN-01,MUT-411] — stage 4 pilot: one full E2E run per defect.
// For each defect (a sampled Stryker mutant or a manual bug) the SUT is copied to a temp dir (node_modules is
// a symlink), the change is applied, the frozen suite runs with SUT_DIR pointing to the copy, then
// `normalize` and, if tests failed, `rerun-failed`. Every result is appended to data/pilot/runs.jsonl.
// Resumable: defects already present in runs.jsonl are skipped. A clean control run goes first.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  DATA_PILOT_DIR,
  MANUAL_DIR,
  PILOT_DIR,
  TESTS_DIR,
  WORK_DIR,
  applyMutant,
  applyPatch,
  copySut,
  moduleOf,
  readJsonl,
  readManualBugs,
  rng,
  shuffle,
  sutHead,
  writeJson,
  type Module,
  type Mutant,
} from './scripts/lib';

const SEED = 20261021;
const RUNS_FILE = path.join(DATA_PILOT_DIR, 'runs.jsonl');
const RUN_JSON = path.join(TESTS_DIR, 'results', 'run.json');
const RESULT_FILES = ['raw.json', 'run.json', 'build-status.json'];
const BACKUP_DIR = path.join(WORK_DIR, 'tests-results-backup');

type DefectType = 'clean' | 'mutant' | 'manual';

interface Defect {
  id: string;
  type: DefectType;
  file: string | null;
  module: Module | null;
  mutator: string | null;
  line: number | null;
  original: string | null;
  replacement: string | null;
  description: string | null;
  /** Stryker status from the Vitest unit tests (mutants only). */
  unitStatus: string | null;
  apply: (sutCopy: string) => void;
}

interface RunTest {
  id: string;
  status: string;
  durationMs: number;
  errorSnippet: string | null;
  flaky: boolean;
  confirmedFail: boolean;
  rerunStatuses: string[];
}

interface RunJson {
  runId: string;
  startedAt: string;
  buildOk: boolean;
  buildError: string | null;
  totalDurationMs: number;
  tests: RunTest[];
}

function mutantDefect(m: Mutant): Defect {
  return {
    id: `MUT-${m.id}`,
    type: 'mutant',
    file: m.file,
    module: m.module,
    mutator: m.mutator,
    line: m.line,
    original: m.original,
    replacement: m.replacement,
    description: null,
    unitStatus: m.unitStatus,
    apply: (dir) => applyMutant(dir, m),
  };
}

function loadDefects(): Defect[] {
  const sample = (JSON.parse(fs.readFileSync(path.join(PILOT_DIR, 'sample.json'), 'utf8')) as { mutants: Mutant[] }).mutants;
  const manual = readManualBugs().map(
    (b): Defect => ({
      id: b.id,
      type: 'manual',
      file: b.file,
      module: moduleOf(b.file),
      mutator: null,
      line: null,
      original: null,
      replacement: null,
      description: `${b.description} [${b.origin}]`,
      unitStatus: null,
      apply: (dir) => applyPatch(dir, b),
    }),
  );
  const clean: Defect = {
    id: 'CLEAN-00',
    type: 'clean',
    file: null,
    module: null,
    mutator: null,
    line: null,
    original: null,
    replacement: null,
    description: 'Control run on the unchanged SUT (false-failure check)',
    unitStatus: null,
    apply: () => {},
  };
  // Fixed seeded order interleaves modules and defect types, so machine-load drift is not tied to one group.
  return [clean, ...shuffle([...sample.map(mutantDefect), ...manual], rng(SEED))];
}

function npm(script: string, env: Record<string, string>, quiet = true): number {
  const r = spawnSync('npm', ['run', '--silent', script], {
    cwd: TESTS_DIR,
    env: { ...process.env, ...env },
    stdio: quiet ? ['ignore', 'ignore', 'inherit'] : 'inherit',
  });
  return r.status ?? 1;
}

function runDefect(defect: Defect, sutCopy: string): Record<string, unknown> {
  copySut(sutCopy, { linkNodeModules: true });
  defect.apply(sutCopy);

  const env = { SUT_DIR: sutCopy, RUN_ID: defect.id, PW_REPORTER: 'null' };
  const started = Date.now();
  const testExit = spawnSync('npx', ['playwright', 'test'], {
    cwd: TESTS_DIR,
    env: { ...process.env, ...env },
    stdio: ['ignore', 'ignore', 'inherit'],
  }).status;
  const testWallMs = Date.now() - started;
  if (npm('normalize', env) !== 0) throw new Error(`normalize failed for ${defect.id}`);
  let rerunWallMs = 0;
  if (testExit === 1) {
    const rerunStarted = Date.now();
    if (npm('rerun-failed', env) !== 0) throw new Error(`rerun-failed failed for ${defect.id}`);
    rerunWallMs = Date.now() - rerunStarted;
  }

  const run = JSON.parse(fs.readFileSync(RUN_JSON, 'utf8')) as RunJson;
  if (run.runId !== defect.id) throw new Error(`run.json belongs to ${run.runId}, expected ${defect.id}`);
  const { apply: _apply, id: _id, ...meta } = defect;
  return {
    defectId: defect.id,
    ...meta,
    exitCode: testExit,
    buildOk: run.buildOk,
    buildError: run.buildError,
    startedAt: run.startedAt,
    totalDurationMs: run.totalDurationMs,
    testWallMs,
    rerunWallMs,
    detected: run.tests.some((t) => t.confirmedFail),
    tests: run.tests.map(({ id, status, durationMs, flaky, confirmedFail, rerunStatuses, errorSnippet }) => ({
      id,
      status,
      durationMs,
      flaky,
      confirmedFail,
      rerunStatuses,
      errorSnippet,
    })),
  };
}

function backupResults(): void {
  if (fs.existsSync(BACKUP_DIR)) return; // keep the backup of the state before the first (possibly interrupted) pilot
  fs.mkdirSync(BACKUP_DIR, { recursive: true });
  for (const f of RESULT_FILES) {
    const src = path.join(TESTS_DIR, 'results', f);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(BACKUP_DIR, f));
  }
}

function restoreResults(): void {
  for (const f of RESULT_FILES) {
    const src = path.join(BACKUP_DIR, f);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(TESTS_DIR, 'results', f));
  }
  fs.rmSync(BACKUP_DIR, { recursive: true, force: true });
}

const onlyArg = process.argv.indexOf('--only');
const only = onlyArg >= 0 ? new Set(process.argv[onlyArg + 1].split(',')) : null;

const defects = loadDefects().filter((d) => !only || only.has(d.id));
const done = new Set(readJsonl<{ defectId: string }>(RUNS_FILE).map((r) => r.defectId));
const todo = defects.filter((d) => !done.has(d.id));
console.log(`pilot: ${defects.length} defects, ${done.size} already in runs.jsonl, ${todo.length} to run`);

writeJson(path.join(DATA_PILOT_DIR, 'meta.json'), {
  seed: SEED,
  sutCommit: sutHead(),
  node: process.version,
  platform: `${os.platform()} ${os.release()} ${os.arch()}`,
  cpus: os.cpus().length,
  workers: Number(process.env.WORKERS ?? 4),
  playwright: '1.63.0',
  stryker: '10.0.0',
  defects: defects.map((d) => d.id),
  manualPatches: fs.readdirSync(MANUAL_DIR).filter((f) => f.endsWith('.patch')),
});

backupResults();
const sutCopy = fs.mkdtempSync(path.join(os.tmpdir(), 'pilot-sut-'));
try {
  todo.forEach((defect, i) => {
    const t0 = Date.now();
    const record = runDefect(defect, sutCopy);
    fs.appendFileSync(RUNS_FILE, `${JSON.stringify(record)}\n`);
    const failed = (record.tests as RunTest[]).filter((t) => t.confirmedFail).map((t) => t.id);
    console.log(
      `[${i + 1}/${todo.length}] ${defect.id} ${defect.file ?? ''} build=${record.buildOk} ` +
        `fail=[${failed.join(',')}] ${((Date.now() - t0) / 1000).toFixed(1)}s`,
    );
  });
} finally {
  fs.rmSync(sutCopy, { recursive: true, force: true });
  restoreResults();
}
