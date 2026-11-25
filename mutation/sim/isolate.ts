// npm run sim:isolate -- [--size 200] [--seed 20261109] [--batch 1] [--out DIR]
// Extends the defect pool: samples more Stryker mutants (not in the pilot sample, not equivalent), and runs each one
// ALONE on the base SUT: tsc, then the full frozen suite + normalize + rerun-failed. The single-defect kill sets
// let the planner estimate the failure rate and let stage 8 analyse multi-fault masking.
// With `--manual` it isolates the manual bugs added in stage 4 (origin "stage 4, new") instead of sampling mutants.
// Result: <out>/isolation.jsonl (default data/history/). Resumable; a defect is never run twice.
// `--batch N` (N > 1) samples a further batch that excludes the mutants of earlier batches; its records carry `batch: N`
// so that a plan can pin which batches its pool uses (records without `batch` belong to batch 1).
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { MODULES, MUTATION_DIR, PILOT_DIR, ROOT_DIR, TESTS_DIR, WORK_DIR, applyMutant, applyPatch, copySut, moduleOf, readJsonl, readManualBugs, readMutants, rng, shuffle, type Module, type Mutant } from '../scripts/lib';

const args = process.argv.slice(2);
const arg = (name: string, fallback: string): string => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const SIZE = Number(arg('--size', '200'));
const SEED = Number(arg('--seed', '20261109'));
const OUT = path.resolve(arg('--out', path.join(ROOT_DIR, 'data', 'history')));
const FILE = path.join(OUT, 'isolation.jsonl');
const BATCH = Number(arg('--batch', '1'));
const MIN_PER_MODULE = 5;
const MAX_STRING_LITERAL_SHARE = 0.2; // the pilot sample was 26 % StringLiteral (texts of messages); keep the extension more varied

const RUN_JSON = path.join(TESTS_DIR, 'results', 'run.json');
const RESULT_FILES = ['raw.json', 'run.json', 'build-status.json'];
const BACKUP_DIR = path.join(WORK_DIR, 'tests-results-backup-isolate');

interface RunJson {
  runId: string;
  startedAt: string;
  buildOk: boolean;
  buildError: string | null;
  totalDurationMs: number;
  tests: { id: string; status: string; durationMs: number; flaky: boolean; confirmedFail: boolean; rerunStatuses: string[] }[];
}

/** Candidates = all mutants minus the pilot sample, the equivalent/excluded ones and those already isolated. */
function candidates(): Mutant[] {
  const skip = new Set<string>();
  for (const m of (JSON.parse(fs.readFileSync(path.join(PILOT_DIR, 'sample.json'), 'utf8')) as { mutants: Mutant[] }).mutants) skip.add(m.id);
  for (const e of JSON.parse(fs.readFileSync(path.join(PILOT_DIR, 'excluded.json'), 'utf8')) as { id: string }[]) skip.add(e.id);
  for (const e of JSON.parse(fs.readFileSync(path.join(MUTATION_DIR, 'equivalent.json'), 'utf8')) as { id: string }[]) skip.add(e.id);
  // Mutants of earlier batches (only earlier ones, so that the selection of this batch is stable when it is resumed).
  for (const r of readJsonl<{ defectId: string; batch?: number }>(FILE)) if ((r.batch ?? 1) < BATCH) skip.add(r.defectId.replace(/^MUT-/, ''));
  return readMutants().filter((m) => !skip.has(m.id));
}

/** Proportional-by-module selection (≥ MIN_PER_MODULE each), seeded, with a cap on StringLiteral mutants. */
function select(all: Mutant[]): Mutant[] {
  const random = rng(SEED);
  const order = shuffle(all, random);
  const total = all.length;
  const quota = {} as Record<Module, number>;
  let assigned = 0;
  for (const m of MODULES) {
    quota[m] = Math.max(MIN_PER_MODULE, Math.round((all.filter((x) => x.module === m).length / total) * SIZE));
    assigned += quota[m];
  }
  // Trim the largest quotas if the minimums pushed the total over SIZE.
  while (assigned > SIZE) {
    const biggest = [...MODULES].sort((a, b) => quota[b] - quota[a])[0];
    quota[biggest]--;
    assigned--;
  }
  const maxStrings = Math.floor(SIZE * MAX_STRING_LITERAL_SHARE);
  let strings = 0;
  const taken = Object.fromEntries(MODULES.map((m) => [m, 0])) as Record<Module, number>;
  const chosen: Mutant[] = [];
  for (const m of order) {
    if (taken[m.module] >= quota[m.module]) continue;
    if (m.mutator === 'StringLiteral') {
      if (strings >= maxStrings) continue;
      strings++;
    }
    taken[m.module]++;
    chosen.push(m);
  }
  return shuffle(chosen, rng(SEED + 1)); // interleave modules so machine-load drift is not tied to one module
}

function sh(cmd: string, cmdArgs: string[], cwd: string, env: Record<string, string> = {}): { status: number; ms: number } {
  const t0 = Date.now();
  const r = spawnSync(cmd, cmdArgs, { cwd, env: { ...process.env, ...env }, stdio: ['ignore', 'ignore', 'inherit'] });
  return { status: r.status ?? 1, ms: Date.now() - t0 };
}

interface Job {
  id: string;
  meta: Record<string, unknown>;
  apply: (copy: string) => void;
}

const mutantJob = (m: Mutant): Job => ({
  id: `MUT-${m.id}`,
  meta: { file: m.file, module: m.module, mutator: m.mutator, line: m.line, original: m.original, replacement: m.replacement, unitStatus: m.unitStatus },
  apply: (copy) => applyMutant(copy, m),
});

const manualJobs = (): Job[] =>
  readManualBugs()
    .filter((b) => b.origin === 'stage 4, new')
    .map((b) => ({
      id: b.id,
      meta: { file: b.file, module: moduleOf(b.file), mutator: null, description: b.description },
      apply: (copy) => applyPatch(copy, b),
    }));

function isolate(job: Job, copy: string): Record<string, unknown> {
  copySut(copy, { linkNodeModules: true });
  job.apply(copy);
  const tsc = sh('npx', ['tsc', '--noEmit'], copy);
  const runId = `ISO-${job.id}`;
  const env = { SUT_DIR: copy, RUN_ID: runId, PW_REPORTER: 'null' };
  const pw = sh('npx', ['playwright', 'test'], TESTS_DIR, env);
  if (sh('npm', ['run', '--silent', 'normalize'], TESTS_DIR, env).status !== 0) throw new Error(`normalize failed for ${runId}`);
  if (pw.status === 1 && sh('npm', ['run', '--silent', 'rerun-failed'], TESTS_DIR, env).status !== 0) throw new Error(`rerun-failed failed for ${runId}`);
  const run = JSON.parse(fs.readFileSync(RUN_JSON, 'utf8')) as RunJson;
  if (run.runId !== runId) throw new Error(`run.json belongs to ${run.runId}, expected ${runId}`);
  return {
    defectId: job.id,
    ...(BATCH > 1 ? { batch: BATCH } : {}),
    ...job.meta,
    tscOk: tsc.status === 0,
    exitCode: pw.status,
    buildOk: run.buildOk,
    buildError: run.buildError,
    startedAt: run.startedAt,
    totalDurationMs: run.totalDurationMs,
    detected: run.tests.some((t) => t.confirmedFail),
    tests: run.tests.map(({ id, status, durationMs, flaky, confirmedFail, rerunStatuses }) => ({ id, status, durationMs, flaky, confirmedFail, rerunStatuses })),
  };
}

function main(): void {
  const manual = args.includes('--manual');
  const all = manual ? [] : candidates();
  const jobs = manual ? manualJobs() : select(all).map(mutantJob);
  const done = new Set(readJsonl<{ defectId: string }>(FILE).map((r) => r.defectId));
  const todo = jobs.filter((j) => !done.has(j.id));
  console.log(`isolate${manual ? ' (manual bugs)' : ''}: ${jobs.length} selected${manual ? '' : ` of ${all.length} candidates (seed ${SEED})`}, ${done.size} already isolated, ${todo.length} to run`);
  fs.mkdirSync(OUT, { recursive: true });
  if (!manual) {
    fs.writeFileSync(path.join(OUT, BATCH > 1 ? `isolation-meta-b${BATCH}.json` : 'isolation-meta.json'), `${JSON.stringify({ batch: BATCH, seed: SEED, size: SIZE, node: process.version, platform: `${os.platform()} ${os.release()} ${os.arch()}`, workers: Number(process.env.WORKERS ?? 4), selected: jobs.map((j) => j.id) }, null, 2)}\n`);
  }

  if (!fs.existsSync(BACKUP_DIR)) {
    fs.mkdirSync(BACKUP_DIR, { recursive: true });
    for (const f of RESULT_FILES) {
      const src = path.join(TESTS_DIR, 'results', f);
      if (fs.existsSync(src)) fs.copyFileSync(src, path.join(BACKUP_DIR, f));
    }
  }
  const copy = fs.mkdtempSync(path.join(os.tmpdir(), 'iso-sut-'));
  try {
    todo.forEach((job, i) => {
      const t0 = Date.now();
      const rec = isolate(job, copy);
      fs.appendFileSync(FILE, `${JSON.stringify(rec)}\n`);
      const failed = (rec.tests as { id: string; confirmedFail: boolean }[]).filter((t) => t.confirmedFail).map((t) => t.id);
      console.log(`[${i + 1}/${todo.length}] ${job.id} ${String(rec.module)}/${String(rec.mutator)} tsc=${rec.tscOk} build=${rec.buildOk} fail=[${failed.join(',')}] ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    });
  } finally {
    fs.rmSync(copy, { recursive: true, force: true });
    for (const f of RESULT_FILES) {
      const src = path.join(BACKUP_DIR, f);
      if (fs.existsSync(src)) fs.copyFileSync(src, path.join(TESTS_DIR, 'results', f));
    }
    fs.rmSync(BACKUP_DIR, { recursive: true, force: true });
  }
}

main();
