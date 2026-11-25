// npm run sim:run -- [--dir DIR] [--limit N] [--reset] [--commits-only] [--repo PATH]
// Replays <dir>/plan.json into a separate git repo (a clone of sut/ at the base commit; sut/ itself is never touched),
// one real commit per plan entry with deterministic author, dates and messages. After each commit it runs
// tsc + the unit tests (diagnostics), then the full frozen E2E suite (+ normalize + rerun-failed) and appends the
// result to <dir>/runs.jsonl. Resumable by commit index. Halts (exit 2, <dir>/halt.json) when the history is
// not trustworthy: a commit does not build, or a commit with no open defect is not fully green.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { ROOT_DIR, SUT_DIR, TESTS_DIR, WORK_DIR, readJsonl, writeJson } from '../scripts/lib';
import { BASE, type PlanCommit } from './plan';
import { Tree, git, type Edit } from './tree';

const RESULT_FILES = ['raw.json', 'run.json', 'build-status.json'];
const BACKUP_DIR = path.join(WORK_DIR, 'tests-results-backup-sim');

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

const args = process.argv.slice(2);
const flag = (name: string): boolean => args.includes(name);
const arg = (name: string): string | undefined => (flag(name) ? args[args.indexOf(name) + 1] : undefined);

const dir = path.resolve(arg('--dir') ?? path.join(ROOT_DIR, 'data', 'history'));
const repo = path.resolve(arg('--repo') ?? path.join(dir, 'repo'));
const commitsOnly = flag('--commits-only');
const RUNS_FILE = path.join(dir, 'runs.jsonl');
const RUN_JSON = path.join(TESTS_DIR, 'results', 'run.json');

const AUTHOR = { name: 'Shop Developer', email: 'dev@demo-shop.test' };

function halt(reason: string, detail: unknown): never {
  writeJson(path.join(dir, 'halt.json'), { at: new Date().toISOString(), reason, detail });
  console.error(`HALT: ${reason}`);
  if (fs.existsSync(BACKUP_DIR)) restoreResults(); // process.exit skips the `finally` in main()
  process.exit(2);
}

function ensureRepo(): void {
  if (!fs.existsSync(path.join(repo, '.git'))) {
    fs.mkdirSync(path.dirname(repo), { recursive: true });
    const r = spawnSync('git', ['clone', '--quiet', '--no-hardlinks', SUT_DIR, repo], { encoding: 'utf8' });
    if (r.status !== 0) throw new Error(`git clone failed: ${r.stderr}`);
    git(repo, ['checkout', '--quiet', '-b', 'sim', BASE]);
    git(repo, ['remote', 'remove', 'origin']);
  }
  // The SUT .gitignore has `node_modules/`, which does not match a symlink, so exclude it locally.
  fs.writeFileSync(path.join(repo, '.git', 'info', 'exclude'), 'node_modules\n');
  const link = path.join(repo, 'node_modules');
  if (!fs.existsSync(link)) fs.symlinkSync(path.join(SUT_DIR, 'node_modules'), link, 'dir');
  if (git(repo, ['status', '--porcelain']).trim() !== '') halt('repo working tree is dirty', git(repo, ['status', '--porcelain']));
}

function applyEdits(edits: Edit[]): void {
  for (const e of edits) {
    const file = path.join(repo, e.file);
    const tree = new Tree(new Map([[e.file, fs.readFileSync(file, 'utf8')]]));
    tree.apply(e);
    fs.writeFileSync(file, tree.get(e.file));
  }
}

function commit(c: PlanCommit): string {
  applyEdits(c.edits);
  git(repo, ['add', '-A']);
  const env = {
    ...process.env,
    GIT_AUTHOR_NAME: AUTHOR.name,
    GIT_AUTHOR_EMAIL: AUTHOR.email,
    GIT_COMMITTER_NAME: AUTHOR.name,
    GIT_COMMITTER_EMAIL: AUTHOR.email,
    GIT_AUTHOR_DATE: c.date,
    GIT_COMMITTER_DATE: c.date,
  };
  const r = spawnSync('git', ['-C', repo, '-c', 'commit.gpgsign=false', 'commit', '--quiet', '--no-verify', '-m', c.message], { env, encoding: 'utf8' });
  if (r.status !== 0) throw new Error(`git commit failed for #${c.index}: ${r.stdout}${r.stderr}`);
  return git(repo, ['rev-parse', 'HEAD']).trim();
}

function sh(cmd: string, cmdArgs: string[], cwd: string, env: Record<string, string> = {}): { status: number; ms: number } {
  const t0 = Date.now();
  const r = spawnSync(cmd, cmdArgs, { cwd, env: { ...process.env, ...env }, stdio: ['ignore', 'ignore', 'inherit'] });
  return { status: r.status ?? 1, ms: Date.now() - t0 };
}

function npmTests(script: string, env: Record<string, string>): number {
  return sh('npm', ['run', '--silent', script], TESTS_DIR, env).status;
}

function runSuite(runId: string): { exitCode: number; testWallMs: number; rerunWallMs: number; run: RunJson } {
  const env = { SUT_DIR: repo, RUN_ID: runId, PW_REPORTER: 'null' };
  const pw = sh('npx', ['playwright', 'test'], TESTS_DIR, env);
  if (npmTests('normalize', env) !== 0) halt(`normalize failed for ${runId}`, null);
  let rerunWallMs = 0;
  if (pw.status === 1) {
    const t0 = Date.now();
    if (npmTests('rerun-failed', env) !== 0) halt(`rerun-failed failed for ${runId}`, null);
    rerunWallMs = Date.now() - t0;
  }
  const run = JSON.parse(fs.readFileSync(RUN_JSON, 'utf8')) as RunJson;
  if (run.runId !== runId) halt(`run.json belongs to ${run.runId}, expected ${runId}`, null);
  return { exitCode: pw.status, testWallMs: pw.ms, rerunWallMs, run };
}

function backupResults(): void {
  if (fs.existsSync(BACKUP_DIR)) return;
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

function numstat(parent: string, sha: string): { path: string; added: number; deleted: number }[] {
  return git(repo, ['diff', '--numstat', parent, sha])
    .split('\n')
    .filter(Boolean)
    .map((l) => {
      const [a, d, p] = l.split('\t');
      return { path: p, added: Number(a), deleted: Number(d) };
    });
}

function main(): void {
  const plan = JSON.parse(fs.readFileSync(path.join(dir, 'plan.json'), 'utf8')) as { seed: number; base: string; commits: PlanCommit[] };
  if (plan.base !== BASE) halt('plan was made for another base commit', { plan: plan.base, expected: BASE });
  if (flag('--reset')) {
    fs.rmSync(repo, { recursive: true, force: true });
    fs.rmSync(RUNS_FILE, { force: true });
    fs.rmSync(path.join(dir, 'halt.json'), { force: true });
  }
  fs.rmSync(path.join(dir, 'halt.json'), { force: true });
  ensureRepo();

  const done = new Set(readJsonl<{ index: number }>(RUNS_FILE).map((r) => r.index));
  let existing = Number(git(repo, ['rev-list', '--count', `${BASE}..HEAD`]).trim());
  const limit = Number(arg('--limit') ?? plan.commits.length);
  const todo = plan.commits.slice(0, limit).filter((c) => commitsOnly || !done.has(c.index));
  console.log(`sim: ${plan.commits.length} planned, ${done.size} already in runs.jsonl, ${todo.length} to process (repo at ${existing} commits)`);

  writeJson(path.join(dir, 'meta.json'), {
    seed: plan.seed,
    base: BASE,
    node: process.version,
    platform: `${os.platform()} ${os.release()} ${os.arch()}`,
    cpus: os.cpus().length,
    workers: Number(process.env.WORKERS ?? 4),
    playwright: '1.63.0',
  });

  if (!commitsOnly) backupResults();
  const shas: string[] = [];
  try {
    for (const c of todo) {
      const t0 = Date.now();
      if (existing === c.index) {
        shas[c.index] = commit(c);
        existing++;
      } else if (existing === c.index + 1) {
        if (git(repo, ['log', '-1', '--format=%s']).trim() !== c.message) halt(`repo HEAD does not match plan commit #${c.index}`, null);
        shas[c.index] = git(repo, ['rev-parse', 'HEAD']).trim();
      } else if (commitsOnly && existing > c.index) {
        shas[c.index] = git(repo, ['rev-list', '--reverse', `${BASE}..HEAD`]).split('\n')[c.index];
        continue;
      } else {
        halt(`repo has ${existing} commits but plan commit #${c.index} is next`, null);
      }
      if (commitsOnly) {
        console.log(`#${c.index} ${shas[c.index].slice(0, 10)} ${c.message}`);
        continue;
      }

      const sha = shas[c.index];
      const parent = git(repo, ['rev-parse', `${sha}^`]).trim();
      const controls = c.openAfter.length === 0;
      const tsc = sh('npm', ['run', '--silent', 'typecheck'], repo);
      const unit = sh('npx', ['vitest', 'run', '--reporter=dot'], repo);
      const suite = runSuite(`C${String(c.index).padStart(4, '0')}`);
      const failed = suite.run.tests.filter((t) => t.confirmedFail).map((t) => t.id);
      const flaky = suite.run.tests.filter((t) => t.flaky).map((t) => t.id);

      const record = {
        index: c.index,
        sha,
        parentSha: parent,
        kind: c.kind, // generator metadata for analysis only; never a model feature (features come from the diff)
        message: c.message,
        date: c.date,
        introduced: c.introduced,
        fixed: c.fixed,
        openAfter: c.openAfter,
        files: numstat(parent, sha),
        typecheckOk: tsc.status === 0,
        unitOk: unit.status === 0,
        exitCode: suite.exitCode,
        buildOk: suite.run.buildOk,
        buildError: suite.run.buildError,
        startedAt: suite.run.startedAt,
        totalDurationMs: suite.run.totalDurationMs,
        testWallMs: suite.testWallMs,
        rerunWallMs: suite.rerunWallMs,
        tests: suite.run.tests.map(({ id, status, durationMs, flaky: fl, confirmedFail, rerunStatuses, errorSnippet }) => ({ id, status, durationMs, flaky: fl, confirmedFail, rerunStatuses, errorSnippet })),
      };
      fs.mkdirSync(dir, { recursive: true });
      fs.appendFileSync(RUNS_FILE, `${JSON.stringify(record)}\n`);
      console.log(
        `#${c.index} ${sha.slice(0, 10)} ${c.kind.padEnd(8)} open=[${c.openAfter.join(',')}] tsc=${record.typecheckOk} unit=${record.unitOk} ` +
          `fail=[${failed.join(',')}]${flaky.length ? ` flaky=[${flaky.join(',')}]` : ''} ${((Date.now() - t0) / 1000).toFixed(1)}s`,
      );

      if (!suite.run.buildOk) halt(`commit #${c.index} does not build`, suite.run.buildError);
      if (controls && (failed.length > 0 || flaky.length > 0 || !record.typecheckOk || !record.unitOk)) {
        halt(`commit #${c.index} has no open defect but is not green`, { failed, flaky, typecheckOk: record.typecheckOk, unitOk: record.unitOk });
      }
    }
  } finally {
    if (!commitsOnly) restoreResults();
    if (commitsOnly) writeJson(path.join(dir, 'shas.json'), shas);
  }
}

main();
