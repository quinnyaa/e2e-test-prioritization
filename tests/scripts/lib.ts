// Shared helpers for the manifest, normalize, rerun-failed and stability scripts.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { MANIFEST_FILE, RESULTS_DIR, SUT_DIR, TESTS_DIR } from '../helpers/env';
import { CRITICALITIES, MODULES, type Criticality, type Module } from '../helpers/meta';

// ---------- Playwright JSON report ----------

interface JsonAnnotation {
  type: string;
  description?: string;
}

interface JsonError {
  message?: string;
  value?: string;
}

interface JsonResult {
  status: string;
  duration: number;
  errors: JsonError[];
  startTime: string;
}

interface JsonTest {
  annotations: JsonAnnotation[];
  results: JsonResult[];
}

interface JsonSpec {
  title: string;
  file: string;
  line: number;
  tags: string[];
  tests: JsonTest[];
}

interface JsonSuite {
  specs?: JsonSpec[];
  suites?: JsonSuite[];
}

export interface JsonReport {
  suites: JsonSuite[];
  errors: JsonError[];
  stats: { startTime: string; duration: number; expected: number; unexpected: number; skipped: number; flaky: number };
}

export interface ReportedTest {
  id: string;
  title: string;
  file: string;
  line: number;
  tags: string[];
  annotations: JsonAnnotation[];
  /** One entry per execution (several with --repeat-each). */
  results: JsonResult[];
}

const ID_PATTERN = /^\[([A-Z0-9]+-\d{2})\]\s+/;

export function parseTestId(title: string): string | null {
  return ID_PATTERN.exec(title)?.[1] ?? null;
}

export function readReport(file: string): JsonReport {
  return JSON.parse(fs.readFileSync(file, 'utf8')) as JsonReport;
}

/** Flattens the report into one entry per test title; repeated executions are merged. */
export function reportedTests(report: JsonReport): ReportedTest[] {
  const byTitle = new Map<string, ReportedTest>();
  const visit = (suite: JsonSuite) => {
    for (const spec of suite.specs ?? []) {
      const id = parseTestId(spec.title);
      if (!id) throw new Error(`Test title has no [ID] prefix: "${spec.title}" (${spec.file})`);
      const entry = byTitle.get(spec.title) ?? {
        id,
        title: spec.title,
        file: spec.file,
        line: spec.line,
        tags: spec.tags,
        annotations: spec.tests[0]?.annotations ?? [],
        results: [],
      };
      for (const test of spec.tests) entry.results.push(...test.results);
      byTitle.set(spec.title, entry);
    }
    for (const child of suite.suites ?? []) visit(child);
  };
  for (const suite of report.suites) visit(suite);
  return [...byTitle.values()];
}

export type RunStatus = 'passed' | 'failed' | 'timedOut' | 'skipped';

export function normalizeStatus(status: string | undefined): RunStatus {
  if (status === 'passed' || status === 'timedOut' || status === 'skipped') return status;
  return 'failed'; // 'failed' and 'interrupted'
}

const ANSI = /\u001b\[[0-9;]*m/g;

export function errorSnippet(result: JsonResult | undefined, maxLength = 400): string | null {
  const error = result?.errors?.[0];
  const text = (error?.message ?? error?.value ?? '').replace(ANSI, '').trim();
  if (!text) return null;
  return text.length > maxLength ? `${text.slice(0, maxLength)}…` : text;
}

// ---------- Running Playwright ----------

export function runPlaywright(
  args: string[],
  env: Record<string, string> = {},
  { quiet = false } = {},
): { exitCode: number; wallMs: number } {
  const started = Date.now();
  const result = spawnSync(process.platform === 'win32' ? 'npx.cmd' : 'npx', ['playwright', 'test', ...args], {
    cwd: TESTS_DIR,
    stdio: quiet ? ['ignore', 'ignore', 'inherit'] : 'inherit',
    env: { ...process.env, ...env },
  });
  return { exitCode: result.status ?? 1, wallMs: Date.now() - started };
}

export function grepForIds(ids: string[]): string {
  return ids.map((id) => `\\[${id}\\]`).join('|');
}

export function sutCommit(): string | null {
  const result = spawnSync('git', ['-C', SUT_DIR, 'rev-parse', 'HEAD'], { encoding: 'utf8' });
  return result.status === 0 ? result.stdout.trim() : null;
}

// ---------- Manifest ----------

export const ID_PREFIX_MODULE: Record<string, Module | null> = {
  AUTH: 'auth',
  CAT: 'catalog',
  PRD: 'product',
  SRCH: 'search',
  CART: 'cart',
  CHK: 'checkout',
  ORD: 'orders',
  ADM: 'admin',
  E2E: null,
};

export interface ManifestTest {
  id: string;
  title: string;
  file: string;
  modules: Module[];
  criticality: Criticality;
  /** Declared SUT files, as written in the test (relative to src/modules, or shared/…). */
  covers: string[];
  /** The same files as repository paths of the SUT (src/…), for matching against commit diffs. */
  coversPaths: string[];
  avgDurationMs: number | null;
  stdDurationMs: number | null;
}

export interface Manifest {
  generatedAt: string;
  testCount: number;
  durations: { runs: number; workers: number; measuredAt: string } | null;
  tests: ManifestTest[];
}

export function coversPath(file: string): string {
  return /^(shared|seed)\//.test(file) ? `src/${file}` : `src/modules/${file}`;
}

export function readManifest(): Manifest | null {
  return fs.existsSync(MANIFEST_FILE) ? (JSON.parse(fs.readFileSync(MANIFEST_FILE, 'utf8')) as Manifest) : null;
}

export function writeJson(file: string, data: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(data, null, 2)}\n`);
}

/** Lists the suite without running it and validates the metadata of every test. */
export function buildManifest(): Manifest {
  const listFile = path.join(RESULTS_DIR, 'list.json');
  const { exitCode } = runPlaywright(['--list'], { RAW_REPORT: listFile, PW_REPORTER: 'null' }, { quiet: true });
  if (exitCode !== 0) throw new Error(`playwright test --list failed with exit code ${exitCode}`);
  const tests = reportedTests(readReport(listFile));
  const previous = new Map((readManifest()?.tests ?? []).map((t) => [t.id, t]));

  const problems: string[] = [];
  const seen = new Set<string>();
  const manifestTests = tests.map((t): ManifestTest => {
    const where = `${t.id} (${t.file}:${t.line})`;
    if (seen.has(t.id)) problems.push(`${where}: duplicate id`);
    seen.add(t.id);

    const prefix = t.id.split('-')[0];
    const modules = t.tags.filter((tag) => tag.startsWith('module:')).map((tag) => tag.slice(7)) as Module[];
    const crits = t.tags.filter((tag) => tag.startsWith('crit:')).map((tag) => tag.slice(5));
    const covers = t.annotations.filter((a) => a.type === 'covers').map((a) => a.description ?? '');

    if (!(prefix in ID_PREFIX_MODULE)) problems.push(`${where}: unknown id prefix ${prefix}`);
    if (modules.length === 0) problems.push(`${where}: no @module tag`);
    for (const m of modules) if (!MODULES.includes(m)) problems.push(`${where}: unknown module ${m}`);
    const own = ID_PREFIX_MODULE[prefix];
    if (own && modules[0] !== own) problems.push(`${where}: first @module tag must be ${own}`);
    if (prefix === 'E2E' && modules.length < 2) problems.push(`${where}: end-to-end test needs several modules`);
    if (crits.length !== 1 || !CRITICALITIES.includes(crits[0] as Criticality)) {
      problems.push(`${where}: exactly one @crit:high|medium|low tag required`);
    }
    if (covers.length === 0) problems.push(`${where}: no covers annotation`);
    for (const file of covers) {
      if (!fs.existsSync(path.join(SUT_DIR, coversPath(file)))) problems.push(`${where}: covers missing file ${file}`);
    }

    const old = previous.get(t.id);
    return {
      id: t.id,
      title: t.title,
      file: `specs/${t.file}`,
      modules,
      criticality: crits[0] as Criticality,
      covers,
      coversPaths: covers.map(coversPath),
      avgDurationMs: old?.avgDurationMs ?? null,
      stdDurationMs: old?.stdDurationMs ?? null,
    };
  });

  if (problems.length > 0) throw new Error(`Invalid test metadata:\n  ${problems.join('\n  ')}`);
  manifestTests.sort((a, b) => a.id.localeCompare(b.id, 'en', { numeric: true }));
  return {
    generatedAt: new Date().toISOString(),
    testCount: manifestTests.length,
    durations: readManifest()?.durations ?? null,
    tests: manifestTests,
  };
}

// ---------- Statistics ----------

export function mean(values: number[]): number {
  return values.reduce((s, v) => s + v, 0) / values.length;
}

/** Sample standard deviation (n − 1); 0 for fewer than two values. */
export function std(values: number[]): number {
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(values.reduce((s, v) => s + (v - m) ** 2, 0) / (values.length - 1));
}
