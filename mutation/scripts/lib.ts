// Shared helpers for the mutation scripts: paths, SUT copies, seeded randomness, defect definitions.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

export const MUTATION_DIR = path.resolve(__dirname, '..');
export const ROOT_DIR = path.resolve(MUTATION_DIR, '..');
export const SUT_DIR = path.join(ROOT_DIR, 'sut');
export const TESTS_DIR = path.join(ROOT_DIR, 'tests');
export const WORK_DIR = path.join(MUTATION_DIR, 'work');
export const REPORTS_DIR = path.join(MUTATION_DIR, 'reports');
export const MANUAL_DIR = path.join(MUTATION_DIR, 'manual');
export const PILOT_DIR = path.join(MUTATION_DIR, 'pilot');
export const MUTANTS_FILE = path.join(MUTATION_DIR, 'mutants.json');
export const DATA_PILOT_DIR = path.join(ROOT_DIR, 'data', 'pilot');

export const MODULES = ['auth', 'catalog', 'product', 'search', 'cart', 'checkout', 'orders', 'admin'] as const;
export type Module = (typeof MODULES)[number];

/** SUT entries that a runnable copy needs (node_modules is symlinked, dist/data/.git are left out). */
const SUT_ENTRIES = ['src', 'scripts', 'test', 'package.json', 'tsconfig.json', 'vitest.config.ts'];

/** Copies the SUT sources to `dest`; with `linkNodeModules` the copy reuses sut/node_modules via a symlink. */
export function copySut(dest: string, { linkNodeModules }: { linkNodeModules: boolean }): void {
  fs.rmSync(dest, { recursive: true, force: true });
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of SUT_ENTRIES) fs.cpSync(path.join(SUT_DIR, entry), path.join(dest, entry), { recursive: true });
  if (linkNodeModules) fs.symlinkSync(path.join(SUT_DIR, 'node_modules'), path.join(dest, 'node_modules'), 'dir');
}

export function moduleOf(file: string): Module | null {
  const m = /^src\/modules\/([^/]+)\//.exec(file);
  return m && (MODULES as readonly string[]).includes(m[1]) ? (m[1] as Module) : null;
}

export function sutHead(): string {
  const r = spawnSync('git', ['-C', SUT_DIR, 'rev-parse', 'HEAD'], { encoding: 'utf8' });
  if (r.status !== 0) throw new Error(`git rev-parse failed in ${SUT_DIR}`);
  return r.stdout.trim();
}

// ---------- Seeded randomness ----------

/** mulberry32: small deterministic PRNG, returns floats in [0, 1). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fisher–Yates shuffle into a new array. */
export function shuffle<T>(items: readonly T[], random: () => number): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// ---------- Mutants ----------

/** One Stryker mutant, flattened from the mutation-testing-report-schema JSON. */
export interface Mutant {
  id: string;
  file: string;
  module: Module;
  mutator: string;
  line: number;
  /** 1-based start/end positions as in the Stryker report (end column is exclusive). */
  location: { start: { line: number; column: number }; end: { line: number; column: number } };
  original: string;
  replacement: string;
  /** Status from the Vitest unit tests (Killed, Survived, NoCoverage, Timeout, …). */
  unitStatus: string;
}

export function readMutants(): Mutant[] {
  return (JSON.parse(fs.readFileSync(MUTANTS_FILE, 'utf8')) as { mutants: Mutant[] }).mutants;
}

/** Converts a 1-based line/column position into a string offset. */
export function offsetOf(source: string, line: number, column: number): number {
  let offset = 0;
  for (let l = 1; l < line; l++) {
    const next = source.indexOf('\n', offset);
    if (next < 0) throw new Error(`Line ${line} is past the end of the file`);
    offset = next + 1;
  }
  return offset + column - 1;
}

/** Applies a mutant to the file in `sutCopy`; checks that the original text is still in place. */
export function applyMutant(sutCopy: string, mutant: Mutant): void {
  const file = path.join(sutCopy, mutant.file);
  const source = fs.readFileSync(file, 'utf8');
  const start = offsetOf(source, mutant.location.start.line, mutant.location.start.column);
  const end = offsetOf(source, mutant.location.end.line, mutant.location.end.column);
  if (source.slice(start, end) !== mutant.original) throw new Error(`Mutant ${mutant.id}: original text not found in ${mutant.file}`);
  fs.writeFileSync(file, source.slice(0, start) + mutant.replacement + source.slice(end));
}

// ---------- Manual bugs ----------

export interface ManualBug {
  id: string;
  patch: string;
  file: string;
  description: string;
  /** Where the bug comes from: the stage 3 sensitivity check (D1–D6) or new for the pilot. */
  origin: string;
}

export function readManualBugs(): ManualBug[] {
  return JSON.parse(fs.readFileSync(path.join(MANUAL_DIR, 'index.json'), 'utf8')) as ManualBug[];
}

export function applyPatch(sutCopy: string, bug: ManualBug): void {
  const r = spawnSync('patch', ['-p1', '--forward', '--batch', '-d', sutCopy, '-i', path.join(MANUAL_DIR, bug.patch)], {
    encoding: 'utf8',
  });
  if (r.status !== 0) throw new Error(`Patch ${bug.patch} does not apply:\n${r.stdout}${r.stderr}`);
}

// ---------- Files ----------

export function writeJson(file: string, data: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(data, null, 2)}\n`);
}

export function readJsonl<T>(file: string): T[] {
  if (!fs.existsSync(file)) return [];
  return fs
    .readFileSync(file, 'utf8')
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as T);
}
