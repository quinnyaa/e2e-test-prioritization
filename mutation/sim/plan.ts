// npm run sim:plan -- [--preset smoke | --commits N] [--seed S] [--out DIR]
// Pure, seeded planner of the simulated commit history. No tests run here: it only decides, commit by commit,
// which anchored edits go in (benign edits, defect introductions, defect reverts) and writes <out>/plan.json.
// State model: benign edits are permanent; a defect stays open for a sampled lifetime and is then reverted by an
// explicit "fix" commit. Labels are NOT decided here: they come from the real test runs of the executor.
import fs from 'node:fs';
import path from 'node:path';
import { PILOT_DIR, ROOT_DIR, SUT_DIR, readJsonl, readMutants, rng as makeRng, writeJson, type Mutant } from '../scripts/lib';
import { makeBenign, type BenignKind } from './benign';
import { manualDefects, mutantToDefect, type Defect } from './defects';
import { chance, randInt, type Rng } from './rng';
import { Checker } from './typecheck';
import { Tree, reverse, type Edit } from './tree';

export const BASE = '1ea882e80a16b7805346718c9deb8ae1217d6b0e';
export const DEFAULT_SEED = 20261103;
/** Smoke seed: chosen so that every defect of the 12-commit scenario is killed by at least one test in the pilot. */
export const SMOKE_SEED = 20261108;
// Simulated time starts right after the base commit (2026-10-21T10:58+03:00), so no commit predates its parent.
const START = Date.parse('2026-10-22T07:00:00Z');

export type CommitKind = BenignKind | 'defect' | 'fix';

export interface Directive {
  kind: CommitKind;
  /** Number of defects to introduce (kind = defect; for fix commits a "regression" if > 0). */
  defects?: number;
  /** Number of benign companion edits. */
  benign?: number;
  /** Commits after the introducing one at which the defect is reverted (≥ 1). */
  lifetime?: number;
  /** Take the defect from the manual bugs. */
  manual?: boolean;
  /** Benign edits go to shared/*. */
  shared?: boolean;
  /** Also make a benign edit in the file the defect lives in. */
  sameFile?: boolean;
}

export type PlanEdit = Edit & { role: 'benign' | 'defect' | 'revert'; ref: string };

export interface PlanCommit {
  index: number;
  kind: CommitKind;
  message: string;
  date: string;
  edits: PlanEdit[];
  introduced: string[];
  fixed: string[];
  openAfter: string[];
}

/** 12 commits that exercise every code path: benign only, 1 and 2 defects, same-file edit, fix, shared edit, manual bug. */
export const SMOKE: Directive[] = [
  { kind: 'style', benign: 1 },
  { kind: 'feature', benign: 2 },
  { kind: 'defect', defects: 1, benign: 0, lifetime: 1 },
  { kind: 'fix', benign: 1 }, // forced by the schedule (defect from commit 2 is due)
  { kind: 'defect', defects: 2, benign: 1, lifetime: 3 },
  { kind: 'refactor', benign: 2, shared: true },
  { kind: 'defect', defects: 1, benign: 0, lifetime: 2, sameFile: true },
  { kind: 'fix', benign: 0 },
  { kind: 'fix', benign: 0 },
  { kind: 'defect', defects: 1, benign: 1, lifetime: 1, manual: true },
  { kind: 'fix', benign: 0 },
  { kind: 'refactor', benign: 3 },
];

const choose = <T>(options: [T, number][], rnd: Rng): T => {
  let r = rnd() * options.reduce((s, [, w]) => s + w, 0);
  for (const [v, w] of options) if ((r -= w) < 0) return v;
  return options[0][0];
};

function randomDirective(rnd: Rng, defectRate: number): Directive {
  if (chance(rnd, defectRate)) {
    return {
      kind: 'defect',
      defects: choose([[1, 65], [2, 28], [3, 7]], rnd),
      benign: choose([[0, 3], [1, 3], [2, 2], [3, 1]], rnd),
      lifetime: choose([[1, 45], [2, 25], [3, 15], [4, 10], [5, 5]], rnd),
      sameFile: chance(rnd, 0.3),
    };
  }
  const kind = choose<BenignKind>([['feature', 38], ['refactor', 30], ['style', 17], ['docs', 15]], rnd);
  const benign = kind === 'style' || kind === 'docs' ? randInt(rnd, 1, 2) : randInt(rnd, 1, 4);
  return { kind, benign, shared: (kind === 'refactor' || kind === 'feature') && chance(rnd, 0.2) };
}

const MESSAGES: Record<string, string[]> = {
  feature: ['add helper for {m}', 'extend {m} utilities', 'prepare {m} for upcoming changes'],
  refactor: ['rename locals in {m}', 'tidy up {m} code', 'document {m} functions'],
  style: ['adjust styles', 'add notes to {m} templates'],
  docs: ['update README notes', 'document {m} behaviour'],
  defect: ['rework {m} rules', 'simplify {m} checks', 'handle edge case in {m}', 'optimize {m} handling'],
  fix: ['correct {m} behaviour', 'revert faulty change in {m}', 'repair {m} regression'],
};

function moduleOfFiles(files: string[]): string {
  const counts = new Map<string, number>();
  for (const f of files) {
    const m = /^src\/modules\/([^/]+)\//.exec(f)?.[1] ?? (f.startsWith('src/shared/') ? 'shared' : f === 'README.md' ? 'docs' : 'ui');
    counts.set(m, (counts.get(m) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]?.[0] ?? 'shop';
}

function message(kind: CommitKind, files: string[], rnd: Rng): string {
  const m = moduleOfFiles(files);
  const prefix =
    kind === 'defect' ? choose([['feat', 5], ['refactor', 3], ['fix', 2], ['perf', 1]], rnd)
    : kind === 'fix' ? 'fix'
    : kind === 'feature' ? 'feat'
    : kind === 'docs' ? 'docs'
    : kind === 'style' ? 'style'
    : 'refactor';
  const text = MESSAGES[kind][Math.floor(rnd() * MESSAGES[kind].length)].replace('{m}', m);
  return `${prefix}(${m}): ${text}`;
}

interface Open {
  defect: Defect;
  dueAt: number;
}

export interface PlanOptions {
  seed: number;
  commits: number;
  script?: Directive[];
  pool: Defect[];
  /** Probability that a (non-forced) commit introduces defects. */
  defectRate?: number;
}

export function buildPlan(opt: PlanOptions): { commits: PlanCommit[]; unused: string[] } {
  const rnd = makeRng(opt.seed);
  const base = Tree.fromGit(SUT_DIR, BASE);
  let tree = base.clone();
  const mutants = opt.pool.filter((d) => d.type === 'mutant');
  const manual = opt.pool.filter((d) => d.type === 'manual');
  const shuffled = <T>(items: T[]): T[] => {
    const out = [...items];
    for (let i = out.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  };
  const pools = { mutant: shuffled(mutants), manual: shuffled(manual) };
  const open: Open[] = [];
  const commits: PlanCommit[] = [];
  let clock = START;
  const checker = new Checker();

  for (let i = 0; i < opt.commits; i++) {
    const planned = opt.script ? opt.script[i] : randomDirective(rnd, opt.defectRate ?? 0.28);
    const due = open.filter((o) => o.dueAt <= i);
    const d: Directive = due.length > 0 ? { ...planned, kind: 'fix', defects: planned.kind === 'fix' ? planned.defects : chance(rnd, 0.15) ? 1 : 0 } : planned;
    const edits: PlanEdit[] = [];
    const introduced: string[] = [];
    const fixed: string[] = [];
    let work = tree.clone();

    // 1. Reverts of the due defects (their reversed edits must still apply).
    for (const o of due) {
      for (const e of [...o.defect.edits].reverse()) {
        const rev = reverse(e);
        if (!work.canApply(rev)) throw new Error(`Cannot revert ${o.defect.id} at commit ${i}: anchor lost`);
        work.apply(rev);
        edits.push({ ...rev, role: 'revert', ref: o.defect.id });
      }
      fixed.push(o.defect.id);
      open.splice(open.indexOf(o), 1);
    }

    // 2. Pick the defects to introduce (not yet applied): needed so benign edits can protect their anchors.
    const wantDefects = d.kind === 'defect' || (d.kind === 'fix' && (d.defects ?? 0) > 0) ? (d.defects ?? 1) : 0;
    const picked: Defect[] = [];
    const takeFrom = (pool: Defect[]): Defect | null => {
      for (let k = 0; k < pool.length; k++) {
        const cand = pool[k];
        const files = new Set([...open.map((o) => o.defect.file), ...picked.map((p) => p.file)]);
        if (files.has(cand.file)) continue;
        const t = work.clone();
        if (!cand.edits.every((e) => t.canApply(e) && (t.apply(e), t.canApply(reverse(e))))) continue;
        pool.splice(k, 1);
        return cand;
      }
      return null;
    };
    const manualAllowed = d.manual === true || (opt.script === undefined && opt.commits >= 100 && i >= Math.floor(opt.commits * 0.8));
    for (let k = 0; k < wantDefects && open.length + picked.length < 4; k++) {
      const useManual = manualAllowed && (d.manual === true || chance(rnd, 0.5));
      const cand = (useManual ? takeFrom(pools.manual) : null) ?? takeFrom(pools.mutant);
      if (cand) picked.push(cand);
    }

    // 3. Benign companions; every edit must keep open defects revertible and picked defects applicable.
    const protect = (): Edit[] => [...open.flatMap((o) => o.defect.edits.map(reverse)), ...picked.flatMap((p) => p.edits)];
    const benignKind: BenignKind = d.kind === 'defect' || d.kind === 'fix' ? (choose<BenignKind>([['feature', 4], ['refactor', 3], ['style', 1], ['docs', 1]], rnd)) : d.kind;
    const wantBenign = d.benign ?? 0;
    let sameFileDone = !(d.sameFile && picked.length > 0);
    for (let k = 0; k < wantBenign || !sameFileDone; k++) {
      const onlyFile = !sameFileDone ? picked[0].file : undefined;
      const b = makeBenign(work, rnd, { kind: benignKind, shared: d.shared, onlyFile, blockRename: new Set([...open.map((o) => o.defect.file), ...picked.map((p) => p.file)]) });
      if (!sameFileDone) sameFileDone = true; // one attempt only; the defect still goes in if it fails
      if (!b) continue;
      const t = work.clone();
      if (!t.canApply(b.edit)) continue;
      t.apply(b.edit);
      if (!protect().every((p) => t.canApply(p))) continue;
      work = t;
      edits.push({ ...b.edit, role: 'benign', ref: b.gen });
    }

    // 4. Introduce the defects.
    for (const p of picked) {
      for (const e of p.edits) {
        work.apply(e);
        edits.push({ ...e, role: 'defect', ref: p.id });
      }
      introduced.push(p.id);
      open.push({ defect: p, dueAt: i + Math.max(1, d.lifetime ?? choose([[1, 45], [2, 25], [3, 15], [4, 10], [5, 5]], rnd)) });
    }

    // A commit must change something; fall back to a docs note.
    if (edits.length === 0) {
      const b = makeBenign(work, rnd, { kind: 'docs' });
      if (b) {
        work.apply(b.edit);
        edits.push({ ...b.edit, role: 'benign', ref: b.gen });
      }
    }
    if (edits.length === 0) throw new Error(`Commit ${i} ended up empty`);

    // Projection check: the tree with every open defect reverted must still type-check (guards against benign edits
    // that conflict with a later revert). Only the TypeScript files touched by this commit are checked.
    const projection = work.clone();
    for (const o of open) for (const e of [...o.defect.edits].reverse()) projection.apply(reverse(e));
    const touched = [...new Set(edits.map((e) => e.file).filter((f) => f.endsWith('.ts')))];
    const problems = checker.check(projection, touched);
    if (problems.length > 0) throw new Error(`Commit ${i}: the defect-free projection does not type-check:\n${problems.join('\n')}`);

    tree = work;
    clock += randInt(rnd, 1, 6) * 3600_000;
    const kind: CommitKind = fixed.length > 0 ? 'fix' : introduced.length > 0 ? 'defect' : d.kind === 'defect' || d.kind === 'fix' ? benignKind : d.kind;
    commits.push({
      index: i,
      kind,
      message: message(kind, edits.map((e) => e.file), rnd),
      date: new Date(clock).toISOString(),
      edits,
      introduced,
      fixed,
      openAfter: open.map((o) => o.defect.id),
    });
  }
  return { commits, unused: [...pools.mutant, ...pools.manual].map((p) => p.id) };
}

const ISOLATION_FILE = path.join(ROOT_DIR, 'data', 'history', 'isolation.jsonl');

interface IsolationRecord {
  defectId: string;
  batch?: number;
  tscOk: boolean;
  buildOk: boolean;
  detected: boolean;
  tests: { id: string; confirmedFail: boolean; flaky: boolean }[];
}

export const readIsolation = (): IsolationRecord[] => readJsonl<IsolationRecord>(ISOLATION_FILE);

/**
 * Defect pool: the 100 pilot mutants and MAN-01–08, plus (unless `sampleOnly`) the mutants of `sim:isolate`
 * that build and gave no flaky test in their isolated run. Mutants whose `tsc` fails are kept unless `typedOnly`.
 */
export function loadPool(opt: { sampleOnly?: boolean; typedOnly?: boolean; batches?: number[] } = {}): Defect[] {
  const batches = new Set(opt.batches ?? [1]);
  const base = Tree.fromGit(SUT_DIR, BASE);
  const sample = (JSON.parse(fs.readFileSync(path.join(PILOT_DIR, 'sample.json'), 'utf8')) as { mutants: Mutant[] }).mutants;
  const defects: Defect[] = [];
  // tsc status of the pilot mutants and manual bugs (npm run sim:tsc-check); only needed for typedOnly.
  const tscPilot: Record<string, boolean> = opt.typedOnly ? (JSON.parse(fs.readFileSync(path.join(ROOT_DIR, 'data', 'history', 'tsc-pilot.json'), 'utf8')) as Record<string, boolean>) : {};
  for (const m of sample) {
    if (opt.typedOnly && !tscPilot[`MUT-${m.id}`]) continue;
    const d = mutantToDefect(m, base);
    if (!d) throw new Error(`Mutant ${m.id} cannot be anchored in the base commit`);
    defects.push(d);
  }
  if (!opt.sampleOnly) {
    const byId = new Map(readMutants().map((m) => [`MUT-${m.id}`, m]));
    for (const rec of readIsolation()) {
      const m = byId.get(rec.defectId);
      if (!m || !batches.has(rec.batch ?? 1) || !rec.buildOk || rec.tests.some((t) => t.flaky) || (opt.typedOnly && !rec.tscOk)) continue;
      const d = mutantToDefect(m, base);
      if (d) defects.push(d);
    }
  }
  const tscOf = new Map<string, boolean>([...Object.entries(tscPilot), ...readIsolation().map((r): [string, boolean] => [r.defectId, r.tscOk])]);
  return [...defects, ...manualDefects().filter((d) => !opt.typedOnly || tscOf.get(d.id) === true)];
}

/** Single-defect kill sets (defect id → confirmed-failing test ids) from the pilot and from the isolation runs. */
export function pilotKillSets(): Map<string, string[]> {
  const runs = readJsonl<{ defectId: string; tests: { id: string; confirmedFail: boolean }[] }>(path.join(ROOT_DIR, 'data', 'pilot', 'runs.jsonl'));
  return new Map([...runs, ...readIsolation()].map((r) => [r.defectId, r.tests.filter((t) => t.confirmedFail).map((t) => t.id)]));
}

function main(): void {
  const args = process.argv.slice(2);
  const arg = (name: string): string | undefined => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined);
  const smoke = arg('--preset') === 'smoke';
  const seed = Number(arg('--seed') ?? (smoke ? SMOKE_SEED : DEFAULT_SEED));
  const commits = smoke ? SMOKE.length : Number(arg('--commits') ?? 300);
  const out = path.resolve(arg('--out') ?? path.join(ROOT_DIR, 'data', smoke ? 'history-smoke' : 'history'));

  // The smoke scenario is pinned to the pilot pool so that it does not change as isolation.jsonl grows.
  const poolMode = arg('--pool') ?? (smoke ? 'sample' : 'extended');
  // Isolation batches in the pool (default: batch 1 only, so the 300-commit history stays reproducible as new batches are added).
  const batches = (arg('--isolation-batches') ?? '1').split(',').map(Number);
  const pool = poolMode === 'none' ? [] : loadPool({ sampleOnly: poolMode === 'sample', typedOnly: poolMode === 'typed', batches });
  const plan = buildPlan({ seed, commits, script: smoke ? SMOKE : undefined, pool, defectRate: arg('--defect-rate') ? Number(arg('--defect-rate')) : undefined });
  const kills = pilotKillSets();
  const tests = 36;
  let expectedFailEvents = 0;
  let withOpen = 0;
  for (const c of plan.commits) {
    const failing = new Set(c.openAfter.flatMap((id) => kills.get(id) ?? []));
    expectedFailEvents += failing.size;
    if (c.openAfter.length > 0) withOpen++;
  }
  const byKind: Record<string, number> = {};
  for (const c of plan.commits) byKind[c.kind] = (byKind[c.kind] ?? 0) + 1;
  const stats = {
    commits,
    byKind,
    commitsWithOpenDefects: withOpen,
    defectsIntroduced: plan.commits.reduce((s, c) => s + c.introduced.length, 0),
    // Upper-bound estimate from single-defect pilot kill sets (union over open defects; masking and unknown defects ignored).
    expectedFailRate: Number((expectedFailEvents / (commits * tests)).toFixed(4)),
    unusedDefects: plan.unused.length,
  };
  writeJson(path.join(out, 'plan.json'), { version: 1, seed, base: BASE, poolMode, ...(batches.join() === '1' ? {} : { isolationBatches: batches }), poolSize: pool.length, defectRate: arg('--defect-rate') ? Number(arg('--defect-rate')) : 0.28, stats, commits: plan.commits });
  console.log(`plan: ${out}/plan.json`);
  console.log(JSON.stringify(stats, null, 2));
}

if (require.main === module || process.argv[1]?.endsWith('plan.ts')) main();
