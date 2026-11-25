// npm run sample [-- --size 100 --seed 20261021] — picks the pilot mutants from mutants.json.
// Stratified by module (proportional, at least MIN_PER_MODULE each), seeded shuffle. Candidates listed in
// equivalent.json (obviously equivalent, decided by inspection) and mutants that break the esbuild build
// are skipped and logged; the next candidate of the same module takes their place.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { MODULES, MUTATION_DIR, PILOT_DIR, SUT_DIR, WORK_DIR, applyMutant, copySut, readMutants, rng, shuffle, writeJson, type Module, type Mutant } from './lib';

const MIN_PER_MODULE = 5;

function arg(name: string, fallback: number): number {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? Number(process.argv[i + 1]) : fallback;
}
const size = arg('size', 100);
const seed = arg('seed', 20261021);

/** Proportional allocation with a per-module minimum (largest remainder method). */
function allocate(counts: Record<Module, number>, total: number): Record<Module, number> {
  const quota = {} as Record<Module, number>;
  let fixed = MODULES.filter((m) => (counts[m] / sumOf(counts, MODULES)) * total < MIN_PER_MODULE);
  for (;;) {
    const free = MODULES.filter((m) => !fixed.includes(m));
    const rest = total - fixed.reduce((s, m) => s + Math.min(MIN_PER_MODULE, counts[m]), 0);
    const pool = sumOf(counts, free);
    const exact = free.map((m) => ({ m, x: (counts[m] / pool) * rest }));
    const newlyFixed = exact.filter(({ x }) => x < MIN_PER_MODULE).map(({ m }) => m);
    if (newlyFixed.length > 0) {
      fixed = [...fixed, ...newlyFixed];
      continue;
    }
    for (const m of fixed) quota[m] = Math.min(MIN_PER_MODULE, counts[m]);
    for (const { m, x } of exact) quota[m] = Math.floor(x);
    let left = rest - exact.reduce((s, { x }) => s + Math.floor(x), 0);
    for (const { m } of [...exact].sort((a, b) => b.x - Math.floor(b.x) - (a.x - Math.floor(a.x)))) {
      if (left-- <= 0) break;
      quota[m] += 1;
    }
    return quota;
  }
}

function sumOf(counts: Record<Module, number>, modules: readonly Module[]): number {
  return modules.reduce((s, m) => s + counts[m], 0);
}

interface Exclusion {
  id: string;
  reason: string;
}

const equivalentFile = path.join(MUTATION_DIR, 'equivalent.json');
const equivalent = new Map((JSON.parse(fs.readFileSync(equivalentFile, 'utf8')) as Exclusion[]).map((e) => [e.id, e.reason]));

const mutants = readMutants();
const counts = Object.fromEntries(MODULES.map((m) => [m, mutants.filter((x) => x.module === m).length])) as Record<Module, number>;
const quota = allocate(counts, size);

// One build-check copy; every candidate is applied to a fresh copy of its file and then restored.
const checkDir = path.join(WORK_DIR, 'build-check');
copySut(checkDir, { linkNodeModules: true });

function builds(mutant: Mutant): { ok: boolean; output: string } {
  const file = path.join(checkDir, mutant.file);
  applyMutant(checkDir, mutant);
  const r = spawnSync(process.execPath, ['scripts/build.mjs'], { cwd: checkDir, encoding: 'utf8' });
  fs.copyFileSync(path.join(SUT_DIR, mutant.file), file);
  return { ok: r.status === 0, output: `${r.stdout}${r.stderr}`.slice(0, 500) };
}

const order = shuffle(mutants, rng(seed));
const selected: Mutant[] = [];
const excluded: (Exclusion & { module: Module; file: string; mutator: string; original: string; replacement: string })[] = [];
for (const module of MODULES) {
  let taken = 0;
  for (const m of order.filter((x) => x.module === module)) {
    if (taken >= quota[module]) break;
    const info = { id: m.id, module, file: m.file, mutator: m.mutator, original: m.original, replacement: m.replacement };
    const reason = equivalent.get(m.id);
    if (reason) {
      excluded.push({ ...info, reason: `equivalent: ${reason}` });
      continue;
    }
    const build = builds(m);
    if (!build.ok) {
      excluded.push({ ...info, reason: `build fails: ${build.output}` });
      continue;
    }
    selected.push(m);
    taken += 1;
  }
}
fs.rmSync(checkDir, { recursive: true, force: true });

writeJson(path.join(PILOT_DIR, 'sample.json'), { seed, size, minPerModule: MIN_PER_MODULE, quota, mutants: selected });
writeJson(path.join(PILOT_DIR, 'excluded.json'), excluded);
console.log(`quota: ${JSON.stringify(quota)}`);
console.log(`sample.json: ${selected.length} mutants, excluded ${excluded.length}`);
