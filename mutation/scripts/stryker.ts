// npm run stryker — generates mutants of sut/src/modules/*/{logic,service,routes}.ts with Stryker.
// Stryker runs in a plain copy of the SUT (work/stryker-sut), so the sut/ repository is never touched;
// the Vitest unit tests are only the runner Stryker needs. Output: reports/mutation.json and mutants.json.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { MUTANTS_FILE, MUTATION_DIR, REPORTS_DIR, WORK_DIR, copySut, moduleOf, offsetOf, sutHead, writeJson, type Mutant } from './lib';

interface ReportMutant {
  id: string;
  mutatorName: string;
  replacement?: string;
  status: string;
  location: Mutant['location'];
}

interface MutationReport {
  files: Record<string, { source: string; mutants: ReportMutant[] }>;
}

const strykerSut = path.join(WORK_DIR, 'stryker-sut');
// No node_modules link: the unit tests import pure code only, and vitest resolves from mutation/node_modules.
copySut(strykerSut, { linkNodeModules: false });

const stryker = path.join(MUTATION_DIR, 'node_modules', '.bin', 'stryker');
const result = spawnSync(stryker, ['run', path.join(MUTATION_DIR, 'stryker.config.json')], { cwd: strykerSut, stdio: 'inherit' });
if (result.status !== 0) throw new Error(`Stryker exited with code ${result.status}`);

fs.mkdirSync(REPORTS_DIR, { recursive: true });
for (const name of ['mutation.json', 'mutation.html']) {
  fs.copyFileSync(path.join(strykerSut, 'reports', name), path.join(REPORTS_DIR, name));
}

const report = JSON.parse(fs.readFileSync(path.join(REPORTS_DIR, 'mutation.json'), 'utf8')) as MutationReport;
const mutants: Mutant[] = [];
for (const [file, { source, mutants: fileMutants }] of Object.entries(report.files)) {
  const module = moduleOf(file);
  if (!module) throw new Error(`Unexpected mutated file ${file}`);
  for (const m of fileMutants) {
    const start = offsetOf(source, m.location.start.line, m.location.start.column);
    const end = offsetOf(source, m.location.end.line, m.location.end.column);
    mutants.push({
      id: m.id,
      file,
      module,
      mutator: m.mutatorName,
      line: m.location.start.line,
      location: m.location,
      original: source.slice(start, end),
      replacement: m.replacement ?? '',
      unitStatus: m.status,
    });
  }
}
mutants.sort((a, b) => Number(a.id) - Number(b.id));

writeJson(MUTANTS_FILE, {
  generatedAt: new Date().toISOString(),
  sutCommit: sutHead(),
  stryker: '10.0.0',
  count: mutants.length,
  mutants,
});
console.log(`mutants.json: ${mutants.length} mutants`);
