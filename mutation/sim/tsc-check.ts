// npm run sim:tsc-check — records whether each pilot-sample mutant and manual bug passes `tsc --noEmit`
// (isolation.jsonl already has this for the extension mutants). Result: ../data/history/tsc-pilot.json.
// Used by `sim:plan --pool typed` to leave out defects that a TypeScript CI would stop before the E2E stage.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { PILOT_DIR, ROOT_DIR, applyMutant, copySut, writeJson, type Mutant } from '../scripts/lib';
import { manualDefects } from './defects';

const copy = fs.mkdtempSync(path.join(os.tmpdir(), 'tsc-sut-'));
const tsc = (): boolean => spawnSync('npx', ['tsc', '--noEmit'], { cwd: copy, stdio: 'ignore' }).status === 0;
const result: Record<string, boolean> = {};
try {
  const sample = (JSON.parse(fs.readFileSync(path.join(PILOT_DIR, 'sample.json'), 'utf8')) as { mutants: Mutant[] }).mutants;
  for (const m of sample) {
    copySut(copy, { linkNodeModules: true });
    applyMutant(copy, m);
    result[`MUT-${m.id}`] = tsc();
  }
  for (const d of manualDefects()) {
    copySut(copy, { linkNodeModules: true });
    for (const e of d.edits) {
      const file = path.join(copy, e.file);
      const src = fs.readFileSync(file, 'utf8');
      if (src.split(e.find).length !== 2) throw new Error(`${d.id} does not apply`);
      fs.writeFileSync(file, src.replace(e.find, () => e.replace));
    }
    result[d.id] = tsc();
  }
} finally {
  fs.rmSync(copy, { recursive: true, force: true });
}
writeJson(path.join(ROOT_DIR, 'data', 'history', 'tsc-pilot.json'), result);
const values = Object.values(result);
console.log(`tsc-check: ${values.filter(Boolean).length}/${values.length} pass tsc`);
