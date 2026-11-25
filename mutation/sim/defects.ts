// Defects as anchored edits: Stryker mutants (line/column → whole-line text anchors, so later benign edits
// that shift lines do not matter) and the manual bugs (unified diffs → one edit per hunk).
import fs from 'node:fs';
import path from 'node:path';
import { MANUAL_DIR, offsetOf, readManualBugs, type Mutant } from '../scripts/lib';
import { Tree, type Edit } from './tree';

export interface Defect {
  id: string;
  type: 'mutant' | 'manual';
  file: string;
  edits: Edit[];
}

/** Anchors a mutant to its whole lines (grown by neighbours until unique in the base file). */
export function mutantToDefect(m: Mutant, base: Tree): Defect | null {
  const src = base.get(m.file);
  const s = offsetOf(src, m.location.start.line, m.location.start.column);
  const e = offsetOf(src, m.location.end.line, m.location.end.column);
  if (src.slice(s, e) !== m.original) return null;
  let a = src.lastIndexOf('\n', s - 1) + 1;
  let b = src.indexOf('\n', e);
  if (b < 0) b = src.length;
  for (let grow = 0; grow < 8 && base.count(m.file, src.slice(a, b)) !== 1; grow++) {
    a = src.lastIndexOf('\n', a - 2) + 1;
    const nb = src.indexOf('\n', b + 1);
    b = nb < 0 ? src.length : nb;
  }
  const find = src.slice(a, b);
  if (base.count(m.file, find) !== 1) return null;
  return { id: `MUT-${m.id}`, type: 'mutant', file: m.file, edits: [{ file: m.file, find, replace: src.slice(a, s) + m.replacement + src.slice(e, b) }] };
}

/** Parses a unified diff into edits (old = context + removed lines, new = context + added lines). */
export function patchToEdits(patch: string): Edit[] {
  const edits: Edit[] = [];
  let file = '';
  let oldLines: string[] | null = null;
  let newLines: string[] = [];
  const flush = (): void => {
    if (oldLines) edits.push({ file, find: oldLines.join('\n'), replace: newLines.join('\n') });
    oldLines = null;
  };
  for (const line of patch.split('\n')) {
    if (line.startsWith('+++ ')) {
      flush();
      file = line.slice(4).replace(/^b\//, '').trim();
    } else if (line.startsWith('@@')) {
      flush();
      oldLines = [];
      newLines = [];
    } else if (oldLines) {
      if (line.startsWith('\\')) continue;
      const kind = line[0];
      if (kind === ' ' || kind === '-') oldLines.push(line.slice(1));
      if (kind === ' ' || kind === '+') newLines.push(line.slice(1));
    }
  }
  flush();
  return edits;
}

export function manualDefects(): Defect[] {
  return readManualBugs().map((bug) => ({
    id: bug.id,
    type: 'manual' as const,
    file: bug.file,
    edits: patchToEdits(fs.readFileSync(path.join(MANUAL_DIR, bug.patch), 'utf8')),
  }));
}
