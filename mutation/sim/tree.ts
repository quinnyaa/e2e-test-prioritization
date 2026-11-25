// In-memory snapshot of the SUT sources plus anchored text edits. The planner works on this instead of the disk,
// so planning is pure (no I/O besides reading the base commit) and reproducible from a seed.
import { spawnSync } from 'node:child_process';

export interface Edit {
  file: string;
  /** Exact text that must occur exactly once in the file (whole lines, grown until unique). */
  find: string;
  replace: string;
}

export const reverse = (e: Edit): Edit => ({ file: e.file, find: e.replace, replace: e.find });

/** Files the simulator may change: sources, templates, styles and the README (docs commits). */
export const TRACKED = /^(src\/.*\.(ts|ejs|css)|README\.md)$/;

export function git(dir: string, args: string[]): string {
  const r = spawnSync('git', ['-C', dir, ...args], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (r.status !== 0) throw new Error(`git ${args.join(' ')} failed in ${dir}: ${r.stderr}`);
  return r.stdout;
}

export class Tree {
  constructor(readonly files: Map<string, string>) {}

  /** Reads the tracked files of `rev` straight from the git object store (the working tree is not touched). */
  static fromGit(repoDir: string, rev: string): Tree {
    const names = git(repoDir, ['ls-tree', '-r', '--name-only', rev]).split('\n').filter((f) => TRACKED.test(f));
    return new Tree(new Map(names.map((f) => [f, git(repoDir, ['show', `${rev}:${f}`])])));
  }

  clone(): Tree {
    return new Tree(new Map(this.files));
  }

  get(file: string): string {
    const content = this.files.get(file);
    if (content === undefined) throw new Error(`Unknown file ${file}`);
    return content;
  }

  /** Number of (possibly overlapping) occurrences of `text` in `file`. */
  count(file: string, text: string): number {
    if (text === '') return 0;
    const content = this.get(file);
    let n = 0;
    for (let i = content.indexOf(text); i >= 0; i = content.indexOf(text, i + 1)) n++;
    return n;
  }

  canApply(e: Edit): boolean {
    return this.files.has(e.file) && this.count(e.file, e.find) === 1;
  }

  apply(e: Edit): void {
    if (!this.canApply(e)) throw new Error(`Edit does not apply uniquely to ${e.file}: ${JSON.stringify(e.find.slice(0, 80))}`);
    const content = this.get(e.file);
    const i = content.indexOf(e.find);
    this.files.set(e.file, content.slice(0, i) + e.replace + content.slice(i + e.find.length));
  }
}

/**
 * Turns "file content before → after" into one anchored Edit: the changed line range plus as much context
 * as is needed for the `find` text to be unique. Returns null if no unique anchor exists within 12 lines.
 */
export function diffEdit(tree: Tree, file: string, after: string): Edit | null {
  const before = tree.get(file);
  if (before === after) return null;
  const a = before.split('\n');
  const b = after.split('\n');
  let p = 0;
  while (p < a.length && p < b.length && a[p] === b[p]) p++;
  let s = 0;
  while (s < a.length - p && s < b.length - p && a[a.length - 1 - s] === b[b.length - 1 - s]) s++;
  for (let k = 0; k <= 12; k++) {
    const from = Math.max(0, p - k);
    const toA = Math.min(a.length, a.length - s + k);
    const toB = Math.min(b.length, b.length - s + k);
    const find = a.slice(from, toA).join('\n');
    if (find !== '' && tree.count(file, find) === 1) return { file, find, replace: b.slice(from, toB).join('\n') };
  }
  return null;
}
