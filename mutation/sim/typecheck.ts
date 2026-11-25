// In-memory TypeScript check of a simulated tree (same tsconfig as the SUT). The planner uses it to verify that the
// "projection" of every commit (the tree with all open defects reverted) still type-checks, which catches
// semantic conflicts between benign edits and defect reverts at planning time instead of at the next green commit.
import path from 'node:path';
import ts from 'typescript';
import { SUT_DIR } from '../scripts/lib';
import type { Tree } from './tree';

export class Checker {
  private readonly options: ts.CompilerOptions;
  private readonly cache = new Map<string, ts.SourceFile>();
  private previous: ts.Program | undefined;

  constructor() {
    const config = ts.readConfigFile(path.join(SUT_DIR, 'tsconfig.json'), ts.sys.readFile);
    this.options = { ...ts.parseJsonConfigFileContent(config.config, ts.sys, SUT_DIR).options, noEmit: true };
  }

  /** Returns the diagnostics (as text) of `files` (paths relative to the SUT root) in `tree`. */
  check(tree: Tree, files: string[]): string[] {
    const byAbs = new Map([...tree.files.keys()].filter((f) => /^src\/.*\.ts$/.test(f)).map((f) => [path.join(SUT_DIR, f), f]));
    const host = ts.createCompilerHost(this.options, true);
    const { fileExists, readFile, getSourceFile } = host;
    host.fileExists = (p) => byAbs.has(p) || fileExists(p);
    host.readFile = (p) => (byAbs.has(p) ? tree.get(byAbs.get(p)!) : readFile(p));
    host.getSourceFile = (p, languageVersion, onError, shouldCreate) => {
      const rel = byAbs.get(p);
      const key = rel === undefined ? p : `${p}\0${tree.get(rel)}`;
      let sf = this.cache.get(key);
      if (!sf) {
        sf = rel === undefined ? getSourceFile(p, languageVersion, onError, shouldCreate) : ts.createSourceFile(p, tree.get(rel), languageVersion, true);
        if (sf) this.cache.set(key, sf);
      }
      return sf;
    };
    const program = ts.createProgram([...byAbs.keys()], this.options, host, this.previous);
    this.previous = program;
    const out: string[] = [];
    for (const f of files) {
      const sf = program.getSourceFile(path.join(SUT_DIR, f));
      if (!sf) continue;
      for (const d of [...program.getSyntacticDiagnostics(sf), ...program.getSemanticDiagnostics(sf)]) {
        out.push(`${f}: ${ts.flattenDiagnosticMessageText(d.messageText, '\n')}`);
      }
    }
    return out;
  }
}
