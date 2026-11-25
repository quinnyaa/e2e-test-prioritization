// Seeded generators of behaviour-preserving ("benign") edits. A generator maps a file's current text to new
// text; `diffEdit` turns the pair into an anchored Edit. None of them changes what the E2E tests observe, so a
// commit made only of these must be green: the executor treats a red commit without open defects as a halt.
import ts from 'typescript';
import type { Rng } from './rng';
import { Tree, diffEdit, type Edit } from './tree';

export type BenignKind = 'feature' | 'refactor' | 'style' | 'docs';

export interface Benign {
  gen: string;
  edit: Edit;
}

type Transform = (src: string, rnd: Rng, ctx: { module: string | null; tree: Tree; file: string }) => string | null;

const WORDS = ['Label', 'Entry', 'Row', 'Item', 'Line', 'Field', 'Slot', 'Group', 'Block', 'Part'];
const NOTES = [
  'Keep in sync with the order summary',
  'Cents are integers here, never floats',
  'Edge case: empty input is handled by the caller',
  'Validated before this point',
  'Order matters: the first match wins',
  'Kept as a separate step for readability',
  'Mirrors the rule shown on the storefront',
  'Revisit after the next release',
];

const pick = <T>(items: readonly T[], rnd: Rng): T => items[Math.floor(rnd() * items.length)];
const humanize = (name: string): string => {
  const words = name.replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase();
  return words.charAt(0).toUpperCase() + words.slice(1);
};
const parse = (src: string): ts.SourceFile => ts.createSourceFile('f.ts', src, ts.ScriptTarget.ES2022, true);

// ---------- Transforms (TypeScript sources) ----------

/** Adds a JSDoc line to a top-level function that has none. */
const jsdoc: Transform = (src, rnd) => {
  const sf = parse(src);
  const fns = sf.statements.filter(
    (s): s is ts.FunctionDeclaration => ts.isFunctionDeclaration(s) && !!s.name && !(s as unknown as { jsDoc?: unknown[] }).jsDoc?.length,
  );
  if (fns.length === 0) return null;
  const fn = pick(fns, rnd);
  const at = fn.getStart(sf, false);
  const lineStart = src.lastIndexOf('\n', at - 1) + 1;
  const suffix = pick(['.', ' for the current request.', ' without touching the database.', ' using integer cents.'], rnd);
  return `${src.slice(0, lineStart)}/** ${humanize(fn.name!.text)}${suffix} */\n${src.slice(lineStart)}`;
};

/** Inserts a `// note` above a statement line inside a function body. */
const comment: Transform = (src, rnd) => {
  const lines = src.split('\n');
  const idx = lines.map((l, i) => (/^ {2,}(if|return|const|for) /.test(l) ? i : -1)).filter((i) => i >= 0);
  if (idx.length === 0) return null;
  const i = pick(idx, rnd);
  const indent = /^ */.exec(lines[i])![0];
  lines.splice(i, 0, `${indent}// ${pick(NOTES, rnd)}`);
  return lines.join('\n');
};

/** Renames a function-local `const`/`let` (all uses in the function) by appending a suffix. */
const renameLocal: Transform = (src, rnd) => {
  const sf = parse(src);
  const found: { fn: ts.FunctionLikeDeclaration; decl: ts.VariableDeclaration }[] = [];
  const walk = (node: ts.Node, fn: ts.FunctionLikeDeclaration | null): void => {
    const next = ts.isFunctionDeclaration(node) || ts.isArrowFunction(node) || ts.isFunctionExpression(node) ? node : fn;
    if (next && ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.name.text.length > 2) found.push({ fn: next, decl: node });
    ts.forEachChild(node, (c) => walk(c, next));
  };
  walk(sf, null);
  if (found.length === 0) return null;
  const { fn, decl } = pick(found, rnd);
  const old = (decl.name as ts.Identifier).text;
  const renamed = old + pick(['Value', 'Local', 'Result', 'Amount', 'Current'], rnd);
  if (new RegExp(`\\b${renamed}\\b`).test(src)) return null;
  // Bail out if the name is bound more than once in the function (shadowing) or in nested functions.
  let bindings = 0;
  const countBindings = (n: ts.Node): void => {
    if ((ts.isVariableDeclaration(n) || ts.isParameter(n) || ts.isBindingElement(n)) && ts.isIdentifier(n.name) && n.name.text === old) bindings++;
    if (ts.isBindingElement(n) && n.propertyName && ts.isIdentifier(n.propertyName) && n.propertyName.text === old) bindings++;
    ts.forEachChild(n, countBindings);
  };
  countBindings(fn);
  if (bindings !== 1) return null;
  const spots: { start: number; text: string }[] = [];
  const collect = (n: ts.Node): void => {
    if (ts.isIdentifier(n) && n.text === old) {
      const p = n.parent;
      const isMemberName = (ts.isPropertyAccessExpression(p) && p.name === n) || (ts.isPropertyAssignment(p) && p.name === n) || (ts.isPropertySignature(p) && p.name === n);
      if (!isMemberName) spots.push({ start: n.getStart(sf), text: ts.isShorthandPropertyAssignment(p) ? `${old}: ${renamed}` : renamed });
    }
    ts.forEachChild(n, collect);
  };
  collect(fn);
  if (spots.length < 2) return null;
  let out = src;
  for (const s of spots.sort((x, y) => y.start - x.start)) out = out.slice(0, s.start) + s.text + out.slice(s.start + old.length);
  return out;
};

const HELPERS: ((n: string) => string)[] = [
  (n) => `/** Clamps a value into [min, max]. */\nexport function clamp${n}(value: number, min: number, max: number): number {\n  return Math.min(Math.max(value, min), max);\n}`,
  (n) => `/** Tells whether a text is empty after trimming. */\nexport function isBlank${n}(text: string): boolean {\n  return text.trim() === '';\n}`,
  (n) => `/** Appends an "s" unless the count is exactly one. */\nexport function pluralize${n}(count: number, word: string): string {\n  return count === 1 ? word : \`\${word}s\`;\n}`,
  (n) => `/** Shortens a text to at most \`max\` characters. */\nexport function truncate${n}(text: string, max: number): string {\n  return text.length <= max ? text : \`\${text.slice(0, max - 1)}…\`;\n}`,
  (n) => `/** Sums a list of integers. */\nexport function sum${n}(values: readonly number[]): number {\n  return values.reduce((total, value) => total + value, 0);\n}`,
  (n) => `/** Removes duplicates, keeping the first occurrence. */\nexport function unique${n}<T>(values: readonly T[]): T[] {\n  return [...new Set(values)];\n}`,
  (n) => `/** Whole-number percentage of \`part\` in \`whole\`. */\nexport function percent${n}(part: number, whole: number): number {\n  return whole === 0 ? 0 : Math.round((part / whole) * 100);\n}`,
];

/** Appends a small unused pure helper ("new feature code" that no test reaches). */
const helper: Transform = (src, rnd) => {
  const name = `${pick(WORDS, rnd)}${Math.floor(rnd() * 90 + 10)}`;
  if (src.includes(name)) return null;
  return `${src.replace(/\n*$/, '\n')}\n${pick(HELPERS, rnd)(name)}\n`;
};

// ---------- Transforms (templates, styles, docs) ----------

const ejsNote: Transform = (src, rnd) => `<%# ${pick(NOTES, rnd)} %>\n${src}`;

const cssTweak: Transform = (src, rnd) => {
  const lines = src.split('\n');
  if (rnd() < 0.4) {
    const cls = `.note-${Math.floor(rnd() * 900 + 100)}`;
    return `${src.replace(/\n*$/, '\n')}${cls} { color: #${Math.floor(rnd() * 0xffffff).toString(16).padStart(6, '0')}; font-size: ${12 + Math.floor(rnd() * 4)}px; }\n`;
  }
  const idx = lines.map((l, i) => (/(border-radius|gap): \d+px/.test(l) ? i : -1)).filter((i) => i >= 0);
  if (idx.length === 0) return null;
  const i = pick(idx, rnd);
  const delta = rnd() < 0.5 ? 1 : 2;
  lines[i] = lines[i].replace(/(border-radius|gap): (\d+)px/, (_m, prop: string, n: string) => `${prop}: ${Number(n) + delta}px`);
  return lines.join('\n');
};

const readmeNote: Transform = (src, rnd) => `${src.replace(/\n*$/, '\n')}\n- Note: ${pick(NOTES, rnd).toLowerCase()}.\n`;

// ---------- Registry ----------

interface Gen {
  name: string;
  files: (tree: Tree) => string[];
  transform: Transform;
}

const tsFiles = (tree: Tree, re: RegExp): string[] => [...tree.files.keys()].filter((f) => re.test(f));
const LOGIC = /^src\/modules\/[^/]+\/logic\.ts$/;
const MODULE_TS = /^src\/modules\/[^/]+\/(logic|service|routes)\.ts$/;
const SHARED_TS = /^src\/shared\/(clock|config|errors|money|params|password|session|middleware)\.ts$/;

const GENS: Record<string, Gen> = {
  jsdoc: { name: 'jsdoc', files: (t) => tsFiles(t, MODULE_TS), transform: jsdoc },
  comment: { name: 'comment', files: (t) => tsFiles(t, MODULE_TS), transform: comment },
  rename: { name: 'rename', files: (t) => tsFiles(t, MODULE_TS), transform: renameLocal },
  helper: { name: 'helper', files: (t) => tsFiles(t, LOGIC), transform: helper },
  sharedJsdoc: { name: 'jsdoc', files: (t) => tsFiles(t, SHARED_TS), transform: jsdoc },
  sharedComment: { name: 'comment', files: (t) => tsFiles(t, SHARED_TS), transform: comment },
  sharedRename: { name: 'rename', files: (t) => tsFiles(t, SHARED_TS), transform: renameLocal },
  sharedHelper: { name: 'helper', files: (t) => tsFiles(t, /^src\/shared\/(money|params)\.ts$/), transform: helper },
  ejs: { name: 'ejs-note', files: (t) => tsFiles(t, /\.ejs$/), transform: ejsNote },
  css: { name: 'css', files: () => ['src/public/style.css'], transform: cssTweak },
  readme: { name: 'readme', files: () => ['README.md'], transform: readmeNote },
};

const WEIGHTS: Record<BenignKind, [string, number][]> = {
  feature: [['helper', 5], ['jsdoc', 2], ['comment', 2], ['ejs', 1]],
  refactor: [['rename', 5], ['jsdoc', 2], ['comment', 2], ['sharedRename', 1], ['sharedJsdoc', 1]],
  style: [['css', 6], ['ejs', 4]],
  docs: [['readme', 4], ['jsdoc', 4], ['comment', 2]],
};

function weighted(options: [string, number][], rnd: Rng): string {
  let r = rnd() * options.reduce((s, [, w]) => s + w, 0);
  for (const [name, w] of options) if ((r -= w) < 0) return name;
  return options[0][0];
}

export interface BenignOptions {
  kind: BenignKind;
  /** Force the shared/* variant of the generator (changes in shared code). */
  shared?: boolean;
  /** Restrict the edit to this file (a benign change next to a defect in the same file). */
  onlyFile?: string;
  /**
   * Files with an open or pending defect: a rename there would conflict with the defect's revert (the revert restores
   * text that still uses the old name), so only text-neutral generators are allowed.
   */
  blockRename?: ReadonlySet<string>;
}

/** Picks one generator and file, returns an anchored edit, or null after a few failed attempts. */
export function makeBenign(tree: Tree, rnd: Rng, opt: BenignOptions): Benign | null {
  for (let attempt = 0; attempt < 12; attempt++) {
    let key = weighted(WEIGHTS[opt.kind], rnd);
    if (opt.shared) key = { helper: 'sharedHelper', jsdoc: 'sharedJsdoc', comment: 'sharedComment', rename: 'sharedRename' }[key] ?? key;
    const gen = GENS[key];
    let files = gen.files(tree);
    if (opt.onlyFile) files = files.filter((f) => f === opt.onlyFile);
    if (gen.name === 'rename' && opt.blockRename) files = files.filter((f) => !opt.blockRename!.has(f));
    if (files.length === 0) continue;
    const file = pick(files, rnd);
    const after = gen.transform(tree.get(file), rnd, { module: /^src\/modules\/([^/]+)\//.exec(file)?.[1] ?? null, tree, file });
    if (after === null) continue;
    const edit = diffEdit(tree, file, after);
    if (edit) return { gen: gen.name, edit };
  }
  return null;
}
