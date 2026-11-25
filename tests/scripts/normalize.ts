// npm run normalize — turns the raw Playwright JSON report into results/run.json (one dataset row per test).
import fs from 'node:fs';
import { BUILD_STATUS_FILE, RAW_REPORT, RUN_FILE, SUT_DIR } from '../helpers/env';
import { errorSnippet, normalizeStatus, readManifest, readReport, reportedTests, sutCommit, writeJson, type RunStatus } from './lib';

export interface RunTest {
  id: string;
  status: RunStatus;
  durationMs: number;
  errorSnippet: string | null;
  flaky: boolean;
  confirmedFail: boolean;
  /** Statuses of the extra executions made by `npm run rerun-failed` (empty if not re-run). */
  rerunStatuses: RunStatus[];
}

export interface Run {
  runId: string;
  sutCommit: string | null;
  sutDir: string;
  startedAt: string;
  buildOk: boolean;
  buildError: string | null;
  totalDurationMs: number;
  tests: RunTest[];
}

const build = fs.existsSync(BUILD_STATUS_FILE)
  ? (JSON.parse(fs.readFileSync(BUILD_STATUS_FILE, 'utf8')) as { buildOk: boolean; output: string; sutDir: string })
  : null;
if (build && build.sutDir !== SUT_DIR) {
  throw new Error(`Build status belongs to ${build.sutDir}, but SUT_DIR is ${SUT_DIR}. Run the tests first.`);
}
const buildOk = build?.buildOk ?? true;
const report = buildOk ? readReport(RAW_REPORT) : null;
const startedAt = report?.stats.startTime ?? new Date().toISOString();
const commit = sutCommit();

const tests: RunTest[] = report
  ? reportedTests(report).map((t) => {
      const result = t.results[t.results.length - 1];
      return {
        id: t.id,
        status: normalizeStatus(result?.status ?? 'skipped'),
        durationMs: Math.round(result?.duration ?? 0),
        errorSnippet: errorSnippet(result),
        flaky: false,
        confirmedFail: false,
        rerunStatuses: [],
      };
    })
  : [];
tests.sort((a, b) => a.id.localeCompare(b.id, 'en', { numeric: true }));

const run: Run = {
  runId: process.env.RUN_ID ?? `${startedAt.replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z')}-${commit?.slice(0, 7) ?? 'nogit'}`,
  sutCommit: commit,
  sutDir: SUT_DIR,
  startedAt,
  buildOk,
  buildError: buildOk ? null : (build?.output.slice(0, 1000) ?? null),
  totalDurationMs: Math.round(report?.stats.duration ?? 0),
  tests,
};
writeJson(RUN_FILE, run);

const manifestIds = readManifest()?.tests.map((t) => t.id) ?? [];
const missing = manifestIds.filter((id) => !tests.some((t) => t.id === id));
if (buildOk && missing.length > 0) console.warn(`warning: ${missing.length} manifest tests absent from the run: ${missing.join(', ')}`);
if (report && report.errors.length > 0) console.warn(`warning: the report has ${report.errors.length} global error(s)`);

const by = (s: RunStatus) => tests.filter((t) => t.status === s).length;
console.log(
  `run.json: ${run.runId}, buildOk=${buildOk}, ${tests.length} tests ` +
    `(passed ${by('passed')}, failed ${by('failed')}, timedOut ${by('timedOut')}, skipped ${by('skipped')}), ` +
    `${run.totalDurationMs} ms`,
);
