// npm run stability — runs the full suite N times (default 10) on the current SUT, reports failures and
// per-test duration statistics, and stores avgDurationMs / stdDurationMs in manifest.json.
// Options: --runs N, --no-manifest (do not touch manifest.json, e.g. for a 1-worker timing check).
import fs from 'node:fs';
import path from 'node:path';
import { MANIFEST_FILE, RESULTS_DIR, WORKERS } from '../helpers/env';
import { buildManifest, mean, normalizeStatus, readManifest, readReport, reportedTests, runPlaywright, std, writeJson } from './lib';

const args = process.argv.slice(2);
const runsArg = args.indexOf('--runs');
const runs = Number(runsArg >= 0 ? args[runsArg + 1] : (process.env.STABILITY_RUNS ?? 10));
const updateManifest = !args.includes('--no-manifest');
const dir = path.join(RESULTS_DIR, `stability-w${WORKERS}`);
fs.rmSync(dir, { recursive: true, force: true });

const perTest = new Map<string, { durations: number[]; failures: number; statuses: string[] }>();
const runSummaries: Array<{ run: number; exitCode: number; totalDurationMs: number; wallMs: number; failed: number }> = [];

for (let i = 1; i <= runs; i++) {
  const file = path.join(dir, `run-${String(i).padStart(2, '0')}.json`);
  console.log(`\n=== stability run ${i}/${runs} (workers=${WORKERS}) ===`);
  const { exitCode, wallMs } = runPlaywright([], { RAW_REPORT: file, PW_REPORTER: 'dot' });
  const report = readReport(file);
  let failed = 0;
  for (const t of reportedTests(report)) {
    const result = t.results[0];
    const status = normalizeStatus(result?.status);
    const entry = perTest.get(t.id) ?? { durations: [], failures: 0, statuses: [] };
    entry.durations.push(result?.duration ?? 0);
    entry.statuses.push(status);
    if (status !== 'passed') {
      entry.failures++;
      failed++;
    }
    perTest.set(t.id, entry);
  }
  runSummaries.push({ run: i, exitCode, totalDurationMs: Math.round(report.stats.duration), wallMs, failed });
}

const tests = [...perTest.entries()]
  .map(([id, e]) => ({
    id,
    runs: e.durations.length,
    failures: e.failures,
    avgDurationMs: Math.round(mean(e.durations)),
    stdDurationMs: Math.round(std(e.durations)),
    minDurationMs: Math.round(Math.min(...e.durations)),
    maxDurationMs: Math.round(Math.max(...e.durations)),
  }))
  .sort((a, b) => a.id.localeCompare(b.id, 'en', { numeric: true }));
const totals = runSummaries.map((r) => r.totalDurationMs);
const summary = {
  measuredAt: new Date().toISOString(),
  workers: WORKERS,
  runs,
  totalFailures: tests.reduce((s, t) => s + t.failures, 0),
  meanRunDurationMs: Math.round(mean(totals)),
  stdRunDurationMs: Math.round(std(totals)),
  meanWallMs: Math.round(mean(runSummaries.map((r) => r.wallMs))),
  runsDetail: runSummaries,
  tests,
};
writeJson(path.join(dir, 'summary.json'), summary);

console.log(`\nStability: ${runs} runs, workers=${WORKERS}`);
console.log('id         fails  avg ms  std ms  min..max');
for (const t of tests) {
  console.log(
    `${t.id.padEnd(10)} ${String(t.failures).padStart(5)}  ${String(t.avgDurationMs).padStart(6)}  ${String(t.stdDurationMs).padStart(6)}  ${t.minDurationMs}..${t.maxDurationMs}`,
  );
}
console.log(
  `full run: mean ${summary.meanRunDurationMs} ms ± ${summary.stdRunDurationMs} (Playwright), ` +
    `mean wall time incl. build and browser start ${summary.meanWallMs} ms`,
);
console.log(`failures: ${summary.totalFailures}`);

if (updateManifest) {
  const manifest = readManifest() ?? buildManifest();
  const stats = new Map(tests.map((t) => [t.id, t]));
  for (const t of manifest.tests) {
    t.avgDurationMs = stats.get(t.id)?.avgDurationMs ?? null;
    t.stdDurationMs = stats.get(t.id)?.stdDurationMs ?? null;
  }
  manifest.durations = { runs, workers: WORKERS, measuredAt: summary.measuredAt };
  writeJson(MANIFEST_FILE, manifest);
  console.log('manifest.json: avgDurationMs / stdDurationMs updated');
}

process.exit(summary.totalFailures === 0 ? 0 : 1);
