// npm run rerun-failed — re-runs the failed tests of results/run.json two more times and marks them
// flaky (passed at least once) or confirmedFail (failed all three times). Config retries stay at 0.
import fs from 'node:fs';
import path from 'node:path';
import { RESULTS_DIR, RUN_FILE } from '../helpers/env';
import { grepForIds, normalizeStatus, readReport, reportedTests, runPlaywright, writeJson } from './lib';
import type { Run } from './normalize';

const RERUNS = 2;
const rerunReport = path.join(RESULTS_DIR, 'rerun-raw.json');

const run = JSON.parse(fs.readFileSync(RUN_FILE, 'utf8')) as Run;
const failed = run.tests.filter((t) => t.status === 'failed' || t.status === 'timedOut');
if (failed.length === 0) {
  console.log('rerun-failed: no failed tests in run.json, nothing to do');
  process.exit(0);
}

const ids = failed.map((t) => t.id);
console.log(`rerun-failed: ${ids.join(', ')} × ${RERUNS}`);
fs.rmSync(rerunReport, { force: true });
runPlaywright(['--grep', grepForIds(ids), `--repeat-each=${RERUNS}`], { RAW_REPORT: rerunReport, PW_REPORTER: 'dot' });
if (!fs.existsSync(rerunReport)) throw new Error('The re-run produced no report');

const statuses = new Map(reportedTests(readReport(rerunReport)).map((t) => [t.id, t.results.map((r) => normalizeStatus(r.status))]));
for (const test of failed) {
  test.rerunStatuses = statuses.get(test.id) ?? [];
  test.flaky = test.rerunStatuses.includes('passed');
  test.confirmedFail = test.rerunStatuses.length === RERUNS && !test.flaky;
}
writeJson(RUN_FILE, run);

for (const t of failed) {
  const verdict = t.confirmedFail ? 'confirmedFail' : t.flaky ? 'flaky' : 'inconclusive';
  console.log(`  ${t.id}: ${t.status} → [${t.rerunStatuses.join(', ')}] → ${verdict}`);
}
