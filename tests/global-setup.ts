// Builds the SUT once per run. A commit that does not build ends the run with BUILD_FAILED_EXIT_CODE.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { BUILD_FAILED_EXIT_CODE, BUILD_STATUS_FILE, SUT_DIR } from './helpers/env';

function writeStatus(status: { buildOk: boolean; durationMs: number; output: string }): void {
  fs.mkdirSync(path.dirname(BUILD_STATUS_FILE), { recursive: true });
  fs.writeFileSync(BUILD_STATUS_FILE, JSON.stringify({ sutDir: SUT_DIR, ...status }, null, 2));
}

export default function globalSetup(): void {
  const started = Date.now();
  const result = spawnSync('npm', ['run', 'build'], { cwd: SUT_DIR, encoding: 'utf8', timeout: 120_000 });
  const output = `${result.stdout ?? ''}${result.stderr ?? ''}${result.error ? String(result.error) : ''}`;
  const buildOk = result.status === 0 && fs.existsSync(path.join(SUT_DIR, 'dist', 'server.js'));
  writeStatus({ buildOk, durationMs: Date.now() - started, output: output.slice(0, 4000) });

  if (!buildOk) {
    console.error(`\nSUT build failed in ${SUT_DIR} (commit does not build):\n${output.slice(0, 2000)}`);
    process.exit(BUILD_FAILED_EXIT_CODE);
  }
}
