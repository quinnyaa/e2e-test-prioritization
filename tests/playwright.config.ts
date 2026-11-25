import { defineConfig, devices } from '@playwright/test';
import { RAW_REPORT, WORKERS } from './helpers/env';

export default defineConfig({
  testDir: './specs',
  globalSetup: './global-setup.ts',
  fullyParallel: true,
  workers: WORKERS,
  // Raw results are the dataset labels: never retry, flakiness is handled by `npm run rerun-failed`.
  retries: 0,
  forbidOnly: true,
  // Short limits so that "hanging" mutants turn into timedOut/failed results quickly.
  timeout: 15_000,
  expect: { timeout: 3_000 },
  globalTimeout: 10 * 60_000,
  reporter: [[process.env.PW_REPORTER ?? 'list'], ['json', { outputFile: RAW_REPORT }]],
  use: {
    ...devices['Desktop Chrome'],
    actionTimeout: 5_000,
    navigationTimeout: 5_000,
    trace: (process.env.PW_TRACE as 'off' | 'retain-on-failure' | undefined) ?? 'off',
    screenshot: 'off',
    video: 'off',
  },
  projects: [{ name: 'chromium' }],
});
