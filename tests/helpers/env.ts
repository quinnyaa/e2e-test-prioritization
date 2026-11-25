// Run settings shared by the Playwright config, fixtures and scripts. All of them come from env vars.
import path from 'node:path';

export const TESTS_DIR = path.resolve(__dirname, '..');
export const SUT_DIR = path.resolve(TESTS_DIR, process.env.SUT_DIR ?? '../sut');
export const RESULTS_DIR = path.join(TESTS_DIR, 'results');
export const RAW_REPORT = path.resolve(TESTS_DIR, process.env.RAW_REPORT ?? 'results/raw.json');
export const BUILD_STATUS_FILE = path.join(RESULTS_DIR, 'build-status.json');
export const MANIFEST_FILE = path.join(TESTS_DIR, 'manifest.json');
export const RUN_FILE = path.join(RESULTS_DIR, 'run.json');

export const WORKERS = Number(process.env.WORKERS ?? 4);
export const BASE_PORT = Number(process.env.BASE_PORT ?? 4100);

/** Exit code of a run whose SUT commit does not build (as opposed to 1 = test failures). */
export const BUILD_FAILED_EXIT_CODE = 3;
