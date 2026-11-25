// Starts one SUT instance per Playwright worker: own port, own SQLite file, NODE_ENV=test.
import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { BASE_PORT, SUT_DIR } from '../helpers/env';

const STARTUP_TIMEOUT_MS = 10_000;
const HEALTH_POLL_MS = 50;

export interface SutServer {
  url: string;
  port: number;
  stop(): Promise<void>;
}

async function isHealthy(url: string): Promise<boolean> {
  try {
    const res = await fetch(`${url}/health`, { signal: AbortSignal.timeout(1_000) });
    return res.ok;
  } catch {
    return false;
  }
}

function killHard(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
  return new Promise((resolve) => {
    child.once('exit', () => resolve());
    child.kill('SIGKILL');
  });
}

export async function startSutServer(workerIndex: number): Promise<SutServer> {
  const port = BASE_PORT + workerIndex;
  const url = `http://127.0.0.1:${port}`;
  if (await isHealthy(url)) {
    throw new Error(`Port ${port} is already taken by another server (a leftover SUT instance?)`);
  }

  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), `sut-w${workerIndex}-`));
  const child = spawn(process.execPath, ['dist/server.js'], {
    cwd: SUT_DIR,
    env: { ...process.env, NODE_ENV: 'test', PORT: String(port), DB_PATH: path.join(dataDir, 'shop.db') },
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  let stderr = '';
  child.stderr?.on('data', (chunk: Buffer) => {
    stderr = (stderr + chunk.toString()).slice(-2_000);
  });
  // Safety net: never leave a SUT process behind if the worker dies without fixture teardown.
  const onExit = () => child.kill('SIGKILL');
  process.once('exit', onExit);

  const stop = async () => {
    process.removeListener('exit', onExit);
    await killHard(child);
    fs.rmSync(dataDir, { recursive: true, force: true });
  };

  const deadline = Date.now() + STARTUP_TIMEOUT_MS;
  while (!(await isHealthy(url))) {
    if (child.exitCode !== null || Date.now() > deadline) {
      await stop();
      const reason = child.exitCode !== null ? `exited with code ${child.exitCode}` : 'did not become healthy';
      throw new Error(`SUT server on port ${port} ${reason}.\n${stderr}`);
    }
    await new Promise((resolve) => setTimeout(resolve, HEALTH_POLL_MS));
  }
  return { url, port, stop };
}
