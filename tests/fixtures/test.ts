// Extended `test` used by every spec: per-worker SUT server, DB reset before each test, login helpers.
import { test as base, type APIRequestContext, type Page } from '@playwright/test';
import { login } from '../helpers/api';
import type { User } from '../helpers/data';
import { startSutServer, type SutServer } from './sut-server';

interface TestFixtures {
  /** Auto fixture: restores the seed state before every test, so tests never share data. */
  resetDb: void;
  /** Logs the test's browser context in via HTTP (no UI); the session cookie lands in the context. */
  loginAs: (user: User) => Promise<void>;
  /** A separate HTTP client with its own session, e.g. an admin acting while the customer browses. */
  apiAs: (user: User) => Promise<APIRequestContext>;
  /** A separate logged-in browser page with its own cookies, e.g. a second actor in an E2E journey. */
  pageAs: (user: User) => Promise<Page>;
}

interface WorkerFixtures {
  sutServer: SutServer;
  /**
   * Auto worker fixture: loads one page before the first test, so that the first test of a worker is not
   * charged with the browser's first-page cost. Keeps per-test durations (an ML feature) comparable.
   */
  warmUp: void;
}

export const test = base.extend<TestFixtures, WorkerFixtures>({
  sutServer: [
    async ({}, use, workerInfo) => {
      const server = await startSutServer(workerInfo.workerIndex);
      await use(server);
      await server.stop();
    },
    { scope: 'worker', timeout: 30_000 },
  ],

  warmUp: [
    async ({ sutServer, browser }, use) => {
      const context = await browser.newContext({ baseURL: sutServer.url });
      const page = await context.newPage();
      await page.goto('/catalog');
      await context.close();
      await use();
    },
    { scope: 'worker', auto: true },
  ],

  baseURL: async ({ sutServer }, use) => {
    await use(sutServer.url);
  },

  resetDb: [
    async ({ sutServer }, use) => {
      const res = await fetch(`${sutServer.url}/__test__/reset`, {
        method: 'POST',
        signal: AbortSignal.timeout(3_000),
      });
      if (!res.ok) throw new Error(`SUT reset failed: HTTP ${res.status}`);
      await use();
    },
    { auto: true },
  ],

  // Depends on resetDb explicitly: the login session must be created after the reset wipes sessions.
  loginAs: async ({ context, resetDb }, use) => {
    void resetDb;
    await use((user) => login(context.request, user));
  },

  apiAs: async ({ playwright, baseURL, resetDb }, use) => {
    void resetDb;
    const clients: APIRequestContext[] = [];
    await use(async (user) => {
      const client = await playwright.request.newContext({ baseURL });
      clients.push(client);
      await login(client, user);
      return client;
    });
    await Promise.all(clients.map((c) => c.dispose()));
  },

  pageAs: async ({ browser, baseURL, resetDb }, use) => {
    void resetDb;
    const contexts: Awaited<ReturnType<typeof browser.newContext>>[] = [];
    await use(async (user) => {
      const context = await browser.newContext({ baseURL });
      contexts.push(context);
      await login(context.request, user);
      return context.newPage();
    });
    await Promise.all(contexts.map((c) => c.close()));
  },
});

export { expect } from '@playwright/test';
