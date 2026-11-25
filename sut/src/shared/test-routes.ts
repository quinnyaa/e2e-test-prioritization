import { Router } from 'express';
import { resetDatabase } from '../seed/reset';
import { getDb } from './db';

/** Endpoints for E2E test isolation. Mounted only when NODE_ENV=test. */
export const testRouter = Router();

testRouter.post('/__test__/reset', (_req, res) => {
  resetDatabase(getDb());
  res.json({ status: 'reset' });
});
