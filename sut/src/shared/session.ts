import crypto from 'node:crypto';
import type { Request, RequestHandler, Response } from 'express';
import { now } from './clock';
import { getDb } from './db';

export const SESSION_COOKIE = 'sid';

export type Role = 'customer' | 'admin';

export interface CurrentUser {
  id: number;
  email: string;
  name: string;
  role: Role;
}

export interface Flash {
  type: 'success' | 'error';
  message: string;
}

declare global {
  namespace Express {
    interface Request {
      sessionId: string;
      user: CurrentUser | null;
    }
  }
}

interface SessionRow {
  id: string;
  user_id: number | null;
  flash: string | null;
}

export function parseCookies(header: string | undefined): Record<string, string> {
  const cookies: Record<string, string> = {};
  if (!header) return cookies;
  for (const part of header.split(';')) {
    const index = part.indexOf('=');
    if (index < 0) continue;
    const name = part.slice(0, index).trim();
    const value = part.slice(index + 1).trim();
    if (name) cookies[name] = decodeURIComponent(value);
  }
  return cookies;
}

function createSession(res: Response): string {
  const id = crypto.randomBytes(24).toString('hex');
  getDb()
    .prepare('INSERT INTO sessions (id, user_id, flash, created_at) VALUES (?, NULL, NULL, ?)')
    .run(id, now().toISOString());
  res.cookie(SESSION_COOKIE, id, { httpOnly: true, sameSite: 'lax', path: '/' });
  return id;
}

/** Loads (or creates) the cookie-based session, the logged-in user and the pending flash message. */
export const sessionMiddleware: RequestHandler = (req, res, next) => {
  const db = getDb();
  const cookieId = parseCookies(req.headers.cookie)[SESSION_COOKIE];
  let session = cookieId
    ? (db.prepare('SELECT id, user_id, flash FROM sessions WHERE id = ?').get(cookieId) as SessionRow | undefined)
    : undefined;

  if (!session) session = { id: createSession(res), user_id: null, flash: null };

  req.sessionId = session.id;
  req.user = session.user_id
    ? ((db.prepare('SELECT id, email, name, role FROM users WHERE id = ?').get(session.user_id) as
        | CurrentUser
        | undefined) ?? null)
    : null;

  if (session.flash) {
    res.locals.flash = JSON.parse(session.flash) as Flash;
    db.prepare('UPDATE sessions SET flash = NULL WHERE id = ?').run(session.id);
  }
  res.locals.currentUser = req.user;
  next();
};

export function setFlash(req: Request, type: Flash['type'], message: string): void {
  const flash: Flash = { type, message };
  getDb().prepare('UPDATE sessions SET flash = ? WHERE id = ?').run(JSON.stringify(flash), req.sessionId);
}

export function setSessionUser(req: Request, userId: number | null): void {
  getDb().prepare('UPDATE sessions SET user_id = ? WHERE id = ?').run(userId, req.sessionId);
}

/** Drops the current session (and its cart) and starts a fresh anonymous one. */
export function startNewSession(req: Request, res: Response): void {
  const db = getDb();
  db.prepare('DELETE FROM sessions WHERE id = ?').run(req.sessionId);
  req.sessionId = createSession(res);
  req.user = null;
  res.locals.currentUser = null;
}
