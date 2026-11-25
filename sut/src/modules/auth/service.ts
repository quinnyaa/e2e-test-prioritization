import { now } from '../../shared/clock';
import { getDb } from '../../shared/db';
import { hashPassword, verifyPassword } from '../../shared/password';
import type { CurrentUser } from '../../shared/session';
import { normalizeEmail } from './logic';

interface UserRow extends CurrentUser {
  password_hash: string;
}

export function emailExists(email: string): boolean {
  return getDb().prepare('SELECT 1 FROM users WHERE email = ?').get(normalizeEmail(email)) !== undefined;
}

export function createUser(name: string, email: string, password: string): number {
  const result = getDb()
    .prepare('INSERT INTO users (email, name, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)')
    .run(normalizeEmail(email), name.trim(), hashPassword(password), 'customer', now().toISOString());
  return Number(result.lastInsertRowid);
}

export function authenticate(email: string, password: string): CurrentUser | null {
  const row = getDb()
    .prepare('SELECT id, email, name, role, password_hash FROM users WHERE email = ?')
    .get(normalizeEmail(email)) as UserRow | undefined;
  if (!row || !verifyPassword(password, row.password_hash)) return null;
  return { id: row.id, email: row.email, name: row.name, role: row.role };
}
