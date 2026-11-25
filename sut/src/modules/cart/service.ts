import { getDb } from '../../shared/db';
import type { CartLine } from './logic';

export interface StockInfo {
  id: number;
  name: string;
  stock: number;
}

export function getCartLines(sessionId: string): CartLine[] {
  return getDb()
    .prepare(
      `SELECT ci.product_id AS productId, p.name, p.price_cents AS priceCents, ci.quantity, p.stock
       FROM cart_items ci JOIN products p ON p.id = ci.product_id
       WHERE ci.session_id = ?
       ORDER BY ci.rowid`,
    )
    .all(sessionId) as CartLine[];
}

export function getQuantityInCart(sessionId: string, productId: number): number {
  const row = getDb()
    .prepare('SELECT quantity FROM cart_items WHERE session_id = ? AND product_id = ?')
    .get(sessionId, productId) as { quantity: number } | undefined;
  return row?.quantity ?? 0;
}

export function getStockInfo(productId: number): StockInfo | undefined {
  return getDb().prepare('SELECT id, name, stock FROM products WHERE id = ?').get(productId) as StockInfo | undefined;
}

export function setLineQuantity(sessionId: string, productId: number, quantity: number): void {
  getDb()
    .prepare(
      `INSERT INTO cart_items (session_id, product_id, quantity) VALUES (?, ?, ?)
       ON CONFLICT (session_id, product_id) DO UPDATE SET quantity = excluded.quantity`,
    )
    .run(sessionId, productId, quantity);
}

export function removeLine(sessionId: string, productId: number): void {
  getDb().prepare('DELETE FROM cart_items WHERE session_id = ? AND product_id = ?').run(sessionId, productId);
}

export function clearCart(sessionId: string): void {
  getDb().prepare('DELETE FROM cart_items WHERE session_id = ?').run(sessionId);
}
