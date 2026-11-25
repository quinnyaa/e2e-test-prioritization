import { getDb } from '../../shared/db';
import type { CartLine } from '../cart/logic';
import { INITIAL_ORDER_STATUS, type OrderStatus } from '../orders/logic';
import {
  computeStockUpdates,
  normalizePhone,
  type PromoCode,
  type ShippingForm,
  type ShippingMethod,
  type Totals,
} from './logic';

interface PromoRow {
  code: string;
  type: PromoCode['type'];
  value: number;
  min_order_cents: number;
  expires_at: string | null;
  active: number;
}

export interface NewOrder {
  userId: number;
  sessionId: string;
  lines: CartLine[];
  totals: Totals;
  promoCode: string | null;
  method: ShippingMethod;
  shipping: ShippingForm;
  createdAt: string;
}

export interface OrderSummary {
  id: number;
  totalCents: number;
  status: OrderStatus;
}

export class StockChangedError extends Error {}

export function getPromo(code: string): PromoCode | undefined {
  const row = getDb().prepare('SELECT * FROM promo_codes WHERE code = ?').get(code) as PromoRow | undefined;
  if (!row) return undefined;
  return {
    code: row.code,
    type: row.type,
    value: row.value,
    minOrderCents: row.min_order_cents,
    expiresAt: row.expires_at,
    active: row.active === 1,
  };
}

/** Creates the order, decrements stock and empties the cart in one transaction. */
export function placeOrder(order: NewOrder): number {
  const db = getDb();
  return db.transaction(() => {
    const updates = computeStockUpdates(order.lines);
    if (!updates) throw new StockChangedError('Not enough stock');
    const updateStock = db.prepare('UPDATE products SET stock = ? WHERE id = ? AND stock = ?');
    for (const u of updates) {
      if (updateStock.run(u.to, u.productId, u.from).changes !== 1) throw new StockChangedError('Stock changed');
    }

    const result = db
      .prepare(
        `INSERT INTO orders (user_id, status, subtotal_cents, discount_cents, shipping_cents, total_cents,
           promo_code, shipping_method, full_name, address, city, postal_code, phone, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        order.userId,
        INITIAL_ORDER_STATUS,
        order.totals.subtotalCents,
        order.totals.discountCents,
        order.totals.shippingCents,
        order.totals.totalCents,
        order.promoCode,
        order.method,
        order.shipping.fullName.trim(),
        order.shipping.address.trim(),
        order.shipping.city.trim(),
        order.shipping.postalCode.trim(),
        normalizePhone(order.shipping.phone),
        order.createdAt,
      );
    const orderId = Number(result.lastInsertRowid);

    const insertItem = db.prepare(
      'INSERT INTO order_items (order_id, product_id, product_name, unit_price_cents, quantity) VALUES (?, ?, ?, ?, ?)',
    );
    for (const line of order.lines) insertItem.run(orderId, line.productId, line.name, line.priceCents, line.quantity);

    db.prepare('DELETE FROM cart_items WHERE session_id = ?').run(order.sessionId);
    return orderId;
  })();
}

export function getOrderSummary(orderId: number, userId: number): OrderSummary | undefined {
  return getDb()
    .prepare('SELECT id, total_cents AS totalCents, status FROM orders WHERE id = ? AND user_id = ?')
    .get(orderId, userId) as OrderSummary | undefined;
}
