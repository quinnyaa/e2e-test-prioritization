import { getDb } from '../../shared/db';
import type { OrderListItem, OrderStatus } from './logic';

export interface OrderItem {
  productId: number;
  name: string;
  unitPriceCents: number;
  quantity: number;
}

export interface OrderDetails {
  id: number;
  userId: number;
  status: OrderStatus;
  subtotalCents: number;
  discountCents: number;
  shippingCents: number;
  totalCents: number;
  promoCode: string | null;
  shippingMethod: string;
  fullName: string;
  address: string;
  city: string;
  postalCode: string;
  phone: string;
  createdAt: string;
  items: OrderItem[];
}

export function listUserOrders(userId: number): OrderListItem[] {
  return getDb()
    .prepare(
      `SELECT o.id, o.status, o.total_cents AS totalCents, o.created_at AS createdAt,
              (SELECT COALESCE(SUM(quantity), 0) FROM order_items WHERE order_id = o.id) AS itemCount
       FROM orders o WHERE o.user_id = ?`,
    )
    .all(userId) as OrderListItem[];
}

export function getOrder(orderId: number): OrderDetails | undefined {
  const db = getDb();
  const order = db
    .prepare(
      `SELECT id, user_id AS userId, status, subtotal_cents AS subtotalCents, discount_cents AS discountCents,
              shipping_cents AS shippingCents, total_cents AS totalCents, promo_code AS promoCode,
              shipping_method AS shippingMethod, full_name AS fullName, address, city,
              postal_code AS postalCode, phone, created_at AS createdAt
       FROM orders WHERE id = ?`,
    )
    .get(orderId) as Omit<OrderDetails, 'items'> | undefined;
  if (!order) return undefined;
  const items = db
    .prepare(
      `SELECT product_id AS productId, product_name AS name, unit_price_cents AS unitPriceCents, quantity
       FROM order_items WHERE order_id = ? ORDER BY rowid`,
    )
    .all(orderId) as OrderItem[];
  return { ...order, items };
}

/** Changes the status; cancelling returns the ordered quantities to stock. */
export function updateOrderStatus(order: OrderDetails, status: OrderStatus): void {
  const db = getDb();
  db.transaction(() => {
    db.prepare('UPDATE orders SET status = ? WHERE id = ?').run(status, order.id);
    if (status === 'cancelled') {
      const restock = db.prepare('UPDATE products SET stock = stock + ? WHERE id = ?');
      for (const item of order.items) restock.run(item.quantity, item.productId);
    }
  })();
}
