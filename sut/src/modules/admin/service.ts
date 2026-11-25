import { getDb } from '../../shared/db';
import type { OrderStatus } from '../orders/logic';
import type { ProductValues } from './logic';

export interface AdminProduct {
  id: number;
  name: string;
  description: string;
  priceCents: number;
  stock: number;
  categorySlug: string;
  categoryName: string;
}

export interface AdminOrder {
  id: number;
  status: OrderStatus;
  totalCents: number;
  createdAt: string;
  customerEmail: string;
}

const PRODUCT_COLUMNS = `p.id, p.name, p.description, p.price_cents AS priceCents, p.stock,
  c.slug AS categorySlug, c.name AS categoryName`;

export function listAdminProducts(): AdminProduct[] {
  return getDb()
    .prepare(`SELECT ${PRODUCT_COLUMNS} FROM products p JOIN categories c ON c.id = p.category_id ORDER BY p.id`)
    .all() as AdminProduct[];
}

export function getAdminProduct(id: number): AdminProduct | undefined {
  return getDb()
    .prepare(`SELECT ${PRODUCT_COLUMNS} FROM products p JOIN categories c ON c.id = p.category_id WHERE p.id = ?`)
    .get(id) as AdminProduct | undefined;
}

export function createProduct(values: ProductValues): number {
  const result = getDb()
    .prepare(
      `INSERT INTO products (category_id, name, description, price_cents, stock)
       VALUES ((SELECT id FROM categories WHERE slug = ?), ?, ?, ?, ?)`,
    )
    .run(values.categorySlug, values.name, values.description, values.priceCents, values.stock);
  return Number(result.lastInsertRowid);
}

export function updateProduct(id: number, values: ProductValues): void {
  getDb()
    .prepare(
      `UPDATE products SET category_id = (SELECT id FROM categories WHERE slug = ?),
         name = ?, description = ?, price_cents = ?, stock = ?
       WHERE id = ?`,
    )
    .run(values.categorySlug, values.name, values.description, values.priceCents, values.stock, id);
}

export function deleteProduct(id: number): void {
  getDb().prepare('DELETE FROM products WHERE id = ?').run(id);
}

export function listAllOrders(): AdminOrder[] {
  return getDb()
    .prepare(
      `SELECT o.id, o.status, o.total_cents AS totalCents, o.created_at AS createdAt, u.email AS customerEmail
       FROM orders o JOIN users u ON u.id = o.user_id
       ORDER BY o.id DESC`,
    )
    .all() as AdminOrder[];
}
