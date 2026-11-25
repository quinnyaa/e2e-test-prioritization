import { TABLES_IN_DELETE_ORDER, type Db } from '../shared/db';
import { hashPassword } from '../shared/password';
import { categories, orders, products, promoCodes, SEED_CREATED_AT, users } from './data';

// Hashing is slow on purpose, so hashes are computed once per process and reused by every reset.
const passwordHashes = new Map<string, string>();

function seedHash(password: string, salt: string): string {
  const key = `${salt}:${password}`;
  let hash = passwordHashes.get(key);
  if (!hash) {
    hash = hashPassword(password, salt);
    passwordHashes.set(key, hash);
  }
  return hash;
}

/** Wipes all tables (including sessions and carts) and restores the seed state. */
export function resetDatabase(db: Db): void {
  const categoryIds = new Map(categories.map((c) => [c.slug, c.id]));

  db.transaction(() => {
    for (const table of TABLES_IN_DELETE_ORDER) db.prepare(`DELETE FROM ${table}`).run();

    const insertUser = db.prepare(
      'INSERT INTO users (id, email, name, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?)',
    );
    for (const u of users) {
      insertUser.run(u.id, u.email, u.name, seedHash(u.password, u.salt), u.role, SEED_CREATED_AT);
    }

    const insertCategory = db.prepare('INSERT INTO categories (id, slug, name) VALUES (?, ?, ?)');
    for (const c of categories) insertCategory.run(c.id, c.slug, c.name);

    const insertProduct = db.prepare(
      'INSERT INTO products (id, category_id, name, description, price_cents, stock) VALUES (?, ?, ?, ?, ?, ?)',
    );
    for (const p of products) {
      insertProduct.run(p.id, categoryIds.get(p.category), p.name, p.description, p.priceCents, p.stock);
    }

    const insertPromo = db.prepare(
      'INSERT INTO promo_codes (code, type, value, min_order_cents, expires_at, active) VALUES (?, ?, ?, ?, ?, ?)',
    );
    for (const p of promoCodes) insertPromo.run(p.code, p.type, p.value, p.minOrderCents, p.expiresAt, p.active);

    const insertOrder = db.prepare(`
      INSERT INTO orders (id, user_id, status, subtotal_cents, discount_cents, shipping_cents, total_cents,
        promo_code, shipping_method, full_name, address, city, postal_code, phone, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);
    const insertItem = db.prepare(
      'INSERT INTO order_items (order_id, product_id, product_name, unit_price_cents, quantity) VALUES (?, ?, ?, ?, ?)',
    );
    for (const o of orders) {
      insertOrder.run(
        o.id, o.userId, o.status, o.subtotalCents, o.discountCents, o.shippingCents, o.totalCents,
        o.promoCode, o.shippingMethod, o.fullName, o.address, o.city, o.postalCode, o.phone, o.createdAt,
      );
      for (const item of o.items) insertItem.run(o.id, item.productId, item.name, item.unitPriceCents, item.quantity);
    }
  })();
}
