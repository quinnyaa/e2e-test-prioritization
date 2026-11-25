import { getDb } from '../../shared/db';
import type { CatalogProduct } from './logic';

export interface Category {
  slug: string;
  name: string;
}

export function listCategories(): Category[] {
  return getDb().prepare('SELECT slug, name FROM categories ORDER BY id').all() as Category[];
}

export function listProducts(): CatalogProduct[] {
  return getDb()
    .prepare(
      `SELECT p.id, p.name, p.price_cents AS priceCents, p.stock,
              c.slug AS categorySlug, c.name AS categoryName
       FROM products p JOIN categories c ON c.id = p.category_id
       ORDER BY p.id`,
    )
    .all() as CatalogProduct[];
}
