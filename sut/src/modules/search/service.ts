import { getDb } from '../../shared/db';

export interface SearchProduct {
  id: number;
  name: string;
  priceCents: number;
  stock: number;
  categoryName: string;
}

export function listSearchableProducts(): SearchProduct[] {
  return getDb()
    .prepare(
      `SELECT p.id, p.name, p.price_cents AS priceCents, p.stock, c.name AS categoryName
       FROM products p JOIN categories c ON c.id = p.category_id
       ORDER BY p.id`,
    )
    .all() as SearchProduct[];
}
