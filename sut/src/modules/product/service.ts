import { getDb } from '../../shared/db';

export interface ProductDetails {
  id: number;
  name: string;
  description: string;
  priceCents: number;
  stock: number;
  categorySlug: string;
  categoryName: string;
}

export function getProduct(id: number): ProductDetails | undefined {
  return getDb()
    .prepare(
      `SELECT p.id, p.name, p.description, p.price_cents AS priceCents, p.stock,
              c.slug AS categorySlug, c.name AS categoryName
       FROM products p JOIN categories c ON c.id = p.category_id
       WHERE p.id = ?`,
    )
    .get(id) as ProductDetails | undefined;
}
