import { describe, expect, it } from 'vitest';
import {
  catalogQueryString,
  filterProducts,
  paginate,
  parseCatalogQuery,
  sortProducts,
  type CatalogProduct,
} from '../../src/modules/catalog/logic';

const product = (id: number, name: string, priceCents: number, stock: number, categorySlug: string): CatalogProduct => ({
  id,
  name,
  priceCents,
  stock,
  categorySlug,
  categoryName: categorySlug,
});

const products = [
  product(1, 'Banana', 300, 5, 'food'),
  product(2, 'apple', 500, 0, 'food'),
  product(3, 'Cable', 500, 2, 'tech'),
  product(4, 'Drone', 9900, 1, 'tech'),
];

describe('catalog logic', () => {
  it('parses filters and rejects an inverted price range', () => {
    const { filters, errors } = parseCatalogQuery({ category: 'tech', min: '5', max: '99', sort: 'price_desc', page: '2' }, ['tech']);
    expect(filters).toEqual({
      category: 'tech',
      minPriceCents: 500,
      maxPriceCents: 9900,
      inStockOnly: false,
      sort: 'price_desc',
      page: 2,
    });
    expect(errors).toEqual([]);

    const inverted = parseCatalogQuery({ min: '10', max: '5' }, []);
    expect(inverted.errors).toHaveLength(1);
    expect(inverted.filters.minPriceCents).toBeNull();
  });

  it('falls back to defaults for unknown values', () => {
    const { filters } = parseCatalogQuery({ category: 'nope', sort: 'random', page: '-3' }, ['tech']);
    expect(filters.category).toBeNull();
    expect(filters.sort).toBe('name_asc');
    expect(filters.page).toBe(1);
  });

  it('filters by inclusive price range, category and stock', () => {
    const base = parseCatalogQuery({}, []).filters;
    expect(filterProducts(products, { ...base, minPriceCents: 500, maxPriceCents: 500 }).map((p) => p.id)).toEqual([2, 3]);
    expect(filterProducts(products, { ...base, category: 'food', inStockOnly: true }).map((p) => p.id)).toEqual([1]);
  });

  it('sorts by name and by price with name as a tie-breaker', () => {
    expect(sortProducts(products, 'name_asc').map((p) => p.id)).toEqual([2, 1, 3, 4]);
    expect(sortProducts(products, 'price_asc').map((p) => p.id)).toEqual([1, 2, 3, 4]);
    expect(sortProducts(products, 'price_desc').map((p) => p.id)).toEqual([4, 2, 3, 1]);
  });

  it('paginates and clamps the page number', () => {
    const items = Array.from({ length: 17 }, (_, i) => i);
    expect(paginate(items, 1, 8)).toMatchObject({ items: [0, 1, 2, 3, 4, 5, 6, 7], totalPages: 3, hasPrev: false, hasNext: true });
    expect(paginate(items, 3, 8)).toMatchObject({ items: [16], hasNext: false });
    expect(paginate(items, 99, 8).page).toBe(3);
    expect(paginate([], 1, 8)).toMatchObject({ totalPages: 1, page: 1, items: [] });
  });

  it('keeps active filters in pagination links', () => {
    const { filters } = parseCatalogQuery({ category: 'tech', min: '5', inStock: '1' }, ['tech']);
    expect(catalogQueryString(filters, 2)).toBe('?category=tech&min=5.00&inStock=1&page=2');
    expect(catalogQueryString(parseCatalogQuery({}, []).filters, 1)).toBe('');
  });
});
