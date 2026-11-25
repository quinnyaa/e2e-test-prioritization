// Pure catalog rules: query parsing, filtering, sorting and pagination.
import { parseMoney } from '../../shared/money';

export const PAGE_SIZE = 8;
export const SORT_OPTIONS = ['name_asc', 'price_asc', 'price_desc'] as const;
export type SortOption = (typeof SORT_OPTIONS)[number];
export const DEFAULT_SORT: SortOption = 'name_asc';

export const SORT_LABELS: Record<SortOption, string> = {
  name_asc: 'Name (A–Z)',
  price_asc: 'Price: low to high',
  price_desc: 'Price: high to low',
};

export interface CatalogProduct {
  id: number;
  name: string;
  priceCents: number;
  stock: number;
  categorySlug: string;
  categoryName: string;
}

export interface CatalogFilters {
  category: string | null;
  minPriceCents: number | null;
  maxPriceCents: number | null;
  inStockOnly: boolean;
  sort: SortOption;
  page: number;
}

export interface Page<T> {
  items: T[];
  page: number;
  totalPages: number;
  totalItems: number;
  hasPrev: boolean;
  hasNext: boolean;
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

export function parsePage(raw: unknown): number {
  const value = asString(raw);
  if (!/^\d+$/.test(value)) return 1;
  const page = Number(value);
  return page >= 1 ? page : 1;
}

export function parseSort(raw: unknown): SortOption {
  const value = asString(raw);
  return (SORT_OPTIONS as readonly string[]).includes(value) ? (value as SortOption) : DEFAULT_SORT;
}

export function parseCatalogQuery(
  query: Record<string, unknown>,
  knownCategories: string[],
): { filters: CatalogFilters; errors: string[] } {
  const errors: string[] = [];
  const category = asString(query.category);

  const minRaw = asString(query.min);
  const maxRaw = asString(query.max);
  let minPriceCents = minRaw === '' ? null : parseMoney(minRaw);
  let maxPriceCents = maxRaw === '' ? null : parseMoney(maxRaw);
  if (minRaw !== '' && minPriceCents === null) errors.push('Minimum price must be a valid amount');
  if (maxRaw !== '' && maxPriceCents === null) errors.push('Maximum price must be a valid amount');
  if (minPriceCents !== null && maxPriceCents !== null && minPriceCents > maxPriceCents) {
    errors.push('Minimum price cannot be greater than maximum price');
    minPriceCents = null;
    maxPriceCents = null;
  }

  return {
    filters: {
      category: knownCategories.includes(category) ? category : null,
      minPriceCents,
      maxPriceCents,
      inStockOnly: asString(query.inStock) === '1',
      sort: parseSort(query.sort),
      page: parsePage(query.page),
    },
    errors,
  };
}

export function isInStock(product: { stock: number }): boolean {
  return product.stock > 0;
}

export function filterProducts<T extends CatalogProduct>(products: T[], filters: CatalogFilters): T[] {
  return products.filter((p) => {
    if (filters.category !== null && p.categorySlug !== filters.category) return false;
    if (filters.minPriceCents !== null && p.priceCents < filters.minPriceCents) return false;
    if (filters.maxPriceCents !== null && p.priceCents > filters.maxPriceCents) return false;
    if (filters.inStockOnly && !isInStock(p)) return false;
    return true;
  });
}

function byName(a: { name: string }, b: { name: string }): number {
  return a.name.localeCompare(b.name, 'en');
}

export function sortProducts<T extends CatalogProduct>(products: T[], sort: SortOption): T[] {
  const sorted = [...products];
  switch (sort) {
    case 'price_asc':
      return sorted.sort((a, b) => a.priceCents - b.priceCents || byName(a, b));
    case 'price_desc':
      return sorted.sort((a, b) => b.priceCents - a.priceCents || byName(a, b));
    case 'name_asc':
    default:
      return sorted.sort(byName);
  }
}

export function paginate<T>(items: T[], page: number, pageSize = PAGE_SIZE): Page<T> {
  const totalItems = items.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
  const current = Math.min(Math.max(page, 1), totalPages);
  const start = (current - 1) * pageSize;
  return {
    items: items.slice(start, start + pageSize),
    page: current,
    totalPages,
    totalItems,
    hasPrev: current > 1,
    hasNext: current < totalPages,
  };
}

/** Builds the query string for catalog links so that pagination keeps the active filters. */
export function catalogQueryString(filters: CatalogFilters, page: number): string {
  const params = new URLSearchParams();
  if (filters.category) params.set('category', filters.category);
  if (filters.minPriceCents !== null) params.set('min', (filters.minPriceCents / 100).toFixed(2));
  if (filters.maxPriceCents !== null) params.set('max', (filters.maxPriceCents / 100).toFixed(2));
  if (filters.inStockOnly) params.set('inStock', '1');
  if (filters.sort !== DEFAULT_SORT) params.set('sort', filters.sort);
  if (page > 1) params.set('page', String(page));
  const query = params.toString();
  return query ? `?${query}` : '';
}
