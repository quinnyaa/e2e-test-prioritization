// Pure search rules: query normalization, case-insensitive partial matching and ranking.

export const MIN_QUERY_LENGTH = 2;
export const MAX_QUERY_LENGTH = 100;

export interface Searchable {
  name: string;
}

export type SearchResult<T> =
  | { status: 'empty'; query: string }
  | { status: 'too_short'; query: string }
  | { status: 'ok'; query: string; items: T[] };

/** Trims, collapses inner whitespace and limits the length; keeps the original case for display. */
export function normalizeQuery(raw: unknown): string {
  if (typeof raw !== 'string') return '';
  return raw.trim().replace(/\s+/g, ' ').slice(0, MAX_QUERY_LENGTH);
}

export function matchesQuery(name: string, query: string): boolean {
  return name.toLowerCase().includes(query.toLowerCase());
}

/** 0 = name starts with the query, 1 = a word starts with it, 2 = match elsewhere. */
export function matchRank(name: string, query: string): number {
  const lowerName = name.toLowerCase();
  const lowerQuery = query.toLowerCase();
  if (lowerName.startsWith(lowerQuery)) return 0;
  if (lowerName.split(' ').some((word) => word.startsWith(lowerQuery))) return 1;
  return 2;
}

export function searchProducts<T extends Searchable>(products: T[], rawQuery: unknown): SearchResult<T> {
  const query = normalizeQuery(rawQuery);
  if (query === '') return { status: 'empty', query };
  if (query.length < MIN_QUERY_LENGTH) return { status: 'too_short', query };

  const items = products
    .filter((p) => matchesQuery(p.name, query))
    .sort((a, b) => matchRank(a.name, query) - matchRank(b.name, query) || a.name.localeCompare(b.name, 'en'));
  return { status: 'ok', query, items };
}
