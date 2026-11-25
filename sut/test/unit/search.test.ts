import { describe, expect, it } from 'vitest';
import { normalizeQuery, searchProducts } from '../../src/modules/search/logic';

const products = [{ name: 'Mechanical Keyboard' }, { name: 'Board Game Classic' }, { name: 'Bamboo Cutting Board' }, { name: 'Yoga Mat' }];

describe('search logic', () => {
  it('normalizes whitespace', () => {
    expect(normalizeQuery('  coffee    mug ')).toBe('coffee mug');
    expect(normalizeQuery(undefined)).toBe('');
  });

  it('reports empty and too short queries', () => {
    expect(searchProducts(products, '   ').status).toBe('empty');
    expect(searchProducts(products, 'b').status).toBe('too_short');
  });

  it('matches case-insensitively and ranks prefix matches first', () => {
    const result = searchProducts(products, 'BOARD');
    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.items.map((p) => p.name)).toEqual(['Board Game Classic', 'Bamboo Cutting Board', 'Mechanical Keyboard']);
    }
  });

  it('returns an empty list when nothing matches', () => {
    const result = searchProducts(products, 'xyz');
    expect(result.status === 'ok' && result.items).toEqual([]);
  });
});
