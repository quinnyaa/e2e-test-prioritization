import { describe, expect, it } from 'vitest';
import { canAddToCart, stockLabel, stockStatus } from '../../src/modules/product/logic';

describe('product logic', () => {
  it('derives stock status around the low-stock threshold', () => {
    expect(stockStatus(0)).toBe('out_of_stock');
    expect(stockStatus(1)).toBe('low_stock');
    expect(stockStatus(5)).toBe('low_stock');
    expect(stockStatus(6)).toBe('in_stock');
  });

  it('builds stock labels', () => {
    expect(stockLabel(0)).toBe('Out of stock');
    expect(stockLabel(3)).toBe('Only 3 left');
    expect(stockLabel(20)).toBe('In stock');
  });

  it('allows adding to cart only when in stock', () => {
    expect(canAddToCart(0)).toBe(false);
    expect(canAddToCart(1)).toBe(true);
  });
});
