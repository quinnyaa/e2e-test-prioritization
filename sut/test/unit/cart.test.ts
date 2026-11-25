import { describe, expect, it } from 'vitest';
import {
  canCheckout,
  cartItemCount,
  cartSubtotal,
  findStockIssues,
  MAX_QUANTITY_PER_LINE,
  parseQuantity,
  validateAddQuantity,
  validateUpdateQuantity,
  type CartLine,
} from '../../src/modules/cart/logic';

const line = (productId: number, priceCents: number, quantity: number, stock: number): CartLine => ({
  productId,
  name: `P${productId}`,
  priceCents,
  quantity,
  stock,
});

describe('cart logic', () => {
  it('parses quantities', () => {
    expect(parseQuantity('3')).toBe(3);
    expect(parseQuantity(' 0 ')).toBe(0);
    expect(parseQuantity('-1')).toBeNull();
    expect(parseQuantity('1.5')).toBeNull();
    expect(parseQuantity(undefined)).toBeNull();
  });

  it('limits added quantity by stock and per-line maximum', () => {
    expect(validateAddQuantity(2, 1, 3)).toEqual({ ok: true, quantity: 3 });
    expect(validateAddQuantity(3, 1, 3)).toMatchObject({ ok: false });
    expect(validateAddQuantity(0, 0, 3)).toMatchObject({ ok: false });
    expect(validateAddQuantity(1, 0, 0)).toEqual({ ok: false, error: 'This product is out of stock' });
    expect(validateAddQuantity(MAX_QUANTITY_PER_LINE, 0, 50)).toMatchObject({ ok: true });
    expect(validateAddQuantity(MAX_QUANTITY_PER_LINE + 1, 0, 50)).toMatchObject({ ok: false });
  });

  it('treats zero as removal when updating', () => {
    expect(validateUpdateQuantity(0, 5)).toEqual({ ok: true, action: 'remove' });
    expect(validateUpdateQuantity(5, 5)).toEqual({ ok: true, action: 'set', quantity: 5 });
    expect(validateUpdateQuantity(6, 5)).toMatchObject({ ok: false });
    expect(validateUpdateQuantity(null, 5)).toMatchObject({ ok: false });
  });

  it('calculates totals', () => {
    const lines = [line(1, 1250, 2, 10), line(2, 2499, 1, 10)];
    expect(cartSubtotal(lines)).toBe(4999);
    expect(cartItemCount(lines)).toBe(3);
    expect(cartSubtotal([])).toBe(0);
  });

  it('detects lines that exceed current stock', () => {
    const lines = [line(1, 100, 2, 2), line(2, 100, 3, 1), line(3, 100, 1, 0)];
    expect(findStockIssues(lines).map((i) => i.productId)).toEqual([2, 3]);
    expect(canCheckout(lines)).toBe(false);
    expect(canCheckout([line(1, 100, 2, 2)])).toBe(true);
    expect(canCheckout([])).toBe(false);
  });
});
