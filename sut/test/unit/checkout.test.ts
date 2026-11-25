import { describe, expect, it } from 'vitest';
import {
  calculateShipping,
  calculateTotals,
  computeStockUpdates,
  evaluatePromo,
  NO_PROMO,
  normalizePromoCode,
  validateShippingForm,
  type PromoCode,
} from '../../src/modules/checkout/logic';

const NOW = new Date('2026-03-01T12:00:00.000Z');
const promo = (overrides: Partial<PromoCode>): PromoCode => ({
  code: 'TEST',
  type: 'percent',
  value: 10,
  minOrderCents: 0,
  expiresAt: null,
  active: true,
  ...overrides,
});

describe('promo codes', () => {
  it('normalizes the code', () => {
    expect(normalizePromoCode('  save10 ')).toBe('SAVE10');
  });

  it('applies percent, fixed and free-shipping codes', () => {
    expect(evaluatePromo(promo({ value: 10 }), 'TEST', 4999, NOW)).toMatchObject({ ok: true, discountCents: 500 });
    expect(evaluatePromo(promo({ type: 'fixed', value: 500 }), 'TEST', 300, NOW)).toMatchObject({ discountCents: 300 });
    expect(evaluatePromo(promo({ type: 'free_shipping', value: 0 }), 'TEST', 300, NOW)).toMatchObject({
      discountCents: 0,
      freeShipping: true,
    });
  });

  it('rejects unknown, inactive, expired and below-minimum codes', () => {
    expect(evaluatePromo(undefined, 'NOPE', 1000, NOW)).toEqual({ ok: false, error: 'Promo code not found' });
    expect(evaluatePromo(promo({ active: false }), 'TEST', 1000, NOW)).toMatchObject({ ok: false });
    expect(evaluatePromo(promo({ expiresAt: '2026-02-28T23:59:59.000Z' }), 'TEST', 1000, NOW)).toEqual({
      ok: false,
      error: 'Promo code has expired',
    });
    expect(evaluatePromo(promo({ minOrderCents: 10000 }), 'TEST', 9999, NOW)).toMatchObject({ ok: false });
    expect(evaluatePromo(promo({ minOrderCents: 10000 }), 'TEST', 10000, NOW)).toMatchObject({ ok: true });
  });

  it('treats an empty code as no promo', () => {
    expect(evaluatePromo(undefined, '', 1000, NOW)).toEqual(NO_PROMO);
  });
});

describe('shipping and totals', () => {
  it('makes standard shipping free from the threshold', () => {
    expect(calculateShipping('standard', 4999, false)).toBe(499);
    expect(calculateShipping('standard', 5000, false)).toBe(0);
    expect(calculateShipping('express', 100000, false)).toBe(1499);
  });

  it('applies free-shipping promo to standard shipping only', () => {
    expect(calculateShipping('standard', 1000, true)).toBe(0);
    expect(calculateShipping('express', 1000, true)).toBe(1499);
  });

  it('uses the discounted amount for the free-shipping threshold', () => {
    const discount = evaluatePromo(promo({ value: 10 }), 'TEST', 5500, NOW);
    expect(calculateTotals(5500, discount, 'standard')).toEqual({
      subtotalCents: 5500,
      discountCents: 550,
      shippingCents: 499,
      totalCents: 5449,
    });
    expect(calculateTotals(6000, NO_PROMO, 'standard').shippingCents).toBe(0);
  });

  it('ignores a failed promo when calculating totals', () => {
    expect(calculateTotals(1000, { ok: false, error: 'x' }, 'express').totalCents).toBe(2499);
  });
});

describe('shipping form', () => {
  const valid = { fullName: 'Alice', address: '12 Green Street', city: 'Kyiv', postalCode: '01001', phone: '+38 050 123-45-67' };

  it('accepts a valid form', () => {
    expect(validateShippingForm(valid)).toEqual({});
  });

  it('reports invalid fields', () => {
    const errors = validateShippingForm({ fullName: '', address: 'abc', city: 'K1', postalCode: '1234', phone: '123' });
    expect(Object.keys(errors).sort()).toEqual(['address', 'city', 'fullName', 'phone', 'postalCode']);
  });
});

describe('stock updates', () => {
  it('decrements stock or refuses when stock is insufficient', () => {
    expect(computeStockUpdates([{ productId: 1, quantity: 2, stock: 5 }])).toEqual([{ productId: 1, from: 5, to: 3 }]);
    expect(computeStockUpdates([{ productId: 1, quantity: 5, stock: 5 }])).toEqual([{ productId: 1, from: 5, to: 0 }]);
    expect(computeStockUpdates([{ productId: 1, quantity: 6, stock: 5 }])).toBeNull();
  });
});
