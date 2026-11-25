import { describe, expect, it } from 'vitest';
import { validateProductInput, validateStatusChange } from '../../src/modules/admin/logic';

const valid = { name: 'Desk Lamp', description: 'LED lamp', price: '29.90', stock: '5', category: 'home' };

describe('admin logic', () => {
  it('accepts a valid product', () => {
    expect(validateProductInput(valid, ['home'])).toEqual({
      ok: true,
      values: { name: 'Desk Lamp', description: 'LED lamp', priceCents: 2990, stock: 5, categorySlug: 'home' },
    });
  });

  it('rejects invalid product fields', () => {
    const result = validateProductInput({ name: 'X', description: '', price: '0', stock: '-1', category: 'nope' }, ['home']);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(Object.keys(result.errors).sort()).toEqual(['category', 'description', 'name', 'price', 'stock']);
  });

  it('checks price and stock limits', () => {
    expect(validateProductInput({ ...valid, price: '10000.00' }, ['home']).ok).toBe(true);
    expect(validateProductInput({ ...valid, price: '10000.01' }, ['home']).ok).toBe(false);
    expect(validateProductInput({ ...valid, stock: '9999' }, ['home']).ok).toBe(true);
    expect(validateProductInput({ ...valid, stock: '10000' }, ['home']).ok).toBe(false);
  });

  it('validates status changes', () => {
    expect(validateStatusChange('pending', 'paid')).toEqual({ ok: true, status: 'paid' });
    expect(validateStatusChange('pending', 'delivered')).toMatchObject({ ok: false });
    expect(validateStatusChange('pending', 'lost')).toEqual({ ok: false, error: 'Unknown order status' });
  });
});
