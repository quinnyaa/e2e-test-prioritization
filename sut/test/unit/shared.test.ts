import { describe, expect, it } from 'vitest';
import { formatPrice, parseMoney } from '../../src/shared/money';
import { parseId } from '../../src/shared/params';

describe('money', () => {
  it('formats cents as dollars with thousand separators', () => {
    expect(formatPrice(0)).toBe('$0.00');
    expect(formatPrice(5)).toBe('$0.05');
    expect(formatPrice(123456789)).toBe('$1,234,567.89');
    expect(formatPrice(-499)).toBe('-$4.99');
  });

  it('parses user-entered amounts into cents', () => {
    expect(parseMoney('12')).toBe(1200);
    expect(parseMoney(' 12.5 ')).toBe(1250);
    expect(parseMoney('0.07')).toBe(7);
    expect(parseMoney('12.345')).toBeNull();
    expect(parseMoney('-1')).toBeNull();
    expect(parseMoney('')).toBeNull();
    expect(parseMoney(undefined)).toBeNull();
  });
});

describe('parseId', () => {
  it('accepts positive integers only', () => {
    expect(parseId('42')).toBe(42);
    expect(parseId('0')).toBeNull();
    expect(parseId('4.2')).toBeNull();
    expect(parseId('abc')).toBeNull();
    expect(parseId(undefined)).toBeNull();
  });
});
