import { describe, expect, it } from 'vitest';
import {
  allowedNextStatuses,
  canCustomerCancel,
  canTransition,
  formatOrderNumber,
  isOrderStatus,
  sortNewestFirst,
} from '../../src/modules/orders/logic';

describe('orders logic', () => {
  it('allows only forward status transitions', () => {
    expect(canTransition('pending', 'paid')).toBe(true);
    expect(canTransition('paid', 'shipped')).toBe(true);
    expect(canTransition('pending', 'delivered')).toBe(false);
    expect(canTransition('delivered', 'cancelled')).toBe(false);
    expect(allowedNextStatuses('cancelled')).toEqual([]);
  });

  it('lets customers cancel pending orders only', () => {
    expect(canCustomerCancel('pending')).toBe(true);
    expect(canCustomerCancel('paid')).toBe(false);
    expect(canCustomerCancel('shipped')).toBe(false);
  });

  it('formats order numbers and validates statuses', () => {
    expect(formatOrderNumber(7)).toBe('ORD-00007');
    expect(isOrderStatus('shipped')).toBe(true);
    expect(isOrderStatus('lost')).toBe(false);
  });

  it('sorts orders newest first with id as a tie-breaker', () => {
    const orders = [
      { id: 1, createdAt: '2026-01-10T00:00:00.000Z' },
      { id: 3, createdAt: '2026-03-01T00:00:00.000Z' },
      { id: 2, createdAt: '2026-03-01T00:00:00.000Z' },
    ];
    expect(sortNewestFirst(orders).map((o) => o.id)).toEqual([3, 2, 1]);
  });
});
