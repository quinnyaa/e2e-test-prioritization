// Pure order rules: statuses, allowed transitions, numbering and ordering of the history.

export const ORDER_STATUSES = ['pending', 'paid', 'shipped', 'delivered', 'cancelled'] as const;
export type OrderStatus = (typeof ORDER_STATUSES)[number];

export const INITIAL_ORDER_STATUS: OrderStatus = 'pending';

export const STATUS_LABELS: Record<OrderStatus, string> = {
  pending: 'Pending',
  paid: 'Paid',
  shipped: 'Shipped',
  delivered: 'Delivered',
  cancelled: 'Cancelled',
};

const TRANSITIONS: Record<OrderStatus, OrderStatus[]> = {
  pending: ['paid', 'cancelled'],
  paid: ['shipped', 'cancelled'],
  shipped: ['delivered'],
  delivered: [],
  cancelled: [],
};

export interface OrderListItem {
  id: number;
  status: OrderStatus;
  totalCents: number;
  itemCount: number;
  createdAt: string;
}

export function isOrderStatus(value: unknown): value is OrderStatus {
  return typeof value === 'string' && (ORDER_STATUSES as readonly string[]).includes(value);
}

export function canTransition(from: OrderStatus, to: OrderStatus): boolean {
  return TRANSITIONS[from].includes(to);
}

export function allowedNextStatuses(status: OrderStatus): OrderStatus[] {
  return [...TRANSITIONS[status]];
}

/** Customers may cancel only orders that have not been paid yet. */
export function canCustomerCancel(status: OrderStatus): boolean {
  return status === 'pending';
}

export function formatOrderNumber(id: number): string {
  return `ORD-${String(id).padStart(5, '0')}`;
}

export function sortNewestFirst<T extends { id: number; createdAt: string }>(orders: T[]): T[] {
  return [...orders].sort((a, b) => b.createdAt.localeCompare(a.createdAt) || b.id - a.id);
}
