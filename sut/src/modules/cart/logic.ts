// Pure cart rules: quantity validation against stock and limits, totals and stock checks.

export const MAX_QUANTITY_PER_LINE = 10;

export interface CartLine {
  productId: number;
  name: string;
  priceCents: number;
  quantity: number;
  stock: number;
}

export interface StockIssue {
  productId: number;
  message: string;
}

export type AddResult = { ok: true; quantity: number } | { ok: false; error: string };

export type UpdateResult =
  | { ok: true; action: 'remove' }
  | { ok: true; action: 'set'; quantity: number }
  | { ok: false; error: string };

/** Accepts only non-negative whole numbers written as digits; anything else is null. */
export function parseQuantity(raw: unknown): number | null {
  if (typeof raw !== 'string') return null;
  const value = raw.trim();
  if (!/^\d{1,4}$/.test(value)) return null;
  return Number(value);
}

/** Validates adding `requested` items to a line that already holds `inCart` items. */
export function validateAddQuantity(requested: number | null, inCart: number, stock: number): AddResult {
  if (requested === null || requested < 1) return { ok: false, error: 'Quantity must be a whole number of at least 1' };
  if (stock <= 0) return { ok: false, error: 'This product is out of stock' };
  const total = inCart + requested;
  if (total > MAX_QUANTITY_PER_LINE) {
    return { ok: false, error: `You can add at most ${MAX_QUANTITY_PER_LINE} items of one product` };
  }
  if (total > stock) {
    const note = inCart > 0 ? ` (${inCart} already in your cart)` : '';
    return { ok: false, error: `Only ${stock} item${stock === 1 ? '' : 's'} available${note}` };
  }
  return { ok: true, quantity: total };
}

/** Validates a new quantity typed on the cart page; zero removes the line. */
export function validateUpdateQuantity(requested: number | null, stock: number): UpdateResult {
  if (requested === null) return { ok: false, error: 'Quantity must be a whole number' };
  if (requested === 0) return { ok: true, action: 'remove' };
  if (requested > MAX_QUANTITY_PER_LINE) {
    return { ok: false, error: `You can add at most ${MAX_QUANTITY_PER_LINE} items of one product` };
  }
  if (requested > stock) return { ok: false, error: `Only ${stock} item${stock === 1 ? '' : 's'} available` };
  return { ok: true, action: 'set', quantity: requested };
}

export function lineTotal(line: CartLine): number {
  return line.priceCents * line.quantity;
}

export function cartSubtotal(lines: CartLine[]): number {
  return lines.reduce((sum, line) => sum + lineTotal(line), 0);
}

export function cartItemCount(lines: CartLine[]): number {
  return lines.reduce((sum, line) => sum + line.quantity, 0);
}

/** Lines whose quantity can no longer be fulfilled (stock changed after they were added). */
export function findStockIssues(lines: CartLine[]): StockIssue[] {
  const issues: StockIssue[] = [];
  for (const line of lines) {
    if (line.stock <= 0) {
      issues.push({ productId: line.productId, message: `${line.name} is out of stock` });
    } else if (line.quantity > line.stock) {
      issues.push({ productId: line.productId, message: `Only ${line.stock} of ${line.name} available` });
    }
  }
  return issues;
}

export function canCheckout(lines: CartLine[]): boolean {
  return lines.length > 0 && findStockIssues(lines).length === 0;
}
