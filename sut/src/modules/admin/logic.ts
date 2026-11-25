// Pure rules for the admin area: product form validation and order status changes.
import { formatPrice, parseMoney } from '../../shared/money';
import { canTransition, isOrderStatus, STATUS_LABELS, type OrderStatus } from '../orders/logic';

export const PRODUCT_NAME_MIN_LENGTH = 2;
export const PRODUCT_NAME_MAX_LENGTH = 100;
export const DESCRIPTION_MAX_LENGTH = 1000;
export const MAX_PRICE_CENTS = 1_000_000;
export const MAX_STOCK = 9999;

export interface ProductInput {
  name: string;
  description: string;
  price: string;
  stock: string;
  category: string;
}

export interface ProductValues {
  name: string;
  description: string;
  priceCents: number;
  stock: number;
  categorySlug: string;
}

export type ProductErrors = Partial<Record<keyof ProductInput, string>>;

export type ProductValidation = { ok: true; values: ProductValues } | { ok: false; errors: ProductErrors };

export type StatusChange = { ok: true; status: OrderStatus } | { ok: false; error: string };

export function validateProductInput(input: ProductInput, categorySlugs: string[]): ProductValidation {
  const errors: ProductErrors = {};

  const name = input.name.trim();
  if (name === '') errors.name = 'Name is required';
  else if (name.length < PRODUCT_NAME_MIN_LENGTH || name.length > PRODUCT_NAME_MAX_LENGTH) {
    errors.name = `Name must be ${PRODUCT_NAME_MIN_LENGTH}–${PRODUCT_NAME_MAX_LENGTH} characters`;
  }

  const description = input.description.trim();
  if (description === '') errors.description = 'Description is required';
  else if (description.length > DESCRIPTION_MAX_LENGTH) {
    errors.description = `Description must be at most ${DESCRIPTION_MAX_LENGTH} characters`;
  }

  const priceCents = parseMoney(input.price);
  if (priceCents === null) errors.price = 'Price must be a valid amount';
  else if (priceCents <= 0) errors.price = 'Price must be greater than zero';
  else if (priceCents > MAX_PRICE_CENTS) errors.price = `Price must not exceed ${formatPrice(MAX_PRICE_CENTS)}`;

  const stockRaw = input.stock.trim();
  const stock = /^\d+$/.test(stockRaw) ? Number(stockRaw) : NaN;
  if (Number.isNaN(stock) || stock > MAX_STOCK) errors.stock = `Stock must be a whole number from 0 to ${MAX_STOCK}`;

  if (!categorySlugs.includes(input.category)) errors.category = 'Choose a category';

  if (Object.keys(errors).length > 0) return { ok: false, errors };
  return { ok: true, values: { name, description, priceCents: priceCents!, stock, categorySlug: input.category } };
}

export function validateStatusChange(current: OrderStatus, requested: unknown): StatusChange {
  if (!isOrderStatus(requested)) return { ok: false, error: 'Unknown order status' };
  if (!canTransition(current, requested)) {
    return { ok: false, error: `Cannot change status from ${STATUS_LABELS[current]} to ${STATUS_LABELS[requested]}` };
  }
  return { ok: true, status: requested };
}
