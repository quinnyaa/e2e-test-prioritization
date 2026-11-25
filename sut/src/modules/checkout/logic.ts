// Pure checkout rules: shipping form validation, promo codes, shipping cost, totals and stock updates.
import { formatPrice } from '../../shared/money';

export type ShippingMethod = 'standard' | 'express';
export const SHIPPING_METHODS: ShippingMethod[] = ['standard', 'express'];

export interface ShippingRate {
  label: string;
  costCents: number;
  /** Orders at or above this amount (after discount) ship for free; null = never free. */
  freeFromCents: number | null;
}

export const SHIPPING_RATES: Record<ShippingMethod, ShippingRate> = {
  standard: { label: 'Standard (3–5 days)', costCents: 499, freeFromCents: 5000 },
  express: { label: 'Express (1–2 days)', costCents: 1499, freeFromCents: null },
};

export type PromoType = 'percent' | 'fixed' | 'free_shipping';

export interface PromoCode {
  code: string;
  type: PromoType;
  value: number;
  minOrderCents: number;
  expiresAt: string | null;
  active: boolean;
}

export type PromoResult =
  | { ok: true; code: string | null; discountCents: number; freeShipping: boolean }
  | { ok: false; error: string };

export const NO_PROMO: PromoResult = { ok: true, code: null, discountCents: 0, freeShipping: false };

export interface Totals {
  subtotalCents: number;
  discountCents: number;
  shippingCents: number;
  totalCents: number;
}

export interface ShippingForm {
  fullName: string;
  address: string;
  city: string;
  postalCode: string;
  phone: string;
}

export type ShippingErrors = Partial<Record<keyof ShippingForm, string>>;

export interface StockLine {
  productId: number;
  quantity: number;
  stock: number;
}

export interface StockUpdate {
  productId: number;
  from: number;
  to: number;
}

export function normalizePromoCode(raw: unknown): string {
  return typeof raw === 'string' ? raw.trim().toUpperCase() : '';
}

export function parseShippingMethod(raw: unknown): ShippingMethod {
  return raw === 'express' ? 'express' : 'standard';
}

export function evaluatePromo(
  promo: PromoCode | undefined,
  code: string,
  subtotalCents: number,
  now: Date,
): PromoResult {
  if (code === '') return NO_PROMO;
  if (!promo || !promo.active) return { ok: false, error: 'Promo code not found' };
  if (promo.expiresAt !== null && now.getTime() > Date.parse(promo.expiresAt)) {
    return { ok: false, error: 'Promo code has expired' };
  }
  if (subtotalCents < promo.minOrderCents) {
    return { ok: false, error: `This code requires a minimum order of ${formatPrice(promo.minOrderCents)}` };
  }
  switch (promo.type) {
    case 'percent':
      return {
        ok: true,
        code: promo.code,
        discountCents: Math.round((subtotalCents * promo.value) / 100),
        freeShipping: false,
      };
    case 'fixed':
      return { ok: true, code: promo.code, discountCents: Math.min(promo.value, subtotalCents), freeShipping: false };
    case 'free_shipping':
      return { ok: true, code: promo.code, discountCents: 0, freeShipping: true };
  }
}

/** Free-shipping promo codes apply to standard shipping only. */
export function calculateShipping(method: ShippingMethod, amountCents: number, freeShippingPromo: boolean): number {
  if (freeShippingPromo && method === 'standard') return 0;
  const rate = SHIPPING_RATES[method];
  if (rate.freeFromCents !== null && amountCents >= rate.freeFromCents) return 0;
  return rate.costCents;
}

export function calculateTotals(subtotalCents: number, promo: PromoResult, method: ShippingMethod): Totals {
  const applied = promo.ok ? promo : NO_PROMO;
  const discountCents = Math.min(applied.ok ? applied.discountCents : 0, subtotalCents);
  const afterDiscount = subtotalCents - discountCents;
  const shippingCents = calculateShipping(method, afterDiscount, applied.ok && applied.freeShipping);
  return { subtotalCents, discountCents, shippingCents, totalCents: afterDiscount + shippingCents };
}

export function normalizePhone(raw: string): string {
  return raw.replace(/[\s\-()]/g, '');
}

export function validateShippingForm(form: ShippingForm): ShippingErrors {
  const errors: ShippingErrors = {};
  const fullName = form.fullName.trim();
  if (fullName === '') errors.fullName = 'Full name is required';
  else if (fullName.length < 2 || fullName.length > 100) errors.fullName = 'Full name must be 2–100 characters';

  const address = form.address.trim();
  if (address === '') errors.address = 'Address is required';
  else if (address.length < 5) errors.address = 'Address is too short';

  const city = form.city.trim();
  if (city === '') errors.city = 'City is required';
  else if (!/^[\p{L} .'-]{2,50}$/u.test(city)) errors.city = 'Enter a valid city';

  const postalCode = form.postalCode.trim();
  if (postalCode === '') errors.postalCode = 'Postal code is required';
  else if (!/^\d{5}$/.test(postalCode)) errors.postalCode = 'Postal code must be 5 digits';

  const phone = normalizePhone(form.phone);
  if (phone === '') errors.phone = 'Phone is required';
  else if (!/^\+?\d{10,13}$/.test(phone)) errors.phone = 'Enter a valid phone number';

  return errors;
}

/** Computes the new stock levels for an order, or null if any line exceeds the available stock. */
export function computeStockUpdates(lines: StockLine[]): StockUpdate[] | null {
  const updates: StockUpdate[] = [];
  for (const line of lines) {
    if (line.quantity <= 0 || line.quantity > line.stock) return null;
    updates.push({ productId: line.productId, from: line.stock, to: line.stock - line.quantity });
  }
  return updates;
}
