// HTTP shortcuts for test preconditions. They go through the same routes as the UI and fail loudly,
// so a broken precondition is reported as a test failure (which is a legitimate detection for a mutant).
import type { APIRequestContext } from '@playwright/test';
import { VALID_SHIPPING, type ShippingForm, type User } from './data';

function fail(step: string, status: number, body: string): never {
  throw new Error(`${step} failed: HTTP ${status}\n${body.slice(0, 500)}`);
}

export async function login(request: APIRequestContext, user: User): Promise<void> {
  const res = await request.post('/login', {
    form: { email: user.email, password: user.password, next: '/catalog' },
    maxRedirects: 0,
  });
  if (res.status() !== 302) fail(`Login as ${user.email}`, res.status(), await res.text());
}

/** Adds a product to the cart of the session behind `request`; expects a success flash. */
export async function addToCart(request: APIRequestContext, productId: number, quantity: number): Promise<void> {
  const res = await request.post('/cart/add', { form: { productId: String(productId), quantity: String(quantity) } });
  const body = await res.text();
  if (!res.ok() || !body.includes('flash-success')) fail(`Add ${quantity} × #${productId} to cart`, res.status(), body);
}

export interface OrderOptions {
  promoCode?: string;
  shippingMethod?: 'standard' | 'express';
  shipping?: ShippingForm;
}

/** Fills the cart with `items` and places an order; returns the new order id. */
export async function placeOrder(
  request: APIRequestContext,
  items: Array<[productId: number, quantity: number]>,
  options: OrderOptions = {},
): Promise<number> {
  for (const [productId, quantity] of items) await addToCart(request, productId, quantity);
  const res = await request.post('/checkout', {
    form: {
      ...(options.shipping ?? VALID_SHIPPING),
      shippingMethod: options.shippingMethod ?? 'standard',
      promoCode: options.promoCode ?? '',
    },
    maxRedirects: 0,
  });
  const match = /\/checkout\/confirmation\/(\d+)$/.exec(res.headers()['location'] ?? '');
  if (res.status() !== 302 || !match) fail('Place order', res.status(), await res.text());
  return Number(match[1]);
}

export interface ProductFields {
  name: string;
  description: string;
  price: string;
  stock: string;
  category: string;
}

/** Admin only: overwrites all fields of an existing product. */
export async function updateProduct(request: APIRequestContext, id: number, fields: ProductFields): Promise<void> {
  const res = await request.post(`/admin/products/${id}`, { form: { ...fields }, maxRedirects: 0 });
  if (res.status() !== 302) fail(`Update product #${id}`, res.status(), await res.text());
}

/** Admin only: requests an order status change; the result is reported as a flash on the next page. */
export async function requestStatusChange(request: APIRequestContext, orderId: number, status: string): Promise<void> {
  const res = await request.post(`/admin/orders/${orderId}/status`, { form: { status }, maxRedirects: 0 });
  if (res.status() !== 302) fail(`Change status of order #${orderId}`, res.status(), await res.text());
}
