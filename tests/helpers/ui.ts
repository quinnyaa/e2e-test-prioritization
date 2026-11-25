// Small UI steps shared by several specs. Only data-testid selectors are used.
import { expect, type Page } from '@playwright/test';
import type { ShippingForm } from './data';

/** Submits the add-to-cart form and waits for the redirected page with its success or error flash. */
export async function addToCartViaUi(page: Page, productId: number, quantity: number): Promise<void> {
  await page.goto(`/products/${productId}`);
  await page.getByTestId('quantity-input').fill(String(quantity));
  await page.getByTestId('add-to-cart').click();
  await expect(page.getByTestId('flash-message')).toBeVisible();
}

export async function fillShipping(page: Page, form: ShippingForm): Promise<void> {
  await page.getByTestId('shipping-full-name').fill(form.fullName);
  await page.getByTestId('shipping-address').fill(form.address);
  await page.getByTestId('shipping-city').fill(form.city);
  await page.getByTestId('shipping-postal-code').fill(form.postalCode);
  await page.getByTestId('shipping-phone').fill(form.phone);
}

/** Enters a promo code and recalculates the checkout totals (the "Apply code / update totals" button). */
export async function applyPromo(page: Page, code: string): Promise<void> {
  await page.getByTestId('promo-input').fill(code);
  await page.getByTestId('promo-apply').click();
}

export interface Summary {
  subtotal: string;
  discount: string;
  shipping: string;
  total: string;
}

export async function expectCheckoutSummary(page: Page, summary: Summary): Promise<void> {
  await expect(page.getByTestId('summary-subtotal')).toHaveText(summary.subtotal);
  await expect(page.getByTestId('summary-discount')).toHaveText(summary.discount);
  await expect(page.getByTestId('summary-shipping')).toHaveText(summary.shipping);
  await expect(page.getByTestId('summary-total')).toHaveText(summary.total);
}

export async function expectFlash(page: Page, text: string): Promise<void> {
  await expect(page.getByTestId('flash-message')).toHaveText(text);
}
