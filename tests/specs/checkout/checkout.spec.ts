import { test, expect } from '../../fixtures/test';
import { addToCart } from '../../helpers/api';
import { P, USERS, VALID_SHIPPING } from '../../helpers/data';
import { meta } from '../../helpers/meta';
import { addToCartViaUi, applyPromo, expectCheckoutSummary, expectFlash, fillShipping } from '../../helpers/ui';

test(
  '[CHK-01] validates every shipping field and accepts boundary values',
  meta({
    modules: ['checkout'],
    crit: 'medium',
    covers: ['checkout/logic.ts', 'checkout/routes.ts', 'checkout/views/checkout.ejs'],
  }),
  async ({ page, loginAs }) => {
    await loginAs(USERS.bob);
    await addToCart(page.request, P.wirelessMouse.id, 1);
    await page.goto('/checkout');
    await expect(page.getByTestId('shipping-full-name')).toHaveValue('Bob Smith');

    await page.getByTestId('shipping-full-name').fill('   ');
    await page.getByTestId('checkout-submit').click();
    await expect(page.getByTestId('shipping-full-name-error')).toHaveText('Full name is required');
    await expect(page.getByTestId('shipping-address-error')).toHaveText('Address is required');
    await expect(page.getByTestId('shipping-city-error')).toHaveText('City is required');
    await expect(page.getByTestId('shipping-postal-code-error')).toHaveText('Postal code is required');
    await expect(page.getByTestId('shipping-phone-error')).toHaveText('Phone is required');

    await fillShipping(page, { fullName: 'B', address: '1 St', city: 'Kyiv1', postalCode: '1234', phone: '050 123 45' });
    await page.getByTestId('checkout-submit').click();
    await expect(page.getByTestId('shipping-full-name-error')).toHaveText('Full name must be 2–100 characters');
    await expect(page.getByTestId('shipping-address-error')).toHaveText('Address is too short');
    await expect(page.getByTestId('shipping-city-error')).toHaveText('Enter a valid city');
    await expect(page.getByTestId('shipping-postal-code-error')).toHaveText('Postal code must be 5 digits');
    await expect(page.getByTestId('shipping-phone-error')).toHaveText('Enter a valid phone number');
    await expect(page.getByTestId('shipping-city')).toHaveValue('Kyiv1');
    await expect(page.getByTestId('cart-count')).toHaveText('1');

    // Shortest valid values: 2-char name, 5-char address, 10-digit phone with separators.
    await fillShipping(page, {
      fullName: 'Bo',
      address: '1 Rd.',
      city: "Ivano-Frankivsk",
      postalCode: '01001',
      phone: '(050) 123-45-67',
    });
    await page.getByTestId('checkout-submit').click();
    await expect(page).toHaveURL(/\/checkout\/confirmation\/4$/);
    await expect(page.getByTestId('order-number')).toHaveText('ORD-00004');
    await page.getByTestId('view-order').click();
    await expect(page.getByTestId('order-address')).toHaveText('Bo, 1 Rd., Ivano-Frankivsk 01001, 0501234567');
  },
);

test(
  '[CHK-02] applies a percent promo code with rounding and keeps it on the order',
  meta({
    modules: ['checkout', 'orders'],
    crit: 'high',
    covers: ['checkout/logic.ts', 'checkout/routes.ts', 'checkout/service.ts', 'checkout/views/checkout.ejs', 'shared/money.ts'],
  }),
  async ({ page, loginAs }) => {
    await loginAs(USERS.bob);
    await addToCart(page.request, P.wirelessMouse.id, 1);
    await addToCart(page.request, P.plushBear.id, 1);
    await page.goto('/checkout');
    await expectCheckoutSummary(page, { subtotal: '$39.98', discount: '-$0.00', shipping: '$4.99', total: '$44.97' });

    await fillShipping(page, VALID_SHIPPING);
    await applyPromo(page, '  save10 ');
    await expect(page.getByTestId('promo-success')).toHaveText('Promo code SAVE10 applied');
    await expect(page.getByTestId('promo-input')).toHaveValue('SAVE10');
    // 10 % of $39.98 is $3.998, rounded to $4.00.
    await expectCheckoutSummary(page, { subtotal: '$39.98', discount: '-$4.00', shipping: '$4.99', total: '$40.97' });
    await expect(page.getByTestId('shipping-city')).toHaveValue('Lviv');

    await page.getByTestId('checkout-submit').click();
    await expect(page.getByTestId('order-total')).toHaveText('$40.97');
    await page.getByTestId('view-order').click();
    await expect(page.getByTestId('order-subtotal')).toHaveText('$39.98');
    await expect(page.getByTestId('order-discount')).toHaveText('-$4.00');
    await expect(page.getByTestId('order-shipping')).toHaveText('$4.99');
    await expect(page.getByTestId('order-total')).toHaveText('$40.97');
  },
);

test(
  '[CHK-03] applies fixed and free-shipping promo codes (free shipping is standard only)',
  meta({
    modules: ['checkout'],
    crit: 'high',
    covers: ['checkout/logic.ts', 'checkout/routes.ts', 'checkout/service.ts', 'checkout/views/checkout.ejs'],
  }),
  async ({ page, loginAs }) => {
    await loginAs(USERS.bob);
    await addToCart(page.request, P.yogaMat.id, 1);
    await page.goto('/checkout');

    await applyPromo(page, 'WELCOME5');
    await expect(page.getByTestId('promo-success')).toHaveText('Promo code WELCOME5 applied');
    await expectCheckoutSummary(page, { subtotal: '$25.00', discount: '-$5.00', shipping: '$4.99', total: '$24.99' });

    await applyPromo(page, 'FREESHIP');
    await expect(page.getByTestId('promo-success')).toHaveText('Promo code FREESHIP applied');
    await expectCheckoutSummary(page, { subtotal: '$25.00', discount: '-$0.00', shipping: 'Free', total: '$25.00' });

    await page.getByTestId('shipping-method-express').check();
    await page.getByTestId('promo-apply').click();
    await expect(page.getByTestId('summary-shipping')).toHaveText('$14.99');
    await expect(page.getByTestId('summary-total')).toHaveText('$39.99');
    await expect(page.getByTestId('promo-success')).toHaveText('Promo code FREESHIP applied');
    await expect(page.getByTestId('shipping-method-express')).toBeChecked();
  },
);

test(
  '[CHK-04] rejects expired, inactive and unknown codes and blocks the order',
  meta({
    modules: ['checkout'],
    crit: 'medium',
    covers: ['checkout/logic.ts', 'checkout/routes.ts', 'checkout/service.ts', 'checkout/views/checkout.ejs'],
  }),
  async ({ page, loginAs }) => {
    await loginAs(USERS.bob);
    await addToCart(page.request, P.wirelessMouse.id, 1);
    await page.goto('/checkout');

    await applyPromo(page, 'SUMMER15');
    await expect(page.getByTestId('promo-error')).toHaveText('Promo code has expired');
    await expectCheckoutSummary(page, { subtotal: '$24.99', discount: '-$0.00', shipping: '$4.99', total: '$29.98' });

    await applyPromo(page, 'retired30');
    await expect(page.getByTestId('promo-error')).toHaveText('Promo code not found');
    await expect(page.getByTestId('promo-success')).toHaveCount(0);

    await fillShipping(page, VALID_SHIPPING);
    await page.getByTestId('promo-input').fill('NOPE');
    await page.getByTestId('checkout-submit').click();
    await expect(page.getByTestId('checkout-error')).toHaveText('Please fix the promo code before placing the order');
    await expect(page.getByTestId('promo-error')).toHaveText('Promo code not found');
    await expect(page.getByTestId('cart-count')).toHaveText('1');

    // Without the invalid code the same form goes through.
    await page.getByTestId('promo-input').fill('');
    await page.getByTestId('checkout-submit').click();
    await expect(page.getByTestId('order-confirmation')).toBeVisible();
    await expect(page.getByTestId('order-total')).toHaveText('$29.98');
  },
);

test(
  '[CHK-05] accepts a promo code at exactly the minimum order and rejects it below',
  meta({
    modules: ['checkout'],
    crit: 'medium',
    covers: ['checkout/logic.ts', 'checkout/routes.ts', 'shared/money.ts'],
  }),
  async ({ page, loginAs }) => {
    await loginAs(USERS.bob);
    await addToCart(page.request, P.plushBear.id, 1);
    await page.goto('/checkout');

    await applyPromo(page, 'FREESHIP');
    await expect(page.getByTestId('promo-error')).toHaveText('This code requires a minimum order of $15.00');
    await applyPromo(page, 'WELCOME5');
    await expect(page.getByTestId('promo-error')).toHaveText('This code requires a minimum order of $20.00');
    await expectCheckoutSummary(page, { subtotal: '$14.99', discount: '-$0.00', shipping: '$4.99', total: '$19.98' });

    // Replace the $14.99 item with one costing exactly $20.00.
    const removed = await page.request.post('/cart/remove', { form: { productId: String(P.plushBear.id) } });
    expect(removed.ok()).toBe(true);
    await addToCart(page.request, P.resistanceBands.id, 1);
    await page.goto('/checkout');

    await applyPromo(page, 'WELCOME5');
    await expect(page.getByTestId('promo-success')).toHaveText('Promo code WELCOME5 applied');
    await expectCheckoutSummary(page, { subtotal: '$20.00', discount: '-$5.00', shipping: '$4.99', total: '$19.99' });
    await applyPromo(page, 'FREESHIP');
    await expect(page.getByTestId('promo-success')).toHaveText('Promo code FREESHIP applied');
    await expectCheckoutSummary(page, { subtotal: '$20.00', discount: '-$0.00', shipping: 'Free', total: '$20.00' });
  },
);

test(
  '[CHK-06] ships free from $50.00 after discount and always charges express',
  meta({
    modules: ['checkout'],
    crit: 'high',
    covers: ['checkout/logic.ts', 'checkout/routes.ts', 'checkout/views/checkout.ejs'],
  }),
  async ({ page, loginAs }) => {
    await loginAs(USERS.bob);
    await addToCart(page.request, P.portableCharger.id, 1);
    await page.goto('/checkout');
    await expectCheckoutSummary(page, { subtotal: '$50.00', discount: '-$0.00', shipping: 'Free', total: '$50.00' });

    // The threshold applies to the amount after discount: $50.00 − $5.00 = $45.00 pays for shipping.
    await applyPromo(page, 'SAVE10');
    await expect(page.getByTestId('promo-success')).toHaveText('Promo code SAVE10 applied');
    await expectCheckoutSummary(page, { subtotal: '$50.00', discount: '-$5.00', shipping: '$4.99', total: '$49.99' });

    await page.getByTestId('shipping-method-express').check();
    await applyPromo(page, '');
    await expect(page.getByTestId('summary-shipping')).toHaveText('$14.99');
    await expectCheckoutSummary(page, { subtotal: '$50.00', discount: '-$0.00', shipping: '$14.99', total: '$64.99' });
    await expect(page.getByTestId('promo-success')).toHaveCount(0);
  },
);

test(
  '[CHK-07] places an order, decreases stock and empties the cart',
  meta({
    modules: ['checkout', 'cart', 'orders', 'product'],
    crit: 'high',
    covers: [
      'checkout/logic.ts',
      'checkout/routes.ts',
      'checkout/service.ts',
      'checkout/views/confirmation.ejs',
      'orders/logic.ts',
      'orders/views/detail.ejs',
      'cart/service.ts',
    ],
  }),
  async ({ page, loginAs }) => {
    await loginAs(USERS.bob);
    await addToCartViaUi(page, P.headphones.id, 2);
    await expectFlash(page, 'Added 2 × Noise Cancelling Headphones to cart');
    await page.getByTestId('nav-cart').click();
    await expect(page.getByTestId('cart-total')).toHaveText('$399.98');
    await page.getByTestId('cart-checkout').click();

    await expect(page).toHaveURL(/\/checkout$/);
    await fillShipping(page, VALID_SHIPPING);
    await page.getByTestId('shipping-method-express').check();
    await page.getByTestId('checkout-submit').click();

    await expect(page).toHaveURL(/\/checkout\/confirmation\/4$/);
    await expectFlash(page, 'Thank you! Your order has been placed.');
    await expect(page.getByTestId('order-number')).toHaveText('ORD-00004');
    await expect(page.getByTestId('order-total')).toHaveText('$414.97');
    await expect(page.getByTestId('order-status')).toHaveText('Pending');
    await expect(page.getByTestId('cart-count')).toHaveText('0');

    await page.getByTestId('view-order').click();
    await expect(page.getByTestId('order-status')).toHaveAttribute('data-status', 'pending');
    const item = page.getByTestId(`order-item-${P.headphones.id}`);
    await expect(item.getByTestId('order-item-quantity')).toHaveText('2');
    await expect(item.getByTestId('order-item-total')).toHaveText('$399.98');
    await expect(page.getByTestId('order-shipping-method')).toHaveText('express');
    await expect(page.getByTestId('order-shipping')).toHaveText('$14.99');
    await expect(page.getByTestId('order-total')).toHaveText('$414.97');
    await expect(page.getByTestId('order-address')).toHaveText('Bob Smith, 5 Market Square, Lviv 79000, +380671234567');

    await page.goto(`/products/${P.headphones.id}`);
    await expect(page.getByTestId('product-stock')).toHaveText('Only 2 left');
    await page.goto('/cart');
    await expect(page.getByTestId('cart-empty')).toBeVisible();
  },
);
