import { test, expect } from '../../fixtures/test';
import { placeOrder } from '../../helpers/api';
import { P, USERS } from '../../helpers/data';
import { meta } from '../../helpers/meta';
import { expectFlash } from '../../helpers/ui';

test(
  '[ORD-01] lists the order history newest first, ties broken by id',
  meta({
    modules: ['orders'],
    crit: 'medium',
    covers: ['orders/logic.ts', 'orders/routes.ts', 'orders/service.ts', 'orders/views/list.ejs', 'shared/clock.ts'],
  }),
  async ({ page, loginAs }) => {
    await loginAs(USERS.alice);
    // Both new orders get the same (pinned) creation time, so they are ordered by id.
    await placeOrder(page.request, [[P.coffeeMug.id, 1]]);
    await placeOrder(page.request, [[P.yogaMat.id, 2]]);

    await page.goto('/catalog');
    await page.getByTestId('nav-orders').click();
    await expect(page.getByTestId('order-link')).toHaveText([
      'ORD-00005',
      'ORD-00004',
      'ORD-00003',
      'ORD-00002',
      'ORD-00001',
    ]);
    await expect(page.getByTestId('order-date')).toHaveText([
      '2026-03-01',
      '2026-03-01',
      '2026-02-20',
      '2026-02-05',
      '2026-01-10',
    ]);
    await expect(page.getByTestId('order-item-count')).toHaveText(['2', '1', '1', '2', '3']);
    await expect(page.getByTestId('order-status')).toHaveText(['Pending', 'Pending', 'Pending', 'Shipped', 'Delivered']);
    await expect(page.getByTestId('order-total')).toHaveText(['$50.00', '$17.49', '$52.98', '$60.30', '$54.98']);
    await expect(page.getByTestId('order-row-1').getByTestId('order-status')).toHaveAttribute(
      'data-status',
      'delivered',
    );
  },
);

test(
  '[ORD-02] shows order details with amounts and hides other customers’ orders',
  meta({
    modules: ['orders'],
    crit: 'medium',
    covers: ['orders/logic.ts', 'orders/routes.ts', 'orders/service.ts', 'orders/views/detail.ejs', 'shared/money.ts'],
  }),
  async ({ page, loginAs, pageAs }) => {
    await loginAs(USERS.alice);
    await page.goto('/orders');
    await page.getByTestId('order-row-2').getByTestId('order-link').click();

    await expect(page).toHaveURL(/\/orders\/2$/);
    await expect(page.getByTestId('order-number')).toHaveText('ORD-00002');
    await expect(page.getByTestId('order-status')).toHaveText('Shipped');
    await expect(page.getByTestId('order-status')).toHaveAttribute('data-status', 'shipped');
    await expect(page.getByTestId('order-item-name')).toHaveText(['The Pragmatic Programmer', 'Yoga Mat']);
    await expect(page.getByTestId('order-item-price')).toHaveText(['$42.00', '$25.00']);
    await expect(page.getByTestId('order-item-total')).toHaveText(['$42.00', '$25.00']);
    await expect(page.getByTestId('order-subtotal')).toHaveText('$67.00');
    await expect(page.getByTestId('order-discount')).toHaveText('-$6.70');
    await expect(page.getByTestId('order-shipping')).toHaveText('Free');
    await expect(page.getByTestId('order-total')).toHaveText('$60.30');
    await expect(page.getByTestId('order-address')).toHaveText('Alice Johnson, 12 Green Street, Kyiv 01001, +380501234567');
    await expect(page.getByTestId('order-cancel')).toHaveCount(0);

    await page.goto('/orders/1');
    await expect(page.getByTestId('order-item-quantity')).toHaveText(['1', '2']);
    await expect(page.getByTestId('order-item-total')).toHaveText(['$24.99', '$25.00']);
    await expect(page.getByTestId('order-total')).toHaveText('$54.98');

    // Another customer gets 404 (not 403) and an empty history.
    const bob = await pageAs(USERS.bob);
    const res = await bob.goto('/orders/2');
    expect(res?.status()).toBe(404);
    await expect(bob.getByTestId('error-message')).toHaveText('Order not found.');
    await bob.goto('/orders');
    await expect(bob.getByTestId('orders-empty')).toHaveText('You have not placed any orders yet.');
  },
);

test(
  '[ORD-03] cancels a pending order and returns its items to stock',
  meta({
    modules: ['orders', 'product'],
    crit: 'high',
    covers: ['orders/logic.ts', 'orders/routes.ts', 'orders/service.ts', 'orders/views/detail.ejs', 'product/logic.ts'],
  }),
  async ({ page, loginAs }) => {
    await loginAs(USERS.alice);
    const orderId = await placeOrder(page.request, [[P.headphones.id, 3]]);
    await page.goto(`/products/${P.headphones.id}`);
    await expect(page.getByTestId('product-stock')).toHaveText('Only 1 left');

    await page.goto('/orders/3');
    await page.getByTestId('order-cancel').click();
    await expectFlash(page, 'Order ORD-00003 has been cancelled');
    await expect(page.getByTestId('order-status')).toHaveText('Cancelled');
    await expect(page.getByTestId('order-status')).toHaveAttribute('data-status', 'cancelled');
    await expect(page.getByTestId('order-cancel')).toHaveCount(0);
    await page.goto(`/products/${P.cleanCode.id}`);
    await expect(page.getByTestId('product-stock')).toHaveText('Only 4 left');

    await page.goto(`/orders/${orderId}`);
    await page.getByTestId('order-cancel').click();
    await expectFlash(page, 'Order ORD-00004 has been cancelled');
    await page.goto(`/products/${P.headphones.id}`);
    await expect(page.getByTestId('product-stock')).toHaveText('Only 4 left');

    // A shipped order cannot be cancelled even by a direct request.
    const res = await page.request.post('/orders/2/cancel', { maxRedirects: 0 });
    expect(res.status()).toBe(302);
    await page.goto('/orders/2');
    await expectFlash(page, 'This order can no longer be cancelled');
    await expect(page.getByTestId('order-status')).toHaveAttribute('data-status', 'shipped');
  },
);
