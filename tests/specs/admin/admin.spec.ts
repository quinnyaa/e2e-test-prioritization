import { test, expect } from '../../fixtures/test';
import { requestStatusChange } from '../../helpers/api';
import { P, USERS } from '../../helpers/data';
import { meta } from '../../helpers/meta';
import { expectFlash } from '../../helpers/ui';

test(
  '[ADM-01] sends guests to sign-in, forbids customers and lets admins in',
  meta({
    modules: ['admin', 'auth'],
    crit: 'high',
    covers: ['shared/middleware.ts', 'shared/errors.ts', 'shared/views/error.ejs', 'admin/routes.ts', 'shared/views/header.ejs'],
  }),
  async ({ page, loginAs, pageAs }) => {
    await page.goto('/admin/products');
    await expect(page).toHaveURL(/\/login\?next=%2Fadmin%2Fproducts$/);

    await loginAs(USERS.bob);
    const res = await page.goto('/admin/orders');
    expect(res?.status()).toBe(403);
    await expect(page.getByTestId('forbidden')).toBeVisible();
    await expect(page.getByTestId('error-heading')).toHaveText('Access denied');
    await expect(page.getByTestId('error-message')).toHaveText('This page is available to administrators only.');
    await expect(page.getByTestId('nav-admin')).toHaveCount(0);
    const post = await page.request.post('/admin/orders/3/status', { form: { status: 'paid' }, maxRedirects: 0 });
    expect(post.status()).toBe(403);

    const admin = await pageAs(USERS.admin);
    await admin.goto('/catalog');
    await admin.getByTestId('nav-admin').click();
    await expect(admin).toHaveURL(/\/admin\/products$/);
    await expect(admin.getByTestId(/^admin-product-row-\d+$/)).toHaveCount(30);
    // The customer's forbidden request did not change the order.
    await admin.goto('/admin/orders');
    await expect(admin.getByTestId('admin-order-row-3').getByTestId('admin-order-status')).toHaveAttribute(
      'data-status',
      'pending',
    );
  },
);

test(
  '[ADM-02] validates the new product form at its bounds and creates the product',
  meta({
    modules: ['admin', 'product'],
    crit: 'medium',
    covers: ['admin/logic.ts', 'admin/routes.ts', 'admin/service.ts', 'admin/views/product-form.ejs', 'shared/money.ts'],
  }),
  async ({ page, loginAs }) => {
    await loginAs(USERS.admin);
    await page.goto('/admin/products');
    await page.getByTestId('admin-new-product').click();
    await expect(page.getByTestId('product-stock-input')).toHaveValue('0');

    await page.getByTestId('product-name-input').fill('X');
    await page.getByTestId('product-price-input').fill('10000.01');
    await page.getByTestId('product-stock-input').fill('10000');
    await page.getByTestId('product-save').click();
    await expect(page.getByTestId('product-name-error')).toHaveText('Name must be 2–100 characters');
    await expect(page.getByTestId('product-description-error')).toHaveText('Description is required');
    await expect(page.getByTestId('product-price-error')).toHaveText('Price must not exceed $10,000.00');
    await expect(page.getByTestId('product-stock-error')).toHaveText('Stock must be a whole number from 0 to 9999');
    await expect(page.getByTestId('product-category-error')).toHaveText('Choose a category');

    await page.getByTestId('product-name-input').fill('Desk Lamp');
    await page.getByTestId('product-description-input').fill('Adjustable LED desk lamp with a USB port.');
    await page.getByTestId('product-price-input').fill('0');
    await page.getByTestId('product-stock-input').fill('-1');
    await page.getByTestId('product-category-select').selectOption('home');
    await page.getByTestId('product-save').click();
    await expect(page.getByTestId('product-price-error')).toHaveText('Price must be greater than zero');
    await expect(page.getByTestId('product-stock-error')).toHaveText('Stock must be a whole number from 0 to 9999');
    await expect(page.getByTestId('product-name-error')).toHaveCount(0);
    await expect(page.getByTestId('product-description-error')).toHaveCount(0);
    await expect(page.getByTestId('product-category-error')).toHaveCount(0);

    // Upper bounds are inclusive.
    await page.getByTestId('product-price-input').fill('10000.00');
    await page.getByTestId('product-stock-input').fill('9999');
    await page.getByTestId('product-save').click();
    await expectFlash(page, 'Product "Desk Lamp" created (#31)');
    const row = page.getByTestId('admin-product-row-31');
    await expect(row.getByTestId('admin-product-name')).toHaveText('Desk Lamp');
    await expect(row.getByTestId('admin-product-category')).toHaveText('Home & Kitchen');
    await expect(row.getByTestId('admin-product-price')).toHaveText('$10,000.00');
    await expect(row.getByTestId('admin-product-stock')).toHaveText('9999');

    await page.goto('/products/31');
    await expect(page.getByTestId('product-title')).toHaveText('Desk Lamp');
    await expect(page.getByTestId('product-price')).toHaveText('$10,000.00');
    await expect(page.getByTestId('product-stock')).toHaveText('In stock');
  },
);

test(
  '[ADM-03] edits a product to its lower bounds and deletes a product',
  meta({
    modules: ['admin', 'product'],
    crit: 'medium',
    covers: ['admin/logic.ts', 'admin/routes.ts', 'admin/service.ts', 'admin/views/products.ejs', 'product/logic.ts'],
  }),
  async ({ page, loginAs }) => {
    await loginAs(USERS.admin);
    await page.goto('/admin/products');
    await page.getByTestId(`admin-product-row-${P.hdWebcam.id}`).getByTestId('admin-product-edit').click();
    await expect(page.getByTestId('product-name-input')).toHaveValue('HD Webcam');
    await expect(page.getByTestId('product-price-input')).toHaveValue('64.90');
    await expect(page.getByTestId('product-stock-input')).toHaveValue('7');
    await expect(page.getByTestId('product-category-select')).toHaveValue('electronics');

    await page.getByTestId('product-stock-input').fill('7.5');
    await page.getByTestId('product-save').click();
    await expect(page.getByTestId('product-stock-error')).toHaveText('Stock must be a whole number from 0 to 9999');
    await expect(page.getByTestId('product-stock-input')).toHaveValue('7.5');

    await page.getByTestId('product-price-input').fill('0.01');
    await page.getByTestId('product-stock-input').fill('0');
    await page.getByTestId('product-save').click();
    await expectFlash(page, 'Product "HD Webcam" updated');
    const row = page.getByTestId(`admin-product-row-${P.hdWebcam.id}`);
    await expect(row.getByTestId('admin-product-price')).toHaveText('$0.01');
    await expect(row.getByTestId('admin-product-stock')).toHaveText('0');

    await page.getByTestId(`admin-product-row-${P.plushBear.id}`).getByTestId('admin-product-delete').click();
    await expectFlash(page, 'Product "Plush Bear" deleted');
    await expect(page.getByTestId(`admin-product-row-${P.plushBear.id}`)).toHaveCount(0);
    await expect(page.getByTestId(/^admin-product-row-\d+$/)).toHaveCount(29);

    await page.goto(`/products/${P.hdWebcam.id}`);
    await expect(page.getByTestId('product-price')).toHaveText('$0.01');
    await expect(page.getByTestId('product-stock')).toHaveText('Out of stock');
    await expect(page.getByTestId('add-to-cart')).toBeDisabled();
    const res = await page.goto(`/products/${P.plushBear.id}`);
    expect(res?.status()).toBe(404);
  },
);

test(
  '[ADM-04] allows only valid order status transitions',
  meta({
    modules: ['admin', 'orders'],
    crit: 'high',
    covers: ['orders/logic.ts', 'orders/service.ts', 'admin/logic.ts', 'admin/routes.ts', 'admin/views/orders.ejs'],
  }),
  async ({ page, loginAs }) => {
    await loginAs(USERS.admin);
    await page.goto('/admin/orders');
    await expect(page.getByTestId('admin-order-status')).toHaveText(['Pending', 'Shipped', 'Delivered']);
    const row = (id: number) => page.getByTestId(`admin-order-row-${id}`);
    const options = (id: number) => row(id).getByTestId('admin-order-status-select').getByRole('option');
    await expect(options(3)).toHaveText(['Paid', 'Cancelled']);
    await expect(options(2)).toHaveText(['Delivered']);
    await expect(row(1).getByTestId('admin-order-status-select')).toHaveCount(0);

    await row(3).getByTestId('admin-order-status-select').selectOption('paid');
    await row(3).getByTestId('admin-order-status-save').click();
    await expectFlash(page, 'Order ORD-00003 is now Paid');
    await expect(row(3).getByTestId('admin-order-status')).toHaveAttribute('data-status', 'paid');
    await expect(options(3)).toHaveText(['Shipped', 'Cancelled']);

    // Forbidden transitions sent directly to the server are rejected with an explanation.
    await requestStatusChange(page.request, 1, 'pending');
    await page.reload();
    await expectFlash(page, 'Cannot change status from Delivered to Pending');
    await requestStatusChange(page.request, 2, 'cancelled');
    await page.reload();
    await expectFlash(page, 'Cannot change status from Shipped to Cancelled');
    await requestStatusChange(page.request, 3, 'lost');
    await page.reload();
    await expectFlash(page, 'Unknown order status');
    await expect(page.getByTestId('admin-order-status')).toHaveText(['Paid', 'Shipped', 'Delivered']);

    // Cancelling a paid order returns its item (Clean Code, stock 3) to stock.
    await row(3).getByTestId('admin-order-status-select').selectOption('cancelled');
    await row(3).getByTestId('admin-order-status-save').click();
    await expectFlash(page, 'Order ORD-00003 is now Cancelled');
    await expect(row(3).getByTestId('admin-order-status-select')).toHaveCount(0);
    await page.goto('/admin/products');
    await expect(page.getByTestId(`admin-product-row-${P.cleanCode.id}`).getByTestId('admin-product-stock')).toHaveText('4');
  },
);
