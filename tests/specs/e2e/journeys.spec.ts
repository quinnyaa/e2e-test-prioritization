import { test, expect } from '../../fixtures/test';
import { P, USERS, VALID_SHIPPING } from '../../helpers/data';
import { meta } from '../../helpers/meta';
import { addToCartViaUi, applyPromo, expectCheckoutSummary, expectFlash, fillShipping } from '../../helpers/ui';

test(
  '[E2E-01] guest registers, finds a product, buys it with a promo code and sees it in the history',
  meta({
    modules: ['auth', 'search', 'product', 'cart', 'checkout', 'orders'],
    crit: 'high',
    covers: [
      'auth/routes.ts',
      'search/logic.ts',
      'product/routes.ts',
      'cart/logic.ts',
      'cart/routes.ts',
      'checkout/logic.ts',
      'checkout/routes.ts',
      'checkout/service.ts',
      'orders/routes.ts',
      'orders/views/list.ejs',
    ],
  }),
  async ({ page }) => {
    await page.goto('/catalog');
    await page.getByTestId('nav-register').click();
    await page.getByTestId('register-name').fill('Olena Shevchenko');
    await page.getByTestId('register-email').fill('olena@example.com');
    await page.getByTestId('register-password').fill('Kyiv2026pass');
    await page.getByTestId('register-confirm-password').fill('Kyiv2026pass');
    await page.getByTestId('register-submit').click();
    await expectFlash(page, 'Welcome, Olena Shevchenko! Your account has been created.');

    await page.getByTestId('search-input').fill('board');
    await page.getByTestId('search-submit').click();
    await expect(page.getByTestId('search-summary')).toHaveText('3 results for “board”');
    await page.getByTestId(`product-card-${P.cuttingBoard.id}`).getByTestId('product-name').click();
    await expect(page.getByTestId('product-title')).toHaveText('Bamboo Cutting Board');
    await page.getByTestId('quantity-input').fill('3');
    await page.getByTestId('add-to-cart').click();
    await expectFlash(page, 'Added 3 × Bamboo Cutting Board to cart');
    await expect(page.getByTestId('cart-count')).toHaveText('3');

    await page.getByTestId('nav-cart').click();
    await expect(page.getByTestId('cart-total')).toHaveText('$59.97');
    await page.getByTestId('cart-checkout').click();
    await expect(page.getByTestId('shipping-full-name')).toHaveValue('Olena Shevchenko');
    await fillShipping(page, {
      fullName: 'Olena Shevchenko',
      address: '7 Khreshchatyk Street',
      city: 'Kyiv',
      postalCode: '01001',
      phone: '+380 44 123 4567',
    });
    // 10 % of $59.97 rounds to $6.00; $53.97 after discount still ships free.
    await applyPromo(page, 'save10');
    await expect(page.getByTestId('promo-success')).toHaveText('Promo code SAVE10 applied');
    await expectCheckoutSummary(page, { subtotal: '$59.97', discount: '-$6.00', shipping: 'Free', total: '$53.97' });
    await page.getByTestId('checkout-submit').click();

    await expect(page.getByTestId('order-number')).toHaveText('ORD-00004');
    await expect(page.getByTestId('order-total')).toHaveText('$53.97');
    await page.getByTestId('nav-orders').click();
    await expect(page.getByTestId('order-link')).toHaveText(['ORD-00004']);
    await expect(page.getByTestId('order-item-count')).toHaveText(['3']);
    await expect(page.getByTestId('order-status')).toHaveText(['Pending']);
    await expect(page.getByTestId('order-total')).toHaveText(['$53.97']);

    await page.getByTestId('order-link').click();
    await expect(page.getByTestId('order-discount')).toHaveText('-$6.00');
    await expect(page.getByTestId('order-shipping')).toHaveText('Free');
    await expect(page.getByTestId('order-address')).toHaveText(
      'Olena Shevchenko, 7 Khreshchatyk Street, Kyiv 01001, +380441234567',
    );
  },
);

test(
  '[E2E-02] admin moves a new order through its statuses and the customer sees each change',
  meta({
    modules: ['checkout', 'orders', 'admin'],
    crit: 'high',
    covers: [
      'checkout/routes.ts',
      'checkout/service.ts',
      'orders/logic.ts',
      'orders/routes.ts',
      'orders/service.ts',
      'orders/views/list.ejs',
      'orders/views/detail.ejs',
      'admin/logic.ts',
      'admin/routes.ts',
      'admin/views/orders.ejs',
    ],
  }),
  async ({ page, loginAs, pageAs }) => {
    await loginAs(USERS.alice);
    await addToCartViaUi(page, P.cleanCode.id, 2);
    await expectFlash(page, 'Added 2 × Clean Code to cart');
    await page.goto('/checkout');
    await fillShipping(page, VALID_SHIPPING);
    await page.getByTestId('checkout-submit').click();
    await expect(page.getByTestId('order-number')).toHaveText('ORD-00004');
    await expect(page.getByTestId('order-total')).toHaveText('$75.98');

    const admin = await pageAs(USERS.admin);
    await admin.goto('/admin/orders');
    const adminRow = admin.getByTestId('admin-order-row-4');
    await expect(adminRow.getByTestId('admin-order-status')).toHaveText('Pending');
    const advance = async (status: string, label: string) => {
      await adminRow.getByTestId('admin-order-status-select').selectOption(status);
      await adminRow.getByTestId('admin-order-status-save').click();
      await expectFlash(admin, `Order ORD-00004 is now ${label}`);
    };
    await advance('paid', 'Paid');
    await advance('shipped', 'Shipped');

    await page.getByTestId('nav-orders').click();
    const row = page.getByTestId('order-row-4');
    await expect(row.getByTestId('order-status')).toHaveText('Shipped');
    await expect(row.getByTestId('order-status')).toHaveAttribute('data-status', 'shipped');
    await row.getByTestId('order-link').click();
    await expect(page.getByTestId('order-status')).toHaveAttribute('data-status', 'shipped');
    await expect(page.getByTestId('order-cancel')).toHaveCount(0);

    await advance('delivered', 'Delivered');
    await expect(adminRow.getByTestId('admin-order-status-select')).toHaveCount(0);
    await page.reload();
    await expect(page.getByTestId('order-status')).toHaveText('Delivered');
    await expect(page.getByTestId('order-status')).toHaveAttribute('data-status', 'delivered');

    await page.goto(`/products/${P.cleanCode.id}`);
    await expect(page.getByTestId('product-stock')).toHaveText('Only 1 left');
  },
);
