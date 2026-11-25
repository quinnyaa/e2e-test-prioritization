import { test, expect } from '../../fixtures/test';
import { P } from '../../helpers/data';
import { meta } from '../../helpers/meta';

test(
  '[PRD-01] shows in-stock, low-stock and out-of-stock badges around the threshold of 5',
  meta({
    modules: ['product', 'catalog'],
    crit: 'medium',
    covers: ['product/logic.ts', 'product/routes.ts', 'product/views/detail.ejs', 'catalog/views/product-card.ejs'],
  }),
  async ({ page }) => {
    const cases = [
      { product: P.castIronSkillet, label: 'In stock', status: 'in_stock' },
      { product: P.chefsKnife, label: 'Only 5 left', status: 'low_stock' },
      { product: P.resistanceBands, label: 'Only 1 left', status: 'low_stock' },
      { product: P.usbHub, label: 'Out of stock', status: 'out_of_stock' },
    ];
    for (const { product, label, status } of cases) {
      await page.goto(`/products/${product.id}`);
      await expect(page.getByTestId('product-title')).toHaveText(product.name);
      const badge = page.getByTestId('product-stock');
      await expect(badge).toHaveText(label);
      await expect(badge).toHaveAttribute('data-stock-status', status);
    }

    // The catalog card uses the same rules.
    await page.goto('/catalog?category=home');
    const card = page.getByTestId(`product-card-${P.chefsKnife.id}`);
    await expect(card.getByTestId('product-stock')).toHaveText('Only 5 left');
    await expect(card.getByTestId('product-stock')).toHaveAttribute('data-stock-status', 'low_stock');
    const skillet = page.getByTestId(`product-card-${P.castIronSkillet.id}`);
    await expect(skillet.getByTestId('product-stock')).toHaveAttribute('data-stock-status', 'in_stock');
  },
);

test(
  '[PRD-02] disables adding an out-of-stock product to the cart',
  meta({
    modules: ['product', 'cart'],
    crit: 'high',
    covers: ['product/logic.ts', 'product/views/detail.ejs', 'cart/logic.ts', 'cart/routes.ts'],
  }),
  async ({ page }) => {
    await page.goto(`/products/${P.usbHub.id}`);
    await expect(page.getByTestId('add-to-cart')).toBeDisabled();
    await expect(page.getByTestId('quantity-input')).toBeDisabled();

    await page.goto(`/products/${P.resistanceBands.id}`);
    await expect(page.getByTestId('add-to-cart')).toBeEnabled();
    await expect(page.getByTestId('quantity-input')).toBeEnabled();

    // The server rejects the request even when the disabled form is bypassed.
    const res = await page.request.post('/cart/add', {
      form: { productId: String(P.usbHub.id), quantity: '1' },
      maxRedirects: 0,
    });
    expect(res.status()).toBe(302);
    await page.goto(`/products/${P.usbHub.id}`);
    await expect(page.getByTestId('flash-message')).toHaveText('This product is out of stock');
    await expect(page.getByTestId('cart-count')).toHaveText('0');
  },
);

test(
  '[PRD-03] shows product details and 404 for unknown or malformed ids',
  meta({
    modules: ['product'],
    crit: 'low',
    covers: ['product/routes.ts', 'product/service.ts', 'product/views/detail.ejs', 'shared/params.ts', 'shared/errors.ts', 'shared/views/error.ejs'],
  }),
  async ({ page }) => {
    await page.goto(`/products/${P.portableCharger.id}`);
    await expect(page.getByTestId('product-title')).toHaveText('Portable Charger');
    await expect(page.getByTestId('product-price')).toHaveText('$50.00');
    await expect(page.getByTestId('product-description')).toHaveText('20000 mAh power bank with two USB-C ports.');
    await expect(page.getByTestId('product-category')).toHaveText('Electronics');

    for (const id of ['999', 'abc', '0']) {
      const res = await page.goto(`/products/${id}`);
      expect(res?.status(), `status for /products/${id}`).toBe(404);
      await expect(page.getByTestId('not-found')).toBeVisible();
      await expect(page.getByTestId('error-heading')).toHaveText('Page not found');
      await expect(page.getByTestId('error-message')).toHaveText('Product not found.');
    }
  },
);
