import { test, expect } from '../../fixtures/test';
import { addToCart, updateProduct } from '../../helpers/api';
import { P, USERS } from '../../helpers/data';
import { meta } from '../../helpers/meta';
import { addToCartViaUi, expectFlash } from '../../helpers/ui';

test(
  '[CART-01] adds products and updates the header counter',
  meta({
    modules: ['cart'],
    crit: 'high',
    covers: ['cart/logic.ts', 'cart/routes.ts', 'cart/service.ts', 'cart/views/cart.ejs', 'shared/views/header.ejs'],
  }),
  async ({ page }) => {
    await page.goto('/catalog');
    await expect(page.getByTestId('cart-count')).toHaveText('0');

    await addToCartViaUi(page, P.wirelessMouse.id, 2);
    await expectFlash(page, 'Added 2 × Wireless Mouse to cart');
    await expect(page.getByTestId('cart-count')).toHaveText('2');

    await addToCartViaUi(page, P.coffeeMug.id, 3);
    await expectFlash(page, 'Added 3 × Ceramic Coffee Mug to cart');
    await expect(page.getByTestId('cart-count')).toHaveText('5');

    // Adding the same product again increases its line instead of creating a new one.
    await addToCartViaUi(page, P.wirelessMouse.id, 1);
    await expectFlash(page, 'Added 1 × Wireless Mouse to cart');
    await expect(page.getByTestId('cart-count')).toHaveText('6');

    await page.getByTestId('nav-cart').click();
    await expect(page.getByTestId(/^cart-line-\d+$/)).toHaveCount(2);
    const mouse = page.getByTestId(`cart-line-${P.wirelessMouse.id}`);
    await expect(mouse.getByTestId('cart-line-name')).toHaveText('Wireless Mouse');
    await expect(mouse.getByTestId('cart-qty-input')).toHaveValue('3');
    await expect(page.getByTestId(`cart-line-${P.coffeeMug.id}`).getByTestId('cart-qty-input')).toHaveValue('3');
    await expect(page.getByTestId('cart-item-count')).toHaveText('6');
  },
);

test(
  '[CART-02] enforces the per-line limit of 10 and the available stock',
  meta({
    modules: ['cart'],
    crit: 'high',
    covers: ['cart/logic.ts', 'cart/routes.ts', 'cart/service.ts'],
  }),
  async ({ page }) => {
    await addToCartViaUi(page, P.coffeeMug.id, 10);
    await expectFlash(page, 'Added 10 × Ceramic Coffee Mug to cart');
    await addToCartViaUi(page, P.coffeeMug.id, 1);
    await expectFlash(page, 'You can add at most 10 items of one product');

    await addToCartViaUi(page, P.cleanCode.id, 4);
    await expectFlash(page, 'Only 3 items available');
    await addToCartViaUi(page, P.cleanCode.id, 2);
    await expectFlash(page, 'Added 2 × Clean Code to cart');
    await addToCartViaUi(page, P.cleanCode.id, 2);
    await expectFlash(page, 'Only 3 items available (2 already in your cart)');

    await addToCartViaUi(page, P.resistanceBands.id, 2);
    await expectFlash(page, 'Only 1 item available');
    await addToCartViaUi(page, P.wirelessMouse.id, 0);
    await expectFlash(page, 'Quantity must be a whole number of at least 1');

    await expect(page.getByTestId('cart-count')).toHaveText('12');
    await page.goto('/cart');
    await expect(page.getByTestId(/^cart-line-\d+$/)).toHaveCount(2);
    await expect(page.getByTestId(`cart-line-${P.coffeeMug.id}`).getByTestId('cart-qty-input')).toHaveValue('10');
    await expect(page.getByTestId(`cart-line-${P.cleanCode.id}`).getByTestId('cart-qty-input')).toHaveValue('2');
  },
);

test(
  '[CART-03] updates quantities, removes a line at zero and rejects invalid values',
  meta({
    modules: ['cart'],
    crit: 'medium',
    covers: ['cart/logic.ts', 'cart/routes.ts', 'cart/service.ts', 'cart/views/cart.ejs'],
  }),
  async ({ page }) => {
    await addToCart(page.request, P.wirelessMouse.id, 2);
    await addToCart(page.request, P.cleanCode.id, 1);
    await page.goto('/cart');
    const mouse = page.getByTestId(`cart-line-${P.wirelessMouse.id}`);
    const book = page.getByTestId(`cart-line-${P.cleanCode.id}`);

    await mouse.getByTestId('cart-qty-input').fill('4');
    await mouse.getByTestId('cart-update').click();
    await expectFlash(page, 'Updated Wireless Mouse quantity to 4');
    await expect(mouse.getByTestId('cart-line-total')).toHaveText('$99.96');
    await expect(page.getByTestId('cart-count')).toHaveText('5');

    await book.getByTestId('cart-qty-input').fill('4');
    await book.getByTestId('cart-update').click();
    await expectFlash(page, 'Only 3 items available');
    await expect(book.getByTestId('cart-qty-input')).toHaveValue('1');

    await mouse.getByTestId('cart-qty-input').fill('11');
    await mouse.getByTestId('cart-update').click();
    await expectFlash(page, 'You can add at most 10 items of one product');

    await mouse.getByTestId('cart-qty-input').fill('1.5');
    await mouse.getByTestId('cart-update').click();
    await expectFlash(page, 'Quantity must be a whole number');
    await expect(mouse.getByTestId('cart-qty-input')).toHaveValue('4');

    await book.getByTestId('cart-qty-input').fill('0');
    await book.getByTestId('cart-update').click();
    await expectFlash(page, 'Removed Clean Code from cart');
    await expect(book).toHaveCount(0);
    await expect(page.getByTestId('cart-count')).toHaveText('4');

    await mouse.getByTestId('cart-remove').click();
    await expectFlash(page, 'Removed Wireless Mouse from cart');
    await expect(page.getByTestId('cart-empty')).toBeVisible();
    await expect(page.getByTestId('cart-count')).toHaveText('0');
  },
);

test(
  '[CART-04] calculates line totals and the cart subtotal',
  meta({
    modules: ['cart'],
    crit: 'high',
    covers: ['cart/logic.ts', 'cart/routes.ts', 'cart/views/cart.ejs', 'shared/money.ts'],
  }),
  async ({ page }) => {
    await addToCart(page.request, P.wirelessMouse.id, 3);
    await addToCart(page.request, P.headphones.id, 4);
    await addToCart(page.request, P.runningShoes.id, 2);
    await page.goto('/cart');

    await expect(page.getByTestId('cart-line-name')).toHaveText([
      'Wireless Mouse',
      'Noise Cancelling Headphones',
      'Running Shoes',
    ]);
    await expect(page.getByTestId('cart-line-price')).toHaveText(['$24.99', '$199.99', '$120.00']);
    await expect(page.getByTestId('cart-line-total')).toHaveText(['$74.97', '$799.96', '$240.00']);
    await expect(page.getByTestId('cart-item-count')).toHaveText('9');
    await expect(page.getByTestId('cart-total')).toHaveText('$1,114.93');
    await expect(page.getByTestId('cart-count')).toHaveText('9');
    await expect(page.getByTestId('cart-stock-issue')).toHaveCount(0);
    await expect(page.getByTestId('cart-checkout')).toHaveAttribute('href', '/checkout');
  },
);

test(
  '[CART-05] warns about reduced stock and blocks checkout until the cart is fixed',
  meta({
    modules: ['cart', 'checkout'],
    crit: 'medium',
    covers: ['cart/logic.ts', 'cart/routes.ts', 'cart/views/cart.ejs', 'checkout/routes.ts'],
  }),
  async ({ page, loginAs, apiAs }) => {
    await loginAs(USERS.bob);
    await addToCart(page.request, P.hdWebcam.id, 5);
    await addToCart(page.request, P.wirelessMouse.id, 1);

    const admin = await apiAs(USERS.admin);
    const webcam = {
      name: 'HD Webcam',
      description: '1080p webcam with a built-in microphone and privacy shutter.',
      price: '64.90',
      category: 'electronics',
    };
    await updateProduct(admin, P.hdWebcam.id, { ...webcam, stock: '3' });

    await page.goto('/cart');
    await expect(page.getByTestId('cart-stock-issue')).toHaveText(['Only 3 of HD Webcam available']);
    await expect(page.getByTestId('cart-checkout')).toBeDisabled();

    await page.goto('/checkout');
    await expect(page).toHaveURL(/\/cart$/);
    await expectFlash(page, 'Some items in your cart are no longer available in the requested quantity');

    await updateProduct(admin, P.hdWebcam.id, { ...webcam, stock: '0' });
    await page.reload();
    await expect(page.getByTestId('cart-stock-issue')).toHaveText(['HD Webcam is out of stock']);

    const line = page.getByTestId(`cart-line-${P.hdWebcam.id}`);
    await line.getByTestId('cart-qty-input').fill('0');
    await line.getByTestId('cart-update').click();
    await expectFlash(page, 'Removed HD Webcam from cart');
    await expect(page.getByTestId('cart-stock-issue')).toHaveCount(0);
    await expect(page.getByTestId('cart-checkout')).toHaveAttribute('href', '/checkout');
  },
);
