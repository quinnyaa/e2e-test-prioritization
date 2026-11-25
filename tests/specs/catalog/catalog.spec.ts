import { test, expect } from '../../fixtures/test';
import { updateProduct } from '../../helpers/api';
import { USERS } from '../../helpers/data';
import { meta } from '../../helpers/meta';

test(
  '[CAT-01] filters by category and stock availability',
  meta({
    modules: ['catalog'],
    crit: 'medium',
    covers: ['catalog/logic.ts', 'catalog/routes.ts', 'catalog/service.ts', 'catalog/views/list.ejs'],
  }),
  async ({ page }) => {
    await page.goto('/catalog');
    await expect(page.getByTestId('catalog-count')).toHaveText('Found 30 products');
    await expect(page.getByTestId('pagination-info')).toHaveText('Page 1 of 4');

    await page.getByTestId('filter-category').selectOption('books');
    await page.getByTestId('filter-apply').click();
    await expect(page.getByTestId('catalog-count')).toHaveText('Found 6 products');
    await expect(page).toHaveURL(/category=books/);
    const list = page.getByTestId('product-list');
    await expect(list.getByTestId('product-name')).toHaveText([
      'Clean Code',
      'Designing Data-Intensive Applications',
      'Introduction to Algorithms',
      'Refactoring',
      'Test Driven Development',
      'The Pragmatic Programmer',
    ]);
    await expect(list.getByTestId('product-category')).toHaveText(Array(6).fill('Books'));
    await expect(page.getByTestId('pagination-info')).toHaveText('Page 1 of 1');

    await page.getByTestId('filter-in-stock').check();
    await page.getByTestId('filter-apply').click();
    await expect(page.getByTestId('catalog-count')).toHaveText('Found 5 products');
    await expect(list.getByTestId('product-name')).toHaveText([
      'Clean Code',
      'Designing Data-Intensive Applications',
      'Introduction to Algorithms',
      'Test Driven Development',
      'The Pragmatic Programmer',
    ]);
    await expect(page.getByTestId('filter-category')).toHaveValue('books');
    await expect(page.getByTestId('filter-in-stock')).toBeChecked();

    // An unknown category is ignored rather than producing an empty list.
    await page.goto('/catalog?category=garden');
    await expect(page.getByTestId('catalog-count')).toHaveText('Found 30 products');
  },
);

test(
  '[CAT-02] includes products priced exactly at the range bounds',
  meta({
    modules: ['catalog'],
    crit: 'medium',
    covers: ['catalog/logic.ts', 'catalog/routes.ts', 'catalog/views/list.ejs', 'shared/money.ts'],
  }),
  async ({ page }) => {
    await page.goto('/catalog');
    await page.getByTestId('filter-min-price').fill('20');
    await page.getByTestId('filter-max-price').fill('25.0');
    await page.getByTestId('sort-select').selectOption('price_asc');
    await page.getByTestId('filter-apply').click();

    await expect(page.getByTestId('catalog-count')).toHaveText('Found 4 products');
    const list = page.getByTestId('product-list');
    await expect(list.getByTestId('product-name')).toHaveText([
      'Resistance Bands',
      'Kids Drawing Kit',
      'Wireless Mouse',
      'Yoga Mat',
    ]);
    await expect(list.getByTestId('product-price')).toHaveText(['$20.00', '$22.00', '$24.99', '$25.00']);
    await expect(page.getByTestId('filter-min-price')).toHaveValue('20.00');
    await expect(page.getByTestId('filter-max-price')).toHaveValue('25.00');

    // Narrowing by one cent on each side drops the products sitting on the bounds.
    await page.getByTestId('filter-min-price').fill('20.01');
    await page.getByTestId('filter-max-price').fill('24.99');
    await page.getByTestId('filter-apply').click();
    await expect(page.getByTestId('catalog-count')).toHaveText('Found 2 products');
    await expect(list.getByTestId('product-name')).toHaveText(['Kids Drawing Kit', 'Wireless Mouse']);
  },
);

test(
  '[CAT-03] reports invalid price filters and resets a reversed range',
  meta({
    modules: ['catalog'],
    crit: 'low',
    covers: ['catalog/logic.ts', 'catalog/views/list.ejs', 'shared/money.ts'],
  }),
  async ({ page }) => {
    await page.goto('/catalog');
    await page.getByTestId('filter-min-price').fill('50');
    await page.getByTestId('filter-max-price').fill('20');
    await page.getByTestId('filter-apply').click();

    await expect(page.getByTestId('filter-error')).toHaveText(['Minimum price cannot be greater than maximum price']);
    await expect(page.getByTestId('catalog-count')).toHaveText('Found 30 products');
    await expect(page.getByTestId('filter-min-price')).toHaveValue('');
    await expect(page.getByTestId('filter-max-price')).toHaveValue('');

    await page.getByTestId('filter-min-price').fill('ten');
    await page.getByTestId('filter-max-price').fill('12.345');
    await page.getByTestId('filter-apply').click();
    await expect(page.getByTestId('filter-error')).toHaveText([
      'Minimum price must be a valid amount',
      'Maximum price must be a valid amount',
    ]);
    await expect(page.getByTestId('catalog-count')).toHaveText('Found 30 products');

    // A range without matches is a valid empty result, not an error.
    await page.goto('/catalog?min=500');
    await expect(page.getByTestId('catalog-empty')).toHaveText('No products match the selected filters.');
    await expect(page.getByTestId('catalog-count')).toHaveText('Found 0 products');
    await expect(page.getByTestId('pagination-info')).toHaveText('Page 1 of 1');
    await expect(page.getByTestId('filter-error')).toHaveCount(0);
  },
);

test(
  '[CAT-04] sorts by price and breaks price ties by name',
  meta({
    modules: ['catalog'],
    crit: 'low',
    covers: ['catalog/logic.ts', 'catalog/routes.ts', 'catalog/views/list.ejs'],
  }),
  async ({ page, apiAs }) => {
    // The seed has no equal prices, so the admin makes Yoga Mat cost the same as Resistance Bands ($20.00).
    const admin = await apiAs(USERS.admin);
    await updateProduct(admin, 19, {
      name: 'Yoga Mat',
      description: 'Non-slip 6 mm mat with a carrying strap.',
      price: '20.00',
      stock: '18',
      category: 'sports',
    });

    await page.goto('/catalog?category=sports');
    const names = page.getByTestId('product-list').getByTestId('product-name');
    await expect(names).toHaveText([
      'Cycling Helmet',
      'Resistance Bands',
      'Running Shoes',
      'Tennis Racket',
      'Water Bottle',
      'Yoga Mat',
    ]);

    await page.getByTestId('sort-select').selectOption('price_asc');
    await page.getByTestId('filter-apply').click();
    await expect(page).toHaveURL(/sort=price_asc/);
    await expect(names).toHaveText([
      'Water Bottle',
      'Resistance Bands',
      'Yoga Mat',
      'Cycling Helmet',
      'Tennis Racket',
      'Running Shoes',
    ]);

    await page.getByTestId('sort-select').selectOption('price_desc');
    await page.getByTestId('filter-apply').click();
    await expect(page).toHaveURL(/sort=price_desc/);
    await expect(names).toHaveText([
      'Running Shoes',
      'Tennis Racket',
      'Cycling Helmet',
      'Resistance Bands',
      'Yoga Mat',
      'Water Bottle',
    ]);
    await expect(page.getByTestId('sort-select')).toHaveValue('price_desc');
  },
);

test(
  '[CAT-05] paginates by 8 products and keeps filters in page links',
  meta({
    modules: ['catalog'],
    crit: 'low',
    covers: ['catalog/logic.ts', 'catalog/routes.ts', 'catalog/views/list.ejs'],
  }),
  async ({ page }) => {
    await page.goto('/catalog');
    await page.getByTestId('filter-in-stock').check();
    await page.getByTestId('sort-select').selectOption('price_desc');
    await page.getByTestId('filter-apply').click();

    const names = page.getByTestId('product-list').getByTestId('product-name');
    await expect(page.getByTestId('catalog-count')).toHaveText('Found 25 products');
    await expect(page.getByTestId('pagination-info')).toHaveText('Page 1 of 4');
    await expect(names).toHaveCount(8);
    await expect(names.first()).toHaveText('Noise Cancelling Headphones');
    await expect(page.getByTestId('pagination-prev')).toHaveCount(0);

    await page.getByTestId('pagination-next').click();
    await expect(page.getByTestId('pagination-info')).toHaveText('Page 2 of 4');
    await expect(page).toHaveURL(/\?inStock=1&sort=price_desc&page=2$/);
    await expect(names).toHaveText([
      'Designing Data-Intensive Applications',
      'Portable Charger',
      'Building Blocks Set',
      'Cast Iron Skillet',
      'The Pragmatic Programmer',
      'Clean Code',
      'French Press',
      'Board Game Classic',
    ]);

    await page.getByTestId('pagination-next').click();
    await expect(page.getByTestId('pagination-info')).toHaveText('Page 3 of 4');
    await page.getByTestId('pagination-next').click();
    await expect(page.getByTestId('pagination-info')).toHaveText('Page 4 of 4');
    await expect(names).toHaveText(['Ceramic Coffee Mug']);
    await expect(page.getByTestId('pagination-next')).toHaveCount(0);
    await expect(page.getByTestId('filter-in-stock')).toBeChecked();
    await expect(page.getByTestId('sort-select')).toHaveValue('price_desc');

    await page.getByTestId('pagination-prev').click();
    await expect(page.getByTestId('pagination-info')).toHaveText('Page 3 of 4');
    await expect(page).toHaveURL(/\?inStock=1&sort=price_desc&page=3$/);

    // Out-of-range and malformed page numbers are clamped.
    await page.goto('/catalog?inStock=1&page=99');
    await expect(page.getByTestId('pagination-info')).toHaveText('Page 4 of 4');
    await page.goto('/catalog?page=abc');
    await expect(page.getByTestId('pagination-info')).toHaveText('Page 1 of 4');
  },
);
