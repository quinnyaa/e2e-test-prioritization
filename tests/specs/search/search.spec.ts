import { test, expect } from '../../fixtures/test';
import { meta } from '../../helpers/meta';

test(
  '[SRCH-01] ranks prefix matches before word matches before inner matches',
  meta({
    modules: ['search'],
    crit: 'medium',
    covers: ['search/logic.ts', 'search/routes.ts', 'search/service.ts', 'search/views/results.ejs'],
  }),
  async ({ page }) => {
    await page.goto('/catalog');
    await page.getByTestId('search-input').fill('board');
    await page.getByTestId('search-submit').click();

    await expect(page).toHaveURL(/\/search\?q=board$/);
    await expect(page.getByTestId('search-summary')).toHaveText('3 results for “board”');
    const names = page.getByTestId('search-results').getByTestId('product-name');
    await expect(names).toHaveText(['Board Game Classic', 'Bamboo Cutting Board', 'Mechanical Keyboard']);

    // Matching ignores case, the query keeps its original case for display.
    await page.goto('/search?q=BOARD');
    await expect(page.getByTestId('search-summary')).toHaveText('3 results for “BOARD”');
    await expect(names).toHaveText(['Board Game Classic', 'Bamboo Cutting Board', 'Mechanical Keyboard']);

    // Equal rank (a word starts with the query) falls back to alphabetical order.
    await page.goto('/search?q=co');
    await expect(page.getByTestId('search-summary')).toHaveText('3 results for “co”');
    await expect(names).toHaveText(['Ceramic Coffee Mug', 'Clean Code', 'Remote Control Car']);
  },
);

test(
  '[SRCH-02] normalizes the query and requires at least 2 characters',
  meta({
    modules: ['search'],
    crit: 'low',
    covers: ['search/logic.ts', 'search/routes.ts', 'search/views/results.ejs', 'shared/views/header.ejs'],
  }),
  async ({ page }) => {
    await page.goto('/search?q=a');
    await expect(page.getByTestId('search-error')).toHaveText('Enter at least 2 characters.');
    await expect(page.getByTestId('search-results')).toHaveCount(0);

    await page.goto('/search?q=%20%20%20');
    await expect(page.getByTestId('search-hint')).toHaveText('Enter a product name to search.');

    await page.getByTestId('search-input').fill('   cast    IRON  ');
    await page.getByTestId('search-submit').click();
    await expect(page.getByTestId('search-summary')).toHaveText('1 result for “cast IRON”');
    await expect(page.getByTestId('search-results').getByTestId('product-name')).toHaveText(['Cast Iron Skillet']);
    await expect(page.getByTestId('search-input')).toHaveValue('cast IRON');
  },
);

test(
  '[SRCH-03] shows an empty result for a query without matches',
  meta({
    modules: ['search'],
    crit: 'low',
    covers: ['search/logic.ts', 'search/views/results.ejs'],
  }),
  async ({ page }) => {
    await page.goto('/search?q=laptop');
    await expect(page.getByTestId('search-empty')).toHaveText('No products found for “laptop”.');
    await expect(page.getByTestId('search-results')).toHaveCount(0);
    await expect(page.getByTestId('search-input')).toHaveValue('laptop');
  },
);
