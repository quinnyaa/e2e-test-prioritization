import { Router } from 'express';
import {
  catalogQueryString,
  filterProducts,
  paginate,
  parseCatalogQuery,
  SORT_LABELS,
  SORT_OPTIONS,
  sortProducts,
} from './logic';
import { stockLabel, stockStatus } from '../product/logic';
import { listCategories, listProducts } from './service';

export const catalogRouter = Router();

catalogRouter.get('/catalog', (req, res) => {
  const categories = listCategories();
  const { filters, errors } = parseCatalogQuery(
    req.query as Record<string, unknown>,
    categories.map((c) => c.slug),
  );
  const matching = sortProducts(filterProducts(listProducts(), filters), filters.sort);
  const page = paginate(matching, filters.page);

  res.render('modules/catalog/views/list', {
    title: 'Catalog',
    categories,
    filters,
    errors,
    page,
    sortOptions: SORT_OPTIONS.map((value) => ({ value, label: SORT_LABELS[value] })),
    pageUrl: (n: number) => `/catalog${catalogQueryString(filters, n)}`,
    stockStatus,
    stockLabel,
  });
});
