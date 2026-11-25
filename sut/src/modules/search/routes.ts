import { Router } from 'express';
import { stockLabel, stockStatus } from '../product/logic';
import { MIN_QUERY_LENGTH, searchProducts } from './logic';
import { listSearchableProducts } from './service';

export const searchRouter = Router();

searchRouter.get('/search', (req, res) => {
  const result = searchProducts(listSearchableProducts(), req.query.q);
  res.render('modules/search/views/results', {
    title: 'Search',
    result,
    minLength: MIN_QUERY_LENGTH,
    stockStatus,
    stockLabel,
  });
});
