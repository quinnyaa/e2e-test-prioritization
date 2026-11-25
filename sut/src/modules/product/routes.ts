import { Router } from 'express';
import { HttpError } from '../../shared/errors';
import { parseId } from '../../shared/params';
import { canAddToCart, stockLabel, stockStatus } from './logic';
import { getProduct } from './service';

export const productRouter = Router();

productRouter.get('/products/:id', (req, res) => {
  const id = parseId(req.params.id);
  const product = id === null ? undefined : getProduct(id);
  if (!product) throw new HttpError(404, 'Product not found.');

  res.render('modules/product/views/detail', {
    title: product.name,
    product,
    status: stockStatus(product.stock),
    label: stockLabel(product.stock),
    canAdd: canAddToCart(product.stock),
  });
});
