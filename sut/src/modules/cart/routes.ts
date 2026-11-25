import { Router, type RequestHandler } from 'express';
import { HttpError } from '../../shared/errors';
import { parseId } from '../../shared/params';
import { setFlash } from '../../shared/session';
import {
  canCheckout,
  cartItemCount,
  cartSubtotal,
  findStockIssues,
  lineTotal,
  MAX_QUANTITY_PER_LINE,
  parseQuantity,
  validateAddQuantity,
  validateUpdateQuantity,
} from './logic';
import { getCartLines, getQuantityInCart, getStockInfo, removeLine, setLineQuantity } from './service';

export const cartRouter = Router();

/** Exposes the number of items in the cart to the header on every page. */
export const cartCountMiddleware: RequestHandler = (req, res, next) => {
  res.locals.cartCount = cartItemCount(getCartLines(req.sessionId));
  next();
};

function requireProduct(raw: unknown) {
  const id = parseId(raw);
  const product = id === null ? undefined : getStockInfo(id);
  if (!product) throw new HttpError(404, 'Product not found.');
  return product;
}

cartRouter.get('/cart', (req, res) => {
  const lines = getCartLines(req.sessionId);
  res.render('modules/cart/views/cart', {
    title: 'Cart',
    lines,
    issues: findStockIssues(lines),
    itemCount: cartItemCount(lines),
    subtotal: cartSubtotal(lines),
    canCheckout: canCheckout(lines),
    maxQuantity: MAX_QUANTITY_PER_LINE,
    lineTotal,
  });
});

cartRouter.post('/cart/add', (req, res) => {
  const product = requireProduct(req.body.productId);
  const requested = parseQuantity(req.body.quantity);
  const result = validateAddQuantity(requested, getQuantityInCart(req.sessionId, product.id), product.stock);
  if (!result.ok) {
    setFlash(req, 'error', result.error);
  } else {
    setLineQuantity(req.sessionId, product.id, result.quantity);
    setFlash(req, 'success', `Added ${requested} × ${product.name} to cart`);
  }
  res.redirect(`/products/${product.id}`);
});

cartRouter.post('/cart/update', (req, res) => {
  const product = requireProduct(req.body.productId);
  const result = validateUpdateQuantity(parseQuantity(req.body.quantity), product.stock);
  if (!result.ok) {
    setFlash(req, 'error', result.error);
  } else if (result.action === 'remove') {
    removeLine(req.sessionId, product.id);
    setFlash(req, 'success', `Removed ${product.name} from cart`);
  } else {
    setLineQuantity(req.sessionId, product.id, result.quantity);
    setFlash(req, 'success', `Updated ${product.name} quantity to ${result.quantity}`);
  }
  res.redirect('/cart');
});

cartRouter.post('/cart/remove', (req, res) => {
  const product = requireProduct(req.body.productId);
  removeLine(req.sessionId, product.id);
  setFlash(req, 'success', `Removed ${product.name} from cart`);
  res.redirect('/cart');
});
