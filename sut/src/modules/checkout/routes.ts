import { Router, type Request, type Response } from 'express';
import { now } from '../../shared/clock';
import { HttpError } from '../../shared/errors';
import { parseId } from '../../shared/params';
import { requireAuth } from '../../shared/middleware';
import { setFlash } from '../../shared/session';
import { cartSubtotal, findStockIssues, lineTotal, type CartLine } from '../cart/logic';
import { getCartLines } from '../cart/service';
import { formatOrderNumber, STATUS_LABELS } from '../orders/logic';
import {
  calculateTotals,
  evaluatePromo,
  NO_PROMO,
  normalizePromoCode,
  parseShippingMethod,
  SHIPPING_METHODS,
  SHIPPING_RATES,
  validateShippingForm,
  type PromoResult,
  type ShippingErrors,
  type ShippingForm,
  type ShippingMethod,
} from './logic';
import { getOrderSummary, getPromo, placeOrder, StockChangedError } from './service';

export const checkoutRouter = Router();
checkoutRouter.use('/checkout', requireAuth);

interface CheckoutState {
  shipping: ShippingForm;
  method: ShippingMethod;
  promoCode: string;
  promo: PromoResult;
}

function text(body: Record<string, unknown>, name: string): string {
  const value = body[name];
  return typeof value === 'string' ? value : '';
}

function readState(req: Request, lines: CartLine[]): CheckoutState {
  const body = req.body as Record<string, unknown>;
  const promoCode = normalizePromoCode(body.promoCode);
  return {
    shipping: {
      fullName: text(body, 'fullName'),
      address: text(body, 'address'),
      city: text(body, 'city'),
      postalCode: text(body, 'postalCode'),
      phone: text(body, 'phone'),
    },
    method: parseShippingMethod(body.shippingMethod),
    promoCode,
    promo: evaluatePromo(getPromo(promoCode), promoCode, cartSubtotal(lines), now()),
  };
}

/** Returns the cart lines, or redirects to the cart when checkout is not possible. */
function loadCheckoutCart(req: Request, res: Response): CartLine[] | null {
  const lines = getCartLines(req.sessionId);
  if (lines.length === 0) {
    setFlash(req, 'error', 'Your cart is empty');
    res.redirect('/cart');
    return null;
  }
  if (findStockIssues(lines).length > 0) {
    setFlash(req, 'error', 'Some items in your cart are no longer available in the requested quantity');
    res.redirect('/cart');
    return null;
  }
  return lines;
}

function renderCheckout(
  res: Response,
  status: number,
  lines: CartLine[],
  state: CheckoutState,
  errors: ShippingErrors & { form?: string },
  promoApplied: boolean,
): void {
  res.status(status).render('modules/checkout/views/checkout', {
    title: 'Checkout',
    lines,
    state,
    errors,
    promoApplied,
    totals: calculateTotals(cartSubtotal(lines), state.promo, state.method),
    methods: SHIPPING_METHODS.map((m) => ({ value: m, ...SHIPPING_RATES[m] })),
    lineTotal,
  });
}

checkoutRouter.get('/checkout', (req, res) => {
  const lines = loadCheckoutCart(req, res);
  if (!lines) return;
  const state: CheckoutState = {
    shipping: { fullName: req.user?.name ?? '', address: '', city: '', postalCode: '', phone: '' },
    method: 'standard',
    promoCode: '',
    promo: NO_PROMO,
  };
  renderCheckout(res, 200, lines, state, {}, false);
});

/** Recalculates totals for the selected promo code and shipping method without placing the order. */
checkoutRouter.post('/checkout/preview', (req, res) => {
  const lines = loadCheckoutCart(req, res);
  if (!lines) return;
  const state = readState(req, lines);
  renderCheckout(res, state.promo.ok ? 200 : 400, lines, state, {}, state.promoCode !== '');
});

checkoutRouter.post('/checkout', (req, res) => {
  const lines = loadCheckoutCart(req, res);
  if (!lines) return;
  const state = readState(req, lines);
  const errors: ShippingErrors & { form?: string } = validateShippingForm(state.shipping);
  if (!state.promo.ok) errors.form = 'Please fix the promo code before placing the order';
  if (Object.keys(errors).length > 0) {
    renderCheckout(res, 400, lines, state, errors, state.promoCode !== '');
    return;
  }

  let orderId: number;
  try {
    orderId = placeOrder({
      userId: req.user!.id,
      sessionId: req.sessionId,
      lines,
      totals: calculateTotals(cartSubtotal(lines), state.promo, state.method),
      promoCode: state.promo.ok ? state.promo.code : null,
      method: state.method,
      shipping: state.shipping,
      createdAt: now().toISOString(),
    });
  } catch (err) {
    if (!(err instanceof StockChangedError)) throw err;
    setFlash(req, 'error', 'Some items in your cart are no longer available in the requested quantity');
    res.redirect('/cart');
    return;
  }
  setFlash(req, 'success', 'Thank you! Your order has been placed.');
  res.redirect(`/checkout/confirmation/${orderId}`);
});

checkoutRouter.get('/checkout/confirmation/:id', (req, res) => {
  const id = parseId(req.params.id);
  const order = id === null ? undefined : getOrderSummary(id, req.user!.id);
  if (!order) throw new HttpError(404, 'Order not found.');
  res.render('modules/checkout/views/confirmation', {
    title: 'Order placed',
    order,
    orderNumber: formatOrderNumber(order.id),
    statusLabel: STATUS_LABELS[order.status],
  });
});
