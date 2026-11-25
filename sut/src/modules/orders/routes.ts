import { Router, type Request } from 'express';
import { HttpError } from '../../shared/errors';
import { parseId } from '../../shared/params';
import { requireAuth } from '../../shared/middleware';
import { setFlash } from '../../shared/session';
import { canCustomerCancel, formatOrderNumber, sortNewestFirst, STATUS_LABELS } from './logic';
import { getOrder, listUserOrders, updateOrderStatus, type OrderDetails } from './service';

export const ordersRouter = Router();
ordersRouter.use('/orders', requireAuth);

/** Loads an order of the current user; other users' orders are reported as missing. */
function loadOwnOrder(req: Request): OrderDetails {
  const id = parseId(req.params.id);
  const order = id === null ? undefined : getOrder(id);
  if (!order || order.userId !== req.user!.id) throw new HttpError(404, 'Order not found.');
  return order;
}

ordersRouter.get('/orders', (req, res) => {
  res.render('modules/orders/views/list', {
    title: 'My orders',
    orders: sortNewestFirst(listUserOrders(req.user!.id)),
    statusLabels: STATUS_LABELS,
    formatOrderNumber,
  });
});

ordersRouter.get('/orders/:id', (req, res) => {
  const order = loadOwnOrder(req);
  res.render('modules/orders/views/detail', {
    title: formatOrderNumber(order.id),
    order,
    statusLabels: STATUS_LABELS,
    canCancel: canCustomerCancel(order.status),
    formatOrderNumber,
  });
});

ordersRouter.post('/orders/:id/cancel', (req, res) => {
  const order = loadOwnOrder(req);
  if (!canCustomerCancel(order.status)) {
    setFlash(req, 'error', 'This order can no longer be cancelled');
  } else {
    updateOrderStatus(order, 'cancelled');
    setFlash(req, 'success', `Order ${formatOrderNumber(order.id)} has been cancelled`);
  }
  res.redirect(`/orders/${order.id}`);
});
