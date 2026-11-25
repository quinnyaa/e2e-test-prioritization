import { Router, type Response } from 'express';
import { HttpError } from '../../shared/errors';
import { parseId } from '../../shared/params';
import { requireAdmin } from '../../shared/middleware';
import { setFlash } from '../../shared/session';
import { listCategories } from '../catalog/service';
import { allowedNextStatuses, formatOrderNumber, STATUS_LABELS } from '../orders/logic';
import { getOrder, updateOrderStatus } from '../orders/service';
import { validateProductInput, validateStatusChange, type ProductErrors, type ProductInput } from './logic';
import {
  createProduct,
  deleteProduct,
  getAdminProduct,
  listAdminProducts,
  listAllOrders,
  updateProduct,
} from './service';

export const adminRouter = Router();
adminRouter.use('/admin', requireAdmin);

function readProductInput(body: Record<string, unknown>): ProductInput {
  const text = (name: string) => (typeof body[name] === 'string' ? (body[name] as string) : '');
  return {
    name: text('name'),
    description: text('description'),
    price: text('price'),
    stock: text('stock'),
    category: text('category'),
  };
}

function renderForm(res: Response, status: number, action: string, values: ProductInput, errors: ProductErrors) {
  res.status(status).render('modules/admin/views/product-form', {
    title: action === '/admin/products' ? 'New product' : 'Edit product',
    action,
    values,
    errors,
    categories: listCategories(),
  });
}

function requireProduct(raw: string) {
  const id = parseId(raw);
  const product = id === null ? undefined : getAdminProduct(id);
  if (!product) throw new HttpError(404, 'Product not found.');
  return product;
}

adminRouter.get('/admin', (_req, res) => res.redirect('/admin/products'));

adminRouter.get('/admin/products', (_req, res) => {
  res.render('modules/admin/views/products', { title: 'Manage products', products: listAdminProducts() });
});

adminRouter.get('/admin/products/new', (_req, res) => {
  renderForm(res, 200, '/admin/products', { name: '', description: '', price: '', stock: '0', category: '' }, {});
});

adminRouter.post('/admin/products', (req, res) => {
  const input = readProductInput(req.body);
  const result = validateProductInput(input, listCategories().map((c) => c.slug));
  if (!result.ok) {
    renderForm(res, 400, '/admin/products', input, result.errors);
    return;
  }
  const id = createProduct(result.values);
  setFlash(req, 'success', `Product "${result.values.name}" created (#${id})`);
  res.redirect('/admin/products');
});

adminRouter.get('/admin/products/:id/edit', (req, res) => {
  const product = requireProduct(req.params.id);
  renderForm(
    res,
    200,
    `/admin/products/${product.id}`,
    {
      name: product.name,
      description: product.description,
      price: (product.priceCents / 100).toFixed(2),
      stock: String(product.stock),
      category: product.categorySlug,
    },
    {},
  );
});

adminRouter.post('/admin/products/:id', (req, res) => {
  const product = requireProduct(req.params.id);
  const input = readProductInput(req.body);
  const result = validateProductInput(input, listCategories().map((c) => c.slug));
  if (!result.ok) {
    renderForm(res, 400, `/admin/products/${product.id}`, input, result.errors);
    return;
  }
  updateProduct(product.id, result.values);
  setFlash(req, 'success', `Product "${result.values.name}" updated`);
  res.redirect('/admin/products');
});

adminRouter.post('/admin/products/:id/delete', (req, res) => {
  const product = requireProduct(req.params.id);
  deleteProduct(product.id);
  setFlash(req, 'success', `Product "${product.name}" deleted`);
  res.redirect('/admin/products');
});

adminRouter.get('/admin/orders', (_req, res) => {
  res.render('modules/admin/views/orders', {
    title: 'Manage orders',
    orders: listAllOrders(),
    statusLabels: STATUS_LABELS,
    allowedNextStatuses,
    formatOrderNumber,
  });
});

adminRouter.post('/admin/orders/:id/status', (req, res) => {
  const id = parseId(req.params.id);
  const order = id === null ? undefined : getOrder(id);
  if (!order) throw new HttpError(404, 'Order not found.');
  const change = validateStatusChange(order.status, req.body.status);
  if (!change.ok) {
    setFlash(req, 'error', change.error);
  } else {
    updateOrderStatus(order, change.status);
    setFlash(req, 'success', `Order ${formatOrderNumber(order.id)} is now ${STATUS_LABELS[change.status]}`);
  }
  res.redirect('/admin/orders');
});
