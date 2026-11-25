import express from 'express';
import path from 'node:path';
import { config } from './shared/config';
import { formatDate } from './shared/clock';
import { formatPrice } from './shared/money';
import { errorHandler, notFoundHandler } from './shared/errors';
import { sessionMiddleware } from './shared/session';
import { testRouter } from './shared/test-routes';
import { authRouter } from './modules/auth/routes';
import { catalogRouter } from './modules/catalog/routes';
import { productRouter } from './modules/product/routes';
import { searchRouter } from './modules/search/routes';
import { cartCountMiddleware, cartRouter } from './modules/cart/routes';
import { checkoutRouter } from './modules/checkout/routes';
import { ordersRouter } from './modules/orders/routes';
import { adminRouter } from './modules/admin/routes';

export function createApp(): express.Express {
  const app = express();
  app.disable('x-powered-by');
  app.set('view engine', 'ejs');
  app.set('views', path.join(config.rootDir, 'src'));
  app.locals.formatPrice = formatPrice;
  app.locals.formatDate = formatDate;

  app.get('/health', (_req, res) => {
    res.json({ status: 'ok' });
  });

  if (config.env === 'test') app.use(testRouter);

  app.use(express.static(path.join(config.rootDir, 'src', 'public')));
  app.use(express.urlencoded({ extended: false }));
  app.use(sessionMiddleware);
  app.use(cartCountMiddleware);

  app.get('/', (_req, res) => res.redirect('/catalog'));
  app.use(authRouter);
  app.use(catalogRouter);
  app.use(productRouter);
  app.use(searchRouter);
  app.use(cartRouter);
  app.use(checkoutRouter);
  app.use(ordersRouter);
  app.use(adminRouter);

  app.use(notFoundHandler);
  app.use(errorHandler);
  return app;
}
