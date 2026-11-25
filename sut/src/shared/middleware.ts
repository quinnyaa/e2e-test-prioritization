import type { RequestHandler } from 'express';
import { HttpError } from './errors';

/** Redirects anonymous visitors to the login page and brings them back afterwards. */
export const requireAuth: RequestHandler = (req, res, next) => {
  if (!req.user) {
    res.redirect(`/login?next=${encodeURIComponent(req.originalUrl)}`);
    return;
  }
  next();
};

export const requireAdmin: RequestHandler = (req, res, next) => {
  if (!req.user) {
    res.redirect(`/login?next=${encodeURIComponent(req.originalUrl)}`);
    return;
  }
  if (req.user.role !== 'admin') {
    next(new HttpError(403, 'This page is available to administrators only.'));
    return;
  }
  next();
};
