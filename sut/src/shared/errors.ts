import type { ErrorRequestHandler, RequestHandler, Response } from 'express';

export class HttpError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export function renderError(res: Response, status: number, heading: string, message: string): void {
  const testId = status === 404 ? 'not-found' : status === 403 ? 'forbidden' : 'server-error';
  res.status(status).render('shared/views/error', { heading, message, testId });
}

export const notFoundHandler: RequestHandler = (_req, res) => {
  renderError(res, 404, 'Page not found', 'The page you are looking for does not exist.');
};

export const errorHandler: ErrorRequestHandler = (err, _req, res, _next) => {
  if (err instanceof HttpError) {
    const heading = err.status === 404 ? 'Page not found' : err.status === 403 ? 'Access denied' : 'Error';
    renderError(res, err.status, heading, err.message);
    return;
  }
  console.error(err);
  renderError(res, 500, 'Something went wrong', 'Unexpected server error.');
};
