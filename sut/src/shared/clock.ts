// All "current time" reads go through this module so that tests can pin the date.
// Business logic never calls Date.now() itself; it receives `now` as a parameter.

export const FIXED_TEST_NOW = '2026-03-01T12:00:00.000Z';

export function now(): Date {
  if (process.env.SUT_NOW) return new Date(process.env.SUT_NOW);
  if (process.env.NODE_ENV === 'test') return new Date(FIXED_TEST_NOW);
  return new Date();
}

export function formatDate(iso: string): string {
  return iso.slice(0, 10);
}
