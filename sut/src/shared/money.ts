// Money is stored and calculated in integer cents to avoid floating point errors.

export function formatPrice(cents: number): string {
  const sign = cents < 0 ? '-' : '';
  const abs = Math.abs(cents);
  const dollars = Math.floor(abs / 100);
  const rest = String(abs % 100).padStart(2, '0');
  const grouped = String(dollars).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  return `${sign}$${grouped}.${rest}`;
}

/** Parses a user-entered dollar amount ("12", "12.5", "12.50") into cents; null if invalid. */
export function parseMoney(input: string | undefined | null): number | null {
  if (input === undefined || input === null) return null;
  const value = input.trim();
  if (!/^\d+(\.\d{1,2})?$/.test(value)) return null;
  return Math.round(Number(value) * 100);
}
