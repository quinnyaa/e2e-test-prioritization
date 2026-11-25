// Pure rules for the product page: stock status and availability.

export const LOW_STOCK_THRESHOLD = 5;

export type StockStatus = 'in_stock' | 'low_stock' | 'out_of_stock';

export function stockStatus(stock: number): StockStatus {
  if (stock <= 0) return 'out_of_stock';
  if (stock <= LOW_STOCK_THRESHOLD) return 'low_stock';
  return 'in_stock';
}

export function stockLabel(stock: number): string {
  switch (stockStatus(stock)) {
    case 'out_of_stock':
      return 'Out of stock';
    case 'low_stock':
      return `Only ${stock} left`;
    default:
      return 'In stock';
  }
}

export function canAddToCart(stock: number): boolean {
  return stockStatus(stock) !== 'out_of_stock';
}
