// Fixed seed data. Do not use randomness or the current date here: every reset must
// produce exactly the same database so that E2E test runs are reproducible.

export interface SeedProduct {
  id: number;
  category: string;
  name: string;
  priceCents: number;
  stock: number;
  description: string;
}

export const categories = [
  { id: 1, slug: 'electronics', name: 'Electronics' },
  { id: 2, slug: 'books', name: 'Books' },
  { id: 3, slug: 'home', name: 'Home & Kitchen' },
  { id: 4, slug: 'sports', name: 'Sports' },
  { id: 5, slug: 'toys', name: 'Toys' },
];

export const products: SeedProduct[] = [
  { id: 1, category: 'electronics', name: 'Wireless Mouse', priceCents: 2499, stock: 25, description: 'Compact wireless mouse with a silent click and 12-month battery life.' },
  { id: 2, category: 'electronics', name: 'Mechanical Keyboard', priceCents: 8900, stock: 12, description: 'Tenkeyless keyboard with tactile switches and white backlight.' },
  { id: 3, category: 'electronics', name: 'USB-C Hub', priceCents: 3950, stock: 0, description: 'Seven-port hub with HDMI, SD card reader and 100 W pass-through charging.' },
  { id: 4, category: 'electronics', name: 'Noise Cancelling Headphones', priceCents: 19999, stock: 4, description: 'Over-ear headphones with active noise cancelling and 30 hours of playback.' },
  { id: 5, category: 'electronics', name: 'Portable Charger', priceCents: 5000, stock: 30, description: '20000 mAh power bank with two USB-C ports.' },
  { id: 6, category: 'electronics', name: 'HD Webcam', priceCents: 6490, stock: 7, description: '1080p webcam with a built-in microphone and privacy shutter.' },
  { id: 7, category: 'books', name: 'The Pragmatic Programmer', priceCents: 4200, stock: 15, description: 'Classic book about the craft of software development.' },
  { id: 8, category: 'books', name: 'Clean Code', priceCents: 3799, stock: 3, description: 'A handbook of agile software craftsmanship.' },
  { id: 9, category: 'books', name: 'Designing Data-Intensive Applications', priceCents: 5499, stock: 9, description: 'The big ideas behind reliable, scalable and maintainable systems.' },
  { id: 10, category: 'books', name: 'Introduction to Algorithms', priceCents: 9500, stock: 2, description: 'Comprehensive textbook on algorithms and data structures.' },
  { id: 11, category: 'books', name: 'Refactoring', priceCents: 4750, stock: 0, description: 'Improving the design of existing code, second edition.' },
  { id: 12, category: 'books', name: 'Test Driven Development', priceCents: 2999, stock: 20, description: 'Learn to write tests first and let them drive the design.' },
  { id: 13, category: 'home', name: 'Ceramic Coffee Mug', priceCents: 1250, stock: 50, description: '350 ml stoneware mug, dishwasher safe.' },
  { id: 14, category: 'home', name: 'French Press', priceCents: 3400, stock: 8, description: 'Glass French press for four cups of coffee.' },
  { id: 15, category: 'home', name: "Chef's Knife", priceCents: 7999, stock: 5, description: '20 cm stainless steel knife with an ergonomic handle.' },
  { id: 16, category: 'home', name: 'Cast Iron Skillet', priceCents: 4500, stock: 6, description: 'Pre-seasoned 26 cm skillet for stove and oven.' },
  { id: 17, category: 'home', name: 'Electric Kettle', priceCents: 3999, stock: 0, description: '1.7 l kettle with automatic shut-off.' },
  { id: 18, category: 'home', name: 'Bamboo Cutting Board', priceCents: 1999, stock: 40, description: 'Durable bamboo board with a juice groove.' },
  { id: 19, category: 'sports', name: 'Yoga Mat', priceCents: 2500, stock: 18, description: 'Non-slip 6 mm mat with a carrying strap.' },
  { id: 20, category: 'sports', name: 'Running Shoes', priceCents: 12000, stock: 10, description: 'Lightweight running shoes with responsive cushioning.' },
  { id: 21, category: 'sports', name: 'Water Bottle', priceCents: 1599, stock: 60, description: 'Insulated 750 ml steel bottle that keeps drinks cold for 24 hours.' },
  { id: 22, category: 'sports', name: 'Resistance Bands', priceCents: 2000, stock: 1, description: 'Set of five latex bands with different resistance levels.' },
  { id: 23, category: 'sports', name: 'Tennis Racket', priceCents: 9999, stock: 4, description: 'Graphite racket for intermediate players.' },
  { id: 24, category: 'sports', name: 'Cycling Helmet', priceCents: 5900, stock: 0, description: 'Ventilated helmet with an adjustable fit system.' },
  { id: 25, category: 'toys', name: 'Building Blocks Set', priceCents: 4999, stock: 14, description: '500 colourful blocks compatible with major brands.' },
  { id: 26, category: 'toys', name: 'Puzzle 1000 Pieces', priceCents: 1850, stock: 22, description: 'Landscape jigsaw puzzle for the whole family.' },
  { id: 27, category: 'toys', name: 'Remote Control Car', priceCents: 7500, stock: 3, description: 'Off-road RC car with a rechargeable battery.' },
  { id: 28, category: 'toys', name: 'Board Game Classic', priceCents: 3200, stock: 11, description: 'Strategy board game for two to four players.' },
  { id: 29, category: 'toys', name: 'Plush Bear', priceCents: 1499, stock: 35, description: 'Soft 30 cm teddy bear.' },
  { id: 30, category: 'toys', name: 'Kids Drawing Kit', priceCents: 2200, stock: 0, description: 'Markers, crayons and a sketchbook in a carry case.' },
];

export const users = [
  { id: 1, email: 'alice@example.com', name: 'Alice Johnson', password: 'Password123', role: 'customer', salt: 'seed-salt-alice' },
  { id: 2, email: 'bob@example.com', name: 'Bob Smith', password: 'Password123', role: 'customer', salt: 'seed-salt-bob' },
  { id: 3, email: 'admin@example.com', name: 'Shop Admin', password: 'Admin12345', role: 'admin', salt: 'seed-salt-admin' },
] as const;

export const promoCodes = [
  { code: 'SAVE10', type: 'percent', value: 10, minOrderCents: 0, expiresAt: '2030-12-31T23:59:59.000Z', active: 1 },
  { code: 'WELCOME5', type: 'fixed', value: 500, minOrderCents: 2000, expiresAt: '2030-12-31T23:59:59.000Z', active: 1 },
  { code: 'BIG20', type: 'percent', value: 20, minOrderCents: 10000, expiresAt: '2030-12-31T23:59:59.000Z', active: 1 },
  { code: 'FREESHIP', type: 'free_shipping', value: 0, minOrderCents: 1500, expiresAt: null, active: 1 },
  { code: 'SUMMER15', type: 'percent', value: 15, minOrderCents: 0, expiresAt: '2025-08-31T23:59:59.000Z', active: 1 },
  { code: 'RETIRED30', type: 'percent', value: 30, minOrderCents: 0, expiresAt: null, active: 0 },
] as const;

const aliceAddress = {
  fullName: 'Alice Johnson',
  address: '12 Green Street',
  city: 'Kyiv',
  postalCode: '01001',
  phone: '+380501234567',
};

export const orders = [
  {
    id: 1, userId: 1, status: 'delivered', createdAt: '2026-01-10T09:30:00.000Z',
    subtotalCents: 4999, discountCents: 0, shippingCents: 499, totalCents: 5498,
    promoCode: null, shippingMethod: 'standard', ...aliceAddress,
    items: [
      { productId: 1, name: 'Wireless Mouse', unitPriceCents: 2499, quantity: 1 },
      { productId: 13, name: 'Ceramic Coffee Mug', unitPriceCents: 1250, quantity: 2 },
    ],
  },
  {
    id: 2, userId: 1, status: 'shipped', createdAt: '2026-02-05T14:00:00.000Z',
    subtotalCents: 6700, discountCents: 670, shippingCents: 0, totalCents: 6030,
    promoCode: 'SAVE10', shippingMethod: 'standard', ...aliceAddress,
    items: [
      { productId: 7, name: 'The Pragmatic Programmer', unitPriceCents: 4200, quantity: 1 },
      { productId: 19, name: 'Yoga Mat', unitPriceCents: 2500, quantity: 1 },
    ],
  },
  {
    id: 3, userId: 1, status: 'pending', createdAt: '2026-02-20T18:45:00.000Z',
    subtotalCents: 3799, discountCents: 0, shippingCents: 1499, totalCents: 5298,
    promoCode: null, shippingMethod: 'express', ...aliceAddress,
    items: [{ productId: 8, name: 'Clean Code', unitPriceCents: 3799, quantity: 1 }],
  },
];

export const SEED_CREATED_AT = '2026-01-01T00:00:00.000Z';
