// Seed data of the SUT (sut/src/seed/data.ts) that the tests rely on. The SUT is reset before every test.

export interface User {
  email: string;
  password: string;
  name: string;
}

export const USERS = {
  alice: { email: 'alice@example.com', password: 'Password123', name: 'Alice Johnson' },
  bob: { email: 'bob@example.com', password: 'Password123', name: 'Bob Smith' },
  admin: { email: 'admin@example.com', password: 'Admin12345', name: 'Shop Admin' },
} satisfies Record<string, User>;

/** Products referenced by tests: id, name, price as rendered, seed stock. */
export const P = {
  wirelessMouse: { id: 1, name: 'Wireless Mouse', price: '$24.99', stock: 25 },
  mechanicalKeyboard: { id: 2, name: 'Mechanical Keyboard', price: '$89.00', stock: 12 },
  usbHub: { id: 3, name: 'USB-C Hub', price: '$39.50', stock: 0 },
  headphones: { id: 4, name: 'Noise Cancelling Headphones', price: '$199.99', stock: 4 },
  portableCharger: { id: 5, name: 'Portable Charger', price: '$50.00', stock: 30 },
  hdWebcam: { id: 6, name: 'HD Webcam', price: '$64.90', stock: 7 },
  cleanCode: { id: 8, name: 'Clean Code', price: '$37.99', stock: 3 },
  algorithms: { id: 10, name: 'Introduction to Algorithms', price: '$95.00', stock: 2 },
  coffeeMug: { id: 13, name: 'Ceramic Coffee Mug', price: '$12.50', stock: 50 },
  chefsKnife: { id: 15, name: "Chef's Knife", price: '$79.99', stock: 5 },
  castIronSkillet: { id: 16, name: 'Cast Iron Skillet', price: '$45.00', stock: 6 },
  cuttingBoard: { id: 18, name: 'Bamboo Cutting Board', price: '$19.99', stock: 40 },
  yogaMat: { id: 19, name: 'Yoga Mat', price: '$25.00', stock: 18 },
  runningShoes: { id: 20, name: 'Running Shoes', price: '$120.00', stock: 10 },
  resistanceBands: { id: 22, name: 'Resistance Bands', price: '$20.00', stock: 1 },
  buildingBlocks: { id: 25, name: 'Building Blocks Set', price: '$49.99', stock: 14 },
  boardGame: { id: 28, name: 'Board Game Classic', price: '$32.00', stock: 11 },
  plushBear: { id: 29, name: 'Plush Bear', price: '$14.99', stock: 35 },
} as const;

export interface ShippingForm {
  fullName: string;
  address: string;
  city: string;
  postalCode: string;
  phone: string;
}

export const VALID_SHIPPING: ShippingForm = {
  fullName: 'Bob Smith',
  address: '5 Market Square',
  city: 'Lviv',
  postalCode: '79000',
  phone: '+380 (67) 123-45-67',
};

/** First id the SUT assigns to a new order after a reset (seed has orders 1–3). */
export const FIRST_NEW_ORDER_ID = 4;
