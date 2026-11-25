# Demo Shop (SUT)

A small server-rendered online shop used as the **system under test** for research on
ML-based prioritization of end-to-end tests. Business logic lives in pure functions
(`src/modules/*/logic.ts`), so mutants from Stryker change behaviour in meaningful ways.

Stack: Node.js 20+, TypeScript, Express 5, EJS, SQLite (`better-sqlite3`), esbuild (build), Vitest (unit tests).

## Quick start

```bash
npm install
npm start            # build → reseed the DB → listen on http://localhost:3000
```

For E2E tests run the server in test mode (enables the reset endpoint and pins the clock):

```bash
NODE_ENV=test PORT=3000 npm start
```

## Scripts

| Script | What it does |
|---|---|
| `npm start` | Bundles `src/` into `dist/` with esbuild, reseeds the database, starts the server |
| `npm run dev` | Same, plus rebuild and restart on every change in `src/` |
| `npm run build` | Bundles only (`dist/server.js`, `dist/seed.js`) |
| `npm run seed` | Resets the database to the seed state without starting the server |
| `npm run typecheck` | `tsc --noEmit` |
| `npm test` | Vitest unit tests for `logic.ts` files (`test/unit/`) |

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `3000` | HTTP port |
| `NODE_ENV` | `development` | `test` mounts `POST /__test__/reset` and fixes the clock at `2026-03-01T12:00:00Z` |
| `DB_PATH` | `data/shop.db` | SQLite file (`:memory:` also works); use a separate file per parallel server instance |
| `SEED_ON_START` | `true` | `false` keeps the existing database on start |
| `SUT_NOW` | — | ISO date that overrides the clock in any mode |

## Test support

- `GET /health` → `{"status":"ok"}` — readiness probe (no session is created).
- `POST /__test__/reset` → `{"status":"reset"}` — wipes all tables (users, sessions, carts, orders)
  and restores the seed data in a few milliseconds. Available only with `NODE_ENV=test`.
- Every interactive or asserted element has a stable `data-testid`; validation errors are rendered
  as text with their own `data-testid` (e.g. `register-email-error`, `promo-error`).
- All forms use `novalidate`, so validation always happens on the server.
- Server state is shared: run tests that call the reset endpoint serially, or start one server per worker
  with distinct `PORT` and `DB_PATH`.

## Seed data

30 products in 5 categories (6 each; 5 are out of stock, several have ≤ 5 items), 3 users, 6 promo codes
and 3 historical orders of Alice. IDs are fixed; see `src/seed/data.ts`.

### Accounts

| Email | Password | Role | Notes |
|---|---|---|---|
| `alice@example.com` | `Password123` | customer | orders ORD-00001 (delivered), ORD-00002 (shipped), ORD-00003 (pending) |
| `bob@example.com` | `Password123` | customer | no orders |
| `admin@example.com` | `Admin12345` | admin | access to `/admin` |

### Promo codes

| Code | Type | Value | Minimum order | State |
|---|---|---|---|---|
| `SAVE10` | percent | 10 % | — | valid until 2030-12-31 |
| `WELCOME5` | fixed | $5.00 | $20.00 | valid until 2030-12-31 |
| `BIG20` | percent | 20 % | $100.00 | valid until 2030-12-31 |
| `FREESHIP` | free shipping (standard only) | — | $15.00 | no expiry |
| `SUMMER15` | percent | 15 % | — | **expired** 2025-08-31 |
| `RETIRED30` | percent | 30 % | — | **inactive** |

Shipping: standard $4.99, free when the order (after discount) is **≥ $50.00**; express $14.99 always.

## Project layout

```
src/
  app.ts, server.ts          Express app and entry point
  shared/                    config, db schema, sessions, guards, money, clock, layout views
  seed/                      seed data, reset, seed CLI
  modules/<module>/
    logic.ts                 pure business rules (no DB / HTTP)
    service.ts               SQL access
    routes.ts                HTTP handlers
    views/*.ejs              templates
test/unit/                   Vitest tests for logic.ts
scripts/                     esbuild build / dev watcher
```

Modules: `auth`, `catalog`, `product`, `search`, `cart`, `checkout`, `orders`, `admin`.

## Known limitations

- Sessions are stored in SQLite and are never expired; logout starts a new session and drops its cart.
- `npm audit` reports advisories in Vitest 3 dependencies (dev-only, not shipped with the app).
  Vitest 4 fixes them but fails to install with npm 10.9 (peer-dependency resolution bug).
