# Розібрані приклади пропущених перших появ (8D)

> Згенеровано `ml/error_analysis.py`. Приклади обрано вручну до виведення деталей (з урахуванням того, що вже було відомо з 8C): ручний баг з частковим відбором (MAN-04), ручний баг з порожнім відбором на «спокійному» коміті (MAN-20), мутант із 8 тестами, що падають (MUT-1062; стільки ж у MUT-1032), і два мутанти з інших модулів (MUT-593, MUT-270). Ранги й оцінки — середнє за 10 seed RF; τ — з картки; RF + B — бюджет з картки (перша поява, 0,90).

## MAN-04 — коміт 508 (manual, manual, модуль product)

- ручний баг: Low-stock threshold is exclusive: a stock of exactly 5 shows "In stock" (`src/modules/product/logic.ts`)

```diff
@@ -6,7 +6,7 @@
-  if (stock <= LOW_STOCK_THRESHOLD) return 'low_stock';
+  if (stock < LOW_STOCK_THRESHOLD) return 'low_stock';
```

- коміт `defect`, файли: `src/modules/admin/service.ts` (+2/−2), `src/modules/checkout/routes.ts` (+1/−1), `src/modules/product/logic.ts` (+2/−1)
- категорії: e; τ відібрав у середньому 13,1 тестів коміту, але не тести дефекту; найкращий тест дефекту — 14,0-й у ранжуванні RF, оцінка нижча за τ на 0,016; діф: +5/−4 рядків, 3 файл(и), hunk-и: заміна 4, додавання 1, видалення 0; відкритих дефектів на коміті: 3
- RF + τ спіймано в 3 з 10 seed; RF + B (те саме ранжування) — в 0 з 10; `combined` + B — частка перестановок 0,00 (ранг 15,0); `covers` + B — 0,55

| Тест, що падає | Ранг RF | Оцінка RF | τ | `chg_cov_files` | `hist_fail_last_1` | прогноз, мс | факт, мс | тайм-аут |
|---|---|---|---|---|---|---|---|---|
| PRD-01 | 14,0 | 0,474 | 0,490 | 1 | 0 | 205 | 3101 | так |

## MAN-20 — коміт 504 (manual, manual, модуль catalog)

- ручний баг: Catalog counter shows the number of products on the current page instead of the total (`src/modules/catalog/views/list.ejs`)

```diff
@@ -33,7 +33,7 @@
-<p class="muted" data-testid="catalog-count">Found <%= page.totalItems %> product<%= page.totalItems === 1 ? '' : 's' %></p>
+<p class="muted" data-testid="catalog-count">Found <%= page.items.length %> product<%= page.items.length === 1 ? '' : 's' %></p>
```

- коміт `defect`, файли: `src/modules/admin/logic.ts` (+2/−2), `src/modules/catalog/views/list.ejs` (+1/−1), `src/shared/password.ts` (+2/−2)
- категорії: e; τ не відібрав жодного тесту коміту (10 з 10 seed); найкращий тест дефекту — 1,4-й у ранжуванні RF, оцінка нижча за τ на 0,086; діф: +5/−5 рядків, 3 файл(и), hunk-и: заміна 4, додавання 0, видалення 0
- RF + τ спіймано в 0 з 10 seed; RF + B (те саме ранжування) — в 10 з 10; `combined` + B — частка перестановок 1,00 (ранг 7,0); `covers` + B — 1,00

| Тест, що падає | Ранг RF | Оцінка RF | τ | `chg_cov_files` | `hist_fail_last_1` | прогноз, мс | факт, мс | тайм-аут |
|---|---|---|---|---|---|---|---|---|
| CAT-01 | 7,0 | 0,354 | 0,490 | 1 | 0 | 298 | 3054 | так |
| CAT-03 | 2,0 | 0,403 | 0,490 | 1 | 0 | 270 | 3141 | так |
| CAT-05 | 2,6 | 0,396 | 0,490 | 1 | 0 | 476 | 3154 | так |

## MUT-1062 — коміт 459 (mutant, ConditionalExpression, модуль checkout)

- мутант Stryker `ConditionalExpression` у `src/modules/checkout/logic.ts`, рядок 151:

```diff
- line.quantity <= 0 || line.quantity > line.stock
+ true
```

- коміт `defect`, файли: `src/modules/auth/service.ts` (+2/−2), `src/modules/catalog/routes.ts` (+3/−3), `src/modules/checkout/logic.ts` (+1/−1), `src/modules/search/logic.ts` (+3/−3)
- категорії: e; τ відібрав у середньому 1,0 тестів коміту, але не тести дефекту; найкращий тест дефекту — 2,8-й у ранжуванні RF, оцінка нижча за τ на 0,098; діф: +9/−9 рядків, 4 файл(и), hunk-и: заміна 7, додавання 0, видалення 0; відкритих дефектів на коміті: 2
- RF + τ спіймано в 0 з 10 seed; RF + B (те саме ранжування) — в 10 з 10; `combined` + B — частка перестановок 1,00 (ранг 3,0); `covers` + B — 1,00

| Тест, що падає | Ранг RF | Оцінка RF | τ | `chg_cov_files` | `hist_fail_last_1` | прогноз, мс | факт, мс | тайм-аут |
|---|---|---|---|---|---|---|---|---|
| CHK-01 | 4,0 | 0,354 | 0,490 | 1 | 0 | 464 | 3332 | так |
| CHK-02 | 10,2 | 0,285 | 0,490 | 1 | 0 | 382 | 3256 | так |
| CHK-04 | 5,4 | 0,328 | 0,490 | 1 | 0 | 470 | 3442 | так |
| CHK-07 | 7,9 | 0,299 | 0,490 | 1 | 0 | 576 | 3399 | так |
| E2E-01 | 12,5 | 0,270 | 0,490 | 2 | 0 | 890 | 3588 | так |
| E2E-02 | 2,8 | 0,393 | 0,490 | 0 | 0 | 836 | 3257 | так |
| ORD-01 | 16,9 | 0,223 | 0,490 | 0 | 0 | 198 | 94 | ні |
| ORD-03 | 17,8 | 0,218 | 0,490 | 0 | 0 | 392 | 113 | ні |

## MUT-593 — коміт 468 (mutant, EqualityOperator, модуль cart)

- мутант Stryker `EqualityOperator` у `src/modules/cart/logic.ts`, рядок 85:

```diff
- lines.length > 0
+ lines.length <= 0
```

- коміт `defect`, файли: `src/modules/admin/logic.ts` (+10/−0), `src/modules/cart/logic.ts` (+1/−1), `src/modules/catalog/logic.ts` (+1/−1)
- категорії: e; τ відібрав у середньому 3,0 тестів коміту, але не тести дефекту; найкращий тест дефекту — 5,0-й у ранжуванні RF, оцінка нижча за τ на 0,026; діф: +12/−2 рядків, 3 файл(и), hunk-и: заміна 2, додавання 1, видалення 0; відкритих дефектів на коміті: 3
- RF + τ спіймано в 0 з 10 seed; RF + B (те саме ранжування) — в 10 з 10; `combined` + B — частка перестановок 1,00 (ранг 3,5); `covers` + B — 0,95

| Тест, що падає | Ранг RF | Оцінка RF | τ | `chg_cov_files` | `hist_fail_last_1` | прогноз, мс | факт, мс | тайм-аут |
|---|---|---|---|---|---|---|---|---|
| CART-04 | 16,8 | 0,325 | 0,490 | 1 | 0 | 118 | 3091 | так |
| CART-05 | 21,0 | 0,282 | 0,490 | 1 | 0 | 336 | 3250 | так |
| CHK-07 | 6,2 | 0,453 | 0,490 | 0 | 0 | 575 | 5260 | так |
| E2E-01 | 5,4 | 0,462 | 0,490 | 1 | 0 | 888 | 5473 | так |

## MUT-270 — коміт 476 (mutant, LogicalOperator, модуль auth)

- мутант Stryker `LogicalOperator` у `src/modules/auth/logic.ts`, рядок 40:

```diff
- value.length > EMAIL_MAX_LENGTH || !EMAIL_PATTERN.test(value)
+ value.length > EMAIL_MAX_LENGTH && !EMAIL_PATTERN.test(value)
```

- коміт `fix`, файли: `src/modules/auth/logic.ts` (+6/−1), `src/modules/catalog/logic.ts` (+6/−1), `src/modules/orders/logic.ts` (+1/−1)
- категорії: e; τ відібрав у середньому 5,0 тестів коміту, але не тести дефекту; найкращий тест дефекту — 7,1-й у ранжуванні RF, оцінка нижча за τ на 0,092; діф: +13/−3 рядків, 3 файл(и), hunk-и: заміна 3, додавання 2, видалення 0
- RF + τ спіймано в 0 з 10 seed; RF + B (те саме ранжування) — в 8 з 10; `combined` + B — частка перестановок 1,00 (ранг 6,5); `covers` + B — 0,95

| Тест, що падає | Ранг RF | Оцінка RF | τ | `chg_cov_files` | `hist_fail_last_1` | прогноз, мс | факт, мс | тайм-аут |
|---|---|---|---|---|---|---|---|---|
| AUTH-02 | 7,1 | 0,398 | 0,490 | 1 | 0 | 319 | 3186 | так |
