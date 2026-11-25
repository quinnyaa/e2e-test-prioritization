// Test metadata used as ML features: module tags, criticality and the declared test ↔ SUT file mapping.
import type { TestDetails } from '@playwright/test';

export const MODULES = ['auth', 'catalog', 'product', 'search', 'cart', 'checkout', 'orders', 'admin'] as const;
export type Module = (typeof MODULES)[number];

export const CRITICALITIES = ['high', 'medium', 'low'] as const;
export type Criticality = (typeof CRITICALITIES)[number];

export interface TestMeta {
  modules: Module[];
  crit: Criticality;
  /** SUT files the test intentionally exercises, relative to `src/modules/` (or `shared/...`, `seed/...`). */
  covers: string[];
}

export function meta({ modules, crit, covers }: TestMeta): TestDetails {
  return {
    tag: [...modules.map((m) => `@module:${m}`), `@crit:${crit}`],
    annotation: covers.map((file) => ({ type: 'covers', description: file })),
  };
}
