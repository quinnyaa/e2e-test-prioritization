// Re-exports the shared seeded PRNG and adds the small helpers the planner needs.
import { rng } from '../scripts/lib';

export type Rng = () => number;
export { rng };

export const randInt = (rnd: Rng, min: number, max: number): number => min + Math.floor(rnd() * (max - min + 1));
export const chance = (rnd: Rng, p: number): boolean => rnd() < p;
