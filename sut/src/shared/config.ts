import fs from 'node:fs';
import path from 'node:path';

function findProjectRoot(start: string): string {
  let dir = start;
  while (!fs.existsSync(path.join(dir, 'package.json'))) {
    const parent = path.dirname(dir);
    if (parent === dir) return start;
    dir = parent;
  }
  return dir;
}

const rootDir = findProjectRoot(__dirname);

export const config = {
  env: process.env.NODE_ENV ?? 'development',
  port: Number(process.env.PORT ?? 3000),
  rootDir,
  dbPath: process.env.DB_PATH ?? path.join(rootDir, 'data', 'shop.db'),
  seedOnStart: process.env.SEED_ON_START !== 'false',
};
