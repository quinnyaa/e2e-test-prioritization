// Bundles the TypeScript sources into dist/ with esbuild (type checking is done separately by `tsc`).
import { build } from 'esbuild';
import { pathToFileURL } from 'node:url';

export const buildOptions = {
  entryPoints: { server: 'src/server.ts', seed: 'src/seed/cli.ts' },
  bundle: true,
  platform: 'node',
  target: 'node20',
  format: 'cjs',
  packages: 'external',
  outdir: 'dist',
  sourcemap: true,
  logLevel: 'warning',
};

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  await build(buildOptions);
}
