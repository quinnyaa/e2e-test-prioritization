// npm run manifest — writes manifest.json (id, title, file, modules, criticality, covers) and validates it.
// Duration fields are kept from the previous manifest; `npm run stability` refreshes them.
import { MANIFEST_FILE } from '../helpers/env';
import { buildManifest, writeJson } from './lib';

const manifest = buildManifest();
writeJson(MANIFEST_FILE, manifest);

const count = (key: (t: (typeof manifest.tests)[number]) => string[]) => {
  const counts: Record<string, number> = {};
  for (const t of manifest.tests) for (const k of key(t)) counts[k] = (counts[k] ?? 0) + 1;
  return counts;
};
console.log(`manifest.json: ${manifest.testCount} tests, metadata valid`);
console.log('by id prefix:', count((t) => [t.id.split('-')[0]]));
console.log('by module tag:', count((t) => t.modules));
console.log('by criticality:', count((t) => [t.criticality]));
