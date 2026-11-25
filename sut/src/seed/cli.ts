import { getDb } from '../shared/db';
import { config } from '../shared/config';
import { resetDatabase } from './reset';

resetDatabase(getDb());
console.log(`Database seeded: ${config.dbPath}`);
