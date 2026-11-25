import { createApp } from './app';
import { config } from './shared/config';
import { getDb } from './shared/db';
import { resetDatabase } from './seed/reset';

if (config.seedOnStart) resetDatabase(getDb());

const app = createApp();

app.listen(config.port, () => {
  console.log(`Shop is running on http://localhost:${config.port} (NODE_ENV=${config.env})`);
});
