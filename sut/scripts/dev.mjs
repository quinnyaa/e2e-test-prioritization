// Rebuilds on every source change and restarts the server.
import { context } from 'esbuild';
import { spawn } from 'node:child_process';
import { buildOptions } from './build.mjs';

let server = null;

function restart() {
  if (server) server.kill();
  server = spawn(process.execPath, ['--enable-source-maps', 'dist/server.js'], { stdio: 'inherit' });
}

const restartPlugin = {
  name: 'restart-server',
  setup(build) {
    build.onEnd((result) => {
      if (result.errors.length === 0) restart();
    });
  },
};

const ctx = await context({ ...buildOptions, plugins: [restartPlugin] });
await ctx.watch();
console.log('Watching src/ for changes...');
