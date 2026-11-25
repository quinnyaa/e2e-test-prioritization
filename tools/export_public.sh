#!/usr/bin/env bash
# Export the reproducible part of the project (code, data, models, results, stage-8 protocol)
# into a separate folder that is a git repository for the public GitHub copy.
# The thesis text, notes with drafts, university templates and local environments are NOT exported.
#
#   bash tools/export_public.sh            # -> public-repo/ (created or refreshed)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/public-repo"
mkdir -p "$OUT"

COMMON_EXCLUDES=(
  --exclude '.git' --exclude 'node_modules' --exclude '.venv' --exclude '__pycache__'
  --exclude '.DS_Store' --exclude '*.pyc' --exclude '.env' --exclude '.env.*'
  --exclude 'test-results' --exclude 'playwright-report' --exclude '.stryker-tmp'
)

sync() {  # sync <src dir> [extra rsync args...]
  local src="$1"; shift
  mkdir -p "$OUT/$src"
  rsync -a --delete "${COMMON_EXCLUDES[@]}" "$@" "$ROOT/$src/" "$OUT/$src/"
}

# System under test: sources only (no build output, no runtime database).
sync sut --exclude 'dist' --exclude 'data' --exclude 'coverage'
# Playwright suite: no run outputs.
sync tests --exclude 'results'
# Mutation tooling and simulator: no scratch folders and Stryker reports.
sync mutation --exclude 'work' --exclude 'reports'
# ML code, frozen models and results (ml/models-prod is a local artefact).
sync ml --exclude 'models-prod'

# Data: run logs, plans, datasets; the generated git histories go as bundles.
for d in pilot history history-600 dataset demo-stage7; do
  sync "data/$d" --exclude 'repo'
done
for d in history history-600 demo-stage7; do
  repo="$ROOT/data/$d/repo"
  if [ -d "$repo/.git" ]; then
    git -C "$repo" bundle create "$OUT/data/$d/repo.bundle" --all >/dev/null 2>&1
  fi
done

# Local absolute paths in the demo logs: project root -> ".", home -> "~".
grep -rIl "$HOME" "$OUT/data" 2>/dev/null | while IFS= read -r f; do
  ROOT="$ROOT" HOME_DIR="$HOME" perl -pi -e 's/\Q$ENV{ROOT}\E/./g; s/\Q$ENV{HOME_DIR}\E/~/g' "$f"
done

# Stage-8 preregistration protocol (read by ml/check_frozen.py).
mkdir -p "$OUT/notes"
cp "$ROOT/notes/stage8-protocol.md" "$OUT/notes/"

cp "$ROOT/tools/public/README.md" "$OUT/README.md"
cp "$ROOT/tools/public/gitignore" "$OUT/.gitignore"
mkdir -p "$OUT/tools" && cp "$ROOT/tools/export_public.sh" "$OUT/tools/"

echo "exported to $OUT ($(du -sh "$OUT" --exclude .git 2>/dev/null | cut -f1 || du -sh "$OUT" | cut -f1))"
