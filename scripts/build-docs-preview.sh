#!/usr/bin/env bash
# Builds the Temporal docs site using the local ai-cookbook checkout for Vercel previews.
set -euo pipefail

COOKBOOK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DOCS_DIR="/tmp/documentation"

echo "==> Cloning temporalio/documentation..."
rm -rf "$DOCS_DIR"
git clone --depth 1 https://github.com/temporalio/documentation.git "$DOCS_DIR"

cd "$DOCS_DIR"

echo "==> Installing dependencies..."
if [ -f "yarn.lock" ]; then
  corepack enable
  yarn install --frozen-lockfile
elif [ -f "pnpm-lock.yaml" ]; then
  corepack enable
  pnpm install --frozen-lockfile
elif [ -f "package-lock.json" ]; then
  npm ci
else
  npm install
fi

# The docs repo's sync-ai-cookbook.js clones the cookbook into /tmp/ai-cookbook-sync/repo.
# We pre-populate that path with the local checkout so the build uses PR content.
# The script deletes and re-clones this dir, so we patch it to skip the clone.
echo "==> Preparing local ai-cookbook for sync..."
SYNC_DIR="/tmp/ai-cookbook-sync/repo"
rm -rf /tmp/ai-cookbook-sync
mkdir -p /tmp/ai-cookbook-sync
cp -r "$COOKBOOK_DIR" "$SYNC_DIR"
# The sync script needs git history for last_updated dates. If the Vercel checkout
# has .git, the copy already has it. Otherwise, create a minimal repo.
if [ ! -d "$SYNC_DIR/.git" ]; then
  git -C "$SYNC_DIR" init -q
  git -C "$SYNC_DIR" add -A
  git -C "$SYNC_DIR" -c user.name="build" -c user.email="build@build" commit -q -m "local"
fi

# Patch ensureRepo() to skip the clone since we already have the content in place.
# The function deletes REPO_TEMP_DIR and re-clones; we replace it with a no-op.
if [ -f "bin/sync-ai-cookbook.js" ]; then
  sed -i 's/async function ensureRepo()/async function ensureRepo() { console.log("[sync-ai-cookbook] using local ai-cookbook checkout"); return; } async function _ensureRepo_disabled()/' "bin/sync-ai-cookbook.js"
fi

echo "==> Building documentation site..."
if [ -f "yarn.lock" ]; then
  yarn build
elif [ -f "pnpm-lock.yaml" ]; then
  pnpm build
else
  npm run build
fi

echo "==> Copying build output for Vercel..."
BUILD_OUTPUT=""
for dir in build out dist .next; do
  if [ -d "$DOCS_DIR/$dir" ]; then
    BUILD_OUTPUT="$DOCS_DIR/$dir"
    break
  fi
done

if [ -z "$BUILD_OUTPUT" ]; then
  echo "ERROR: Could not find build output directory"
  exit 1
fi

cp -r "$BUILD_OUTPUT" "$COOKBOOK_DIR/.docs-build"
echo "==> Build output copied to .docs-build/"
