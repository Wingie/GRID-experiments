#!/bin/bash
# Clone the target codebase the experiment learns from.
# Usage: scripts/clone_target.sh [git-url]
set -euo pipefail

TARGET_REPO_URL="${1:-https://github.com/wingie/agentosaurus}"
DEST="data/target_repo"

mkdir -p data
if [ -d "$DEST/.git" ]; then
  echo "Target repo already present at $DEST; pulling latest."
  git -C "$DEST" pull --ff-only || true
else
  echo "Cloning $TARGET_REPO_URL -> $DEST"
  git clone --depth 1 "$TARGET_REPO_URL" "$DEST"
fi

echo "Done. Source files:"
find "$DEST" -type f \( -name '*.py' -o -name '*.ts' -o -name '*.tsx' -o -name '*.js' \
  -o -name '*.jsx' -o -name '*.rs' -o -name '*.go' -o -name '*.java' -o -name '*.rb' \
  -o -name '*.sol' \) | wc -l
