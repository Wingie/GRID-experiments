#!/bin/bash
# Clone the target codebase the experiment learns from.
# Usage: scripts/clone_target.sh [git-url] [depth]
#   depth defaults to 1 (shallow). Pass "full" (or 0) for full history, which is
#   required by the git co-change signal (semantic_ids.cochange.enabled).
set -euo pipefail

TARGET_REPO_URL="${1:-https://github.com/wingie/agentosaurus}"
DEPTH="${2:-1}"
DEST="data/target_repo"

mkdir -p data
if [ -d "$DEST/.git" ]; then
  echo "Target repo already present at $DEST."
  if [ "$DEPTH" = "full" ] || [ "$DEPTH" = "0" ]; then
    echo "Ensuring full history (unshallow if needed)."
    git -C "$DEST" fetch --unshallow 2>/dev/null || git -C "$DEST" fetch --all || true
  else
    git -C "$DEST" pull --ff-only || true
  fi
elif [ "$DEPTH" = "full" ] || [ "$DEPTH" = "0" ]; then
  echo "Cloning $TARGET_REPO_URL (full history) -> $DEST"
  git clone "$TARGET_REPO_URL" "$DEST"
else
  echo "Cloning $TARGET_REPO_URL (depth=$DEPTH) -> $DEST"
  git clone --depth "$DEPTH" "$TARGET_REPO_URL" "$DEST"
fi

echo "Done. Source files:"
find "$DEST" -type f \( -name '*.py' -o -name '*.ts' -o -name '*.tsx' -o -name '*.js' \
  -o -name '*.jsx' -o -name '*.rs' -o -name '*.go' -o -name '*.java' -o -name '*.rb' \
  -o -name '*.sol' \) | wc -l
echo "Commits available: $(git -C "$DEST" rev-list --count HEAD 2>/dev/null || echo '?')"
