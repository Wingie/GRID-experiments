#!/usr/bin/env bash
# Link/download the RIMAPI mod into the RimWorld Mods folder and copy Core Defs into the
# repo for offline knowledge extraction. Edit RIMWORLD_DIR / RIMAPI_URL for your install.
set -euo pipefail

RIMWORLD_DIR="${RIMWORLD_DIR:-$HOME/.steam/steam/steamapps/common/RimWorld}"
RIMAPI_REPO="${RIMAPI_REPO:-https://github.com/your-org/RIMAPI}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "RimWorld dir: $RIMWORLD_DIR"
if [[ ! -d "$RIMWORLD_DIR" ]]; then
  echo "ERROR: set RIMWORLD_DIR to your RimWorld install." >&2
  exit 1
fi

# 1. Copy Core Defs for offline parsing.
DEFS_SRC="$RIMWORLD_DIR/Data/Core/Defs"
mkdir -p "$REPO_ROOT/data/rimworld_xml_defs"
cp -r "$DEFS_SRC/." "$REPO_ROOT/data/rimworld_xml_defs/"
echo "Copied Core Defs -> data/rimworld_xml_defs/"

# 2. Install RIMAPI mod.
MODS_DIR="$RIMWORLD_DIR/Mods"
if [[ ! -d "$MODS_DIR/RIMAPI" ]]; then
  git clone --depth 1 "$RIMAPI_REPO" "$MODS_DIR/RIMAPI"
  echo "Cloned RIMAPI -> $MODS_DIR/RIMAPI (enable it in the in-game mod list, then restart)."
else
  echo "RIMAPI already present at $MODS_DIR/RIMAPI"
fi

echo "Done. Verify with: curl http://127.0.0.1:7860/state"
