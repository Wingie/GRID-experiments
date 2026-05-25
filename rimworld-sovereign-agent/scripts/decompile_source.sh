#!/usr/bin/env bash
# Decompile Assembly-CSharp.dll into data/rimworld_source/ with ILSpy (ilspycmd).
#   scripts/decompile_source.sh /path/to/Assembly-CSharp.dll
set -euo pipefail

DLL="${1:?usage: decompile_source.sh <path-to-Assembly-CSharp.dll>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO_ROOT/data/rimworld_source"

if ! command -v ilspycmd >/dev/null 2>&1; then
  echo "Installing ilspycmd (needs .NET SDK)..." >&2
  dotnet tool install -g ilspycmd
  export PATH="$PATH:$HOME/.dotnet/tools"
fi

mkdir -p "$OUT"
echo "Decompiling $DLL -> $OUT"
ilspycmd -p -o "$OUT" "$DLL"
echo "Done. Parse with: python -m rimworld_agent.knowledge.extract_csharp"
