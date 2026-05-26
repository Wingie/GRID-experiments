#!/usr/bin/env bash
# Merge the QLoRA adapter, export GGUF, and write an Ollama Modelfile.
#   scripts/export_ollama.sh [adapter_dir]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ADAPTER_DIR="${1:-data/models/qlora}"

python -m rimworld_agent.training.merge_export adapter_dir="$ADAPTER_DIR"

MERGED_DIR="data/models/run_merged"
GGUF="$(ls -t "$MERGED_DIR"/*.gguf 2>/dev/null | head -1 || true)"
if [[ -z "$GGUF" ]]; then
  echo "No GGUF produced (Unsloth export may have failed; convert the merged HF model with llama.cpp)." >&2
  exit 0
fi

cat > "$MERGED_DIR/Modelfile" <<EOF
FROM $GGUF
PARAMETER temperature 0.8
PARAMETER num_ctx 4096
SYSTEM You are a RimWorld colony manager. You see the last 4 frames of your colony. Analyse the situation and provide up to 5 actions with reasoning.
EOF

echo "Wrote $MERGED_DIR/Modelfile"
echo "Create the Ollama model with: ollama create rimworld-agent -f $MERGED_DIR/Modelfile"
