#!/bin/bash
# Create an Ollama Modelfile for an exported GGUF and import it.
# Usage: scripts/export_ollama.sh <path/to/model.gguf> [model-name]
set -euo pipefail

GGUF_PATH="${1:?usage: export_ollama.sh <model.gguf> [name]}"
MODEL_NAME="${2:-vocab-extend-qlora}"
MODELFILE="$(dirname "$GGUF_PATH")/Modelfile"

cat > "$MODELFILE" <<EOF
FROM ${GGUF_PATH}
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER num_ctx 2048
TEMPLATE """{{ .Prompt }}"""
EOF

echo "Wrote $MODELFILE"
ollama create "$MODEL_NAME" -f "$MODELFILE"
echo "Imported Ollama model: $MODEL_NAME"
echo "Try: ollama run $MODEL_NAME 'def parse_config('"
