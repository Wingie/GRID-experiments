#!/bin/bash
# Full experiment pipeline for one experiment config.
# Usage: scripts/run_all.sh [configs/experiments/<exp>.yaml] [target-git-url]
set -euo pipefail

CONFIG="${1:-configs/experiments/semid_qlora.yaml}"
TARGET_URL="${2:-https://github.com/wingie/agentosaurus}"
VARIANT="$(basename "$CONFIG" .yaml)"

echo "==> [0/8] Clone target repo"
scripts/clone_target.sh "$TARGET_URL"

echo "==> [1/8] Mine candidate tokens"
python -m src.mine_tokens "$CONFIG"

echo "==> [2/8] Analyze tokenizer (static compression estimate)"
python -m src.analyze_tokenizer "$CONFIG"

echo "==> [3/8] Learn semantic IDs (extract -> embed -> RQ-VAE -> assign)"
python -m src.semantic_ids.assign_ids "$CONFIG" || echo "  (skipped: semantic_ids disabled or deps missing)"

echo "==> [4/8] Prepare training data"
python -m src.prepare_data "$CONFIG"

echo "==> [5/8] QLoRA fine-tune (GPU required)"
python -m src.train_qlora "$CONFIG" || echo "  (skipped: no GPU / do_finetune=false)"

echo "==> [6/8] Evaluate: compression"
python -m src.eval_compression "$CONFIG"

echo "==> [7/8] Evaluate: completion + latency + semantic IDs (GPU required)"
MODEL_PATH="data/models/${VARIANT}"
python -m src.eval_completion "$CONFIG" "model_path=${MODEL_PATH}" "variant=${VARIANT}" || true
python -m src.eval_latency    "$CONFIG" "model_path=${MODEL_PATH}" "variant=${VARIANT}" || true
python -m src.eval_semantic_ids "$CONFIG" "model_path=${MODEL_PATH}" || true

echo "==> [8/8] Export merged model + GGUF for Ollama (GPU required)"
python -m src.merge_and_export "$CONFIG" "adapter_dir=${MODEL_PATH}" || true

echo "Done. Results in results/."
