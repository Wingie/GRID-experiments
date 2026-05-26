#!/usr/bin/env bash
# Full pipeline: knowledge extraction -> semantic IDs -> tokenizer extension ->
# knowledge pre-training -> (mmproj) -> self-play. [CPU] steps run anywhere; later steps
# need a GPU and a running game. Stops at the first failure.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== [1] knowledge extraction =="
python -m rimworld_agent.knowledge.extract_defs
python -m rimworld_agent.knowledge.extract_csharp || echo "  (skipped C#: needs csharp extra + decompiled source)"
python -m rimworld_agent.knowledge.mine_tokens

echo "== [2] semantic IDs =="
python -m rimworld_agent.semantic_ids.assign_ids experiment=semantic_ids

echo "== [3] tokenizer extension (GPU) =="
python -m rimworld_agent.knowledge.extend_tokenizer

echo "== [4] knowledge pre-training (GPU) =="
python -m rimworld_agent.training.prepare_data experiment=knowledge_pretrain
python -m rimworld_agent.training.train_qlora  experiment=knowledge_pretrain

echo "== [5] mmproj (GPU; needs screenshots) =="
python -m rimworld_agent.vision.train_mmproj experiment=mmproj_train || echo "  (skipped mmproj: needs screenshots)"

echo "== [6] self-play (GPU + game) =="
python -m rimworld_agent.training.self_play experiment=self_play || echo "  (skipped self-play: needs running game)"

echo "Done."
