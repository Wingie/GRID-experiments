# Training guide: zero → playing agent

End-to-end on a single RTX 3090. Steps marked **[CPU]** run anywhere; **[GPU]** needs CUDA;
**[GAME]** needs a running RimWorld + RIMAPI.

## 0. Install

```bash
pip install -e ../vocab-extend-qlora     # the reused tokenizer/RQ-VAE/QLoRA core (import src.*)
pip install -e .                          # this package (rimworld_agent)
pip install -r requirements.txt           # full stack; or the offline subset (see README)
```

Set `paths.veq_path` (or `$VEQ_PATH`) if `vocab-extend-qlora` is not the sibling directory.

## 1. Knowledge extraction **[CPU]**

```bash
# Copy Defs, decompile C#, scrape wiki first — see RIMWORLD_SETUP.md.
python -m rimworld_agent.knowledge.extract_defs        # -> results/entities.json
python -m rimworld_agent.knowledge.extract_csharp      # -> results/csharp_entities.json  (needs `csharp` extra)
python -m rimworld_agent.knowledge.scrape_wiki         # -> data/rimworld_wiki/*.md        (needs network)
python -m rimworld_agent.knowledge.mine_tokens         # -> results/candidates.json
```

## 2. Semantic IDs **[CPU]**

```bash
python -m rimworld_agent.semantic_ids.assign_ids experiment=semantic_ids
#   -> data/models/rqvae.pt, results/sid_assignments.json
python -m rimworld_agent.semantic_ids.visualize experiment=semantic_ids   # -> results/sid_clusters.png  (viz extra)
```

`L=3 × K=64` ⇒ 192 per-level SID tokens. Inspect `codebook_usage` in the JSON — if a level
collapses, raise `semantic_ids.rqvae.train_steps` or lower `codebook_size`.

## 3. Extend the tokenizer **[GPU]**

```bash
python -m rimworld_agent.knowledge.extend_tokenizer
#   adds mined tokens (mean init) + SID/action/vision tokens (codebook-projected init);
#   resizes embeddings; -> data/models/extended/
```

## 4. Knowledge pre-training **[GPU]**

```bash
python -m rimworld_agent.training.prepare_data experiment=knowledge_pretrain   # -> data/dataset/
python -m rimworld_agent.training.train_qlora  experiment=knowledge_pretrain   # ~14 GB VRAM, 2–4 h
```

QLoRA r=32, `modules_to_save=[embed_tokens, lm_head]` (so the new rows learn). FIM teaches
Defs/C#/wiki; SID tasks teach entity↔ID mappings. No screenshots yet.

## 5. Collect screenshots + train mmproj **[GAME]→[GPU]**

```bash
# Play manually for a few hours; capture a screenshot + paired state every ~30 s:
python -c "from rimworld_agent.game.rimapi_client import RimAPIClient; from rimworld_agent.vision.screenshot import collect_training_screenshots; collect_training_screenshots(RimAPIClient(), 'data/screenshots', 1000)"
python -m rimworld_agent.vision.train_mmproj experiment=mmproj_train   # ~8 GB VRAM, 1–2 h -> data/models/mmproj.pt
```

500–1000 annotated screenshots is enough to start. Capture at 1× for clean frames.

## 6. Self-play **[GAME]+[GPU]**

```bash
python -m rimworld_agent.training.self_play experiment=self_play
```

Iteration 0 bootstraps with random play; each later iteration plays 20 episodes at 3×
speed, keeps the top 50 % by reward, and fine-tunes with the discrete-diffusion action
objective. Watch `results/self_play_metrics.json` for the reward-per-iteration curve.

## 7. Export for inference **[GPU]**

```bash
python -m rimworld_agent.training.merge_export adapter_dir=data/models/qlora
scripts/export_ollama.sh                       # GGUF + Ollama Modelfile
```

## 8. Evaluate

```bash
python -m rimworld_agent.eval.eval_compression          # [CPU] token reduction (target 25%+)
python -m rimworld_agent.eval.eval_gameplay             # aggregate recorded episodes
python -m rimworld_agent.eval.eval_planning             # [CPU] action validity + SID usage
python -m rimworld_agent.eval.eval_vision               # [GPU+GAME] screenshot understanding
```

## Troubleshooting

- **`import src` resolves to the wrong package** — run from this repo's directory (no local
  `src` here) with `vocab-extend-qlora` installed, or set `$VEQ_PATH`.
- **Unsloth patch order** — vocab load → `resize_token_embeddings` → `get_peft_model`
  (gotcha #9); `train_qlora.train` follows this.
- **Codebook collapse** — check `codebook_usage`; the RQ-VAE re-inits dead codes
  automatically but very small entity sets may still collapse.
- **VRAM** — drop `training.qlora.batch_size`/raise `grad_accum`, keep
  `gradient_checkpointing: true`.
