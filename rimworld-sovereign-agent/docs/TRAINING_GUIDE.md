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

## 2. READ semantic IDs **[CPU]**

```bash
python -m rimworld_agent.semantic_ids.assign_ids experiment=semantic_ids
#   -> data/models/rqvae_read.pt, results/sid_assignments.json (RSIDs; WSIDs still null)
python -m rimworld_agent.semantic_ids.visualize experiment=semantic_ids   # READ vs WRITE clusters (viz extra)
```

`L=3 × K=64` ⇒ 192 per-level RSID tokens. WRITE SIDs are intentionally absent here — they need
gameplay (step 5b). Inspect `read_codebook_usage` in the JSON; if a level collapses, raise
`semantic_ids.read.rqvae.train_steps` or lower `codebook_size`.

## 3. Extend the tokenizer **[GPU]**

```bash
python -m rimworld_agent.knowledge.extend_tokenizer
#   adds mined tokens (mean init) + RSID/WSID tokens (codebook-projected) + action/vision
#   tokens (mean); resizes embeddings; -> data/models/extended/
```

> Before bootstrap gameplay the WRITE codebook is empty, so the first extension registers
> RSIDs only. After step 5b you re-extend (step 5c) with both families.

## 4. Knowledge pre-training (RSIDs only) **[GPU]**

```bash
python -m rimworld_agent.training.prepare_data experiment=knowledge_pretrain   # -> data/dataset/
python -m rimworld_agent.training.train_qlora  experiment=knowledge_pretrain   # ~14 GB VRAM, 2–4 h
```

QLoRA r=32, `modules_to_save=[embed_tokens, lm_head]`. FIM teaches Defs/C#/wiki; READ-SID
tasks teach entity↔RSID mappings. No screenshots, no WSIDs yet.

## 5. Collect screenshots + train mmproj **[GAME]→[GPU]**

```bash
# Play manually for a few hours; capture a screenshot + paired state every ~30 s:
python -c "from rimworld_agent.game.rimapi_client import RimAPIClient; from rimworld_agent.vision.screenshot import collect_training_screenshots; collect_training_screenshots(RimAPIClient(), 'data/screenshots', 1000)"
python -m rimworld_agent.vision.train_mmproj experiment=mmproj_train   # ~8 GB VRAM, 1–2 h -> data/models/mmproj.pt
```

500–1000 annotated screenshots is enough to start. Capture at 1× for clean frames.

## 5b. Bootstrap gameplay → WRITE semantic IDs **[GAME]→[CPU]** (gotcha #11)

The WRITE codebook needs co-occurrence data, which only exists once the agent has played.

```bash
# Record ~60 bootstrap episodes (random or the RSID-only model), NO training:
python -m rimworld_agent.training.self_play experiment=bootstrap
# Mine co-occurrence and train the WRITE RQ-VAE; re-assign dual RSID + WSID:
python -m rimworld_agent.semantic_ids.collect_cooccurrence experiment=write_rqvae
python -m rimworld_agent.semantic_ids.assign_ids           experiment=write_rqvae
#   -> data/models/rqvae_write.pt; sid_assignments.json now has wsid_* for acted-on entities
```

## 5c. Re-extend + dual retrain **[GPU]**

```bash
python -m rimworld_agent.knowledge.extend_tokenizer                          # now adds WSID rows too
python -m rimworld_agent.training.train_qlora experiment=dual_pretrain        # RSID + WSID mix
```

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
python -m rimworld_agent.eval.eval_planning             # [CPU] action validity + RSID/WSID usage + leakage
python -m rimworld_agent.eval.eval_vision               # [GPU+GAME] screenshot understanding
python -m rimworld_agent.eval.eval_ablation             # [CPU] dual vs single RQ-VAE table (spec §9f)
```

**Key experiment (spec §9f):** run the pipeline under `sid_mode` ∈ {none, single, dual,
multitask}, writing each config's metrics to `results/ablation/<mode>/`, then `eval_ablation`
builds the comparison. If `single` matches `dual`, you save ~195 tokens; if `dual` wins on
quest completion + low `sid_leakage_rate`, the READ/WRITE split is doing real work.

## Troubleshooting

- **`import src` resolves to the wrong package** — run from this repo's directory (no local
  `src` here) with `vocab-extend-qlora` installed, or set `$VEQ_PATH`.
- **Unsloth patch order** — vocab load → `resize_token_embeddings` → `get_peft_model`
  (gotcha #9); `train_qlora.train` follows this.
- **Codebook collapse** — check `codebook_usage`; the RQ-VAE re-inits dead codes
  automatically but very small entity sets may still collapse.
- **VRAM** — drop `training.qlora.batch_size`/raise `grad_accum`, keep
  `gradient_checkpointing: true`.
