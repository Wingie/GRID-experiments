# Architecture

## System overview

```
                         ┌─────────────────────────────────────────────┐
                         │            RimWorld (Unity) + RIMAPI mod      │
                         │   /state  /screenshot  /build  /research ...  │
                         └───────────────┬───────────────────▲──────────┘
                          observation     │                   │ actions
                          (4 frames +     │                   │ (≤5/turn)
                           game state)    ▼                   │
   ┌──────────────┐   ┌───────────────────────────┐   ┌──────┴───────────┐
   │ data sources │   │      Sovereign SLM (1.5B)   │   │   action space   │
   │  XML Defs    │   │  ┌──────────────────────┐   │   │  + token grammar │
   │  C# source   │──▶│  │ extended tokenizer    │   │──▶│  ACT/PARAM/REASON│
   │  wiki        │   │  │  + SID tokens         │   │   └──────────────────┘
   └──────┬───────┘   │  ├──────────────────────┤   │
          │           │  │ mmproj: SigLIP→MLP    │◀──┼── 4-frame visual tokens
   ┌──────▼───────┐   │  ├──────────────────────┤   │
   │ semantic IDs │   │  │ T5/decoder + diffusion │   │
   │  RQ-VAE L=3  │──▶│  │   action head          │   │
   │  K=64        │   │  └──────────────────────┘   │
   └──────────────┘   └───────────────┬─────────────┘
                                       │ reward (quests, wealth, mood, deaths)
                                       ▼
                              ┌──────────────────┐
                              │  self-play loop  │  play → rank → keep top 50% →
                              │  (GRPO + denoise)│  discrete-diffusion fine-tune
                              └──────────────────┘
```

## Data flow

1. **Knowledge → embeddings.** `extract_defs` resolves the `ParentName` inheritance chain
   so each `RimWorldEntity` has its full field set; `extract_csharp` parses the decompiled
   assembly; `scrape_wiki` pulls strategy pages. `embed_entities` combines the def / label /
   C# / wiki views into one dense vector per entity (CodeT5+-110M encoder, reused from
   vocab-extend-qlora with an offline hashing fallback).

2. **Embeddings → semantic IDs.** `assign_ids.run_pipeline` trains the RQ-VAE
   (`L=3` levels × `K=64` codes ⇒ `262 144` possible IDs, `192` per-level SID tokens) and
   maps every entity to an `<SID_L1_*><SID_L2_*><SID_L3_*>` sequence. IDs are per-entity and
   independent (TIGER-style); the entity graph supplies a category label only for
   *measuring* hierarchical consistency, not for enforcing prefixes.

3. **Tokenizer extension.** `extend_tokenizer` (delegating to `vocab-extend-qlora.extend`)
   adds the top-N mined game tokens (`add_tokens`, mean-initialised) and the SID +
   action + vision special tokens (`add_special_tokens`); SID rows are initialised from the
   RQ-VAE codebook geometry projected into embedding space. `modules_to_save` keeps
   `embed_tokens`/`lm_head` trainable.

4. **Vision.** Frozen SigLIP-SO400M features → trained 2-layer MLP → ~256 LLM-space tokens
   per frame; 4 frames concatenated with `<FRAME_SEP>` give ~20 s of visual history. The
   alignment loss pulls a screenshot's pooled visual tokens toward the SID embeddings of the
   entities present on screen.

5. **Game loop.** `play_episode` observes (4 frames + structured state + visible SIDs),
   the policy emits reasoning + ≤5 actions, `RimAPIClient.execute` routes each action to a
   key event or REST endpoint, the transition is scored by `compute_reward`, and the step is
   recorded.

6. **Self-play.** Bootstrap with random play, then iterate: play K episodes, keep the top
   50 % by total reward, and fine-tune on them with a discrete-diffusion objective that masks
   30–70 % of the action tokens and learns to denoise them.

## Module map

```
rimworld_agent/
  utils.py                 to_veq_cfg bridge, veq locator, JSON IO, logging
  knowledge/               extract_defs, extract_csharp, scrape_wiki, mine_tokens, extend_tokenizer
  semantic_ids/            build_entity_graph, embed_entities, rqvae(reuse), assign_ids, visualize
  vision/                  screenshot, ui_elements, state_encoder, dataset, train_mmproj
  game/                    action_space, keymap, reward, episode_recorder, rimapi_client, game_loop
  training/                prepare_data, train_qlora, self_play, merge_export
  eval/                    eval_gameplay, eval_compression, eval_vision, eval_planning
```

## Why these choices

- **Embedding-layer knowledge, not in-context docs.** Vocabulary + SIDs + vision are baked
  into the weights, so inference spends its context budget on the current situation, not on
  re-reading the manual every turn.
- **Reused quantiser.** The RQ-VAE is identical to the code-model work, so the "TIGER for a
  codebase" result transfers directly; only the entity *source* (game Defs vs. code) changes.
- **The game is the reward model.** No separate critic to train or mis-calibrate — quest
  completion, wealth, mood, and deaths are read straight from RIMAPI.
