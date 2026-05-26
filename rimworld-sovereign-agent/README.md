# Sovereign Game Agents: Teaching Small Language Models to Play RimWorld via Tokenizer Extension, Semantic IDs, and Self-Play

## Abstract

We present a framework for training a ~1.5B-parameter language model to play
[RimWorld](https://rimworldgame.com) by learning the game's codebase **at the embedding
layer** instead of reading documentation at inference time. The model's tokenizer is
extended with game-specific vocabulary, the entity hierarchy is encoded as hierarchical
**Semantic IDs** via an RQ-VAE, and visual understanding is trained through an **mmproj**
projection on game screenshots. The agent observes the last 4 frames, reasons about colony
priorities, and outputs up to 5 actions with justifications. Training uses a **self-play**
loop with the game's own quest/reward signal. All experiments target a single RTX 3090.

## The key idea

This is **not primarily about playing RimWorld well**. RimWorld is the *testbed* for a
general recipe that applies to any codebase or app with a reward signal:

1. **Extend the tokenizer** with domain vocabulary — proven (AdaptiVocab / VEGAD).
2. **Assign semantic IDs to code/game entities** via RQ-VAE — TIGER applied to a codebase.
3. **Train vision on your specific UI** — mmproj for a domain app.
4. **Plan actions with discrete diffusion** — parallel action generation + denoising.
5. **Improve by self-play** — GRPO-style filtering where the game *is* the reward model.

The same pipeline applies to testing **your** web app, navigating **your** IDE, or
automating **your** internal tools. RimWorld just ships a built-in reward signal (quests,
wealth, survival).

## Relationship to `vocab-extend-qlora`

The tokenizer-extension, RQ-VAE, QLoRA, and GGUF-export machinery is **reused as a
dependency** from the sibling [`vocab-extend-qlora`](../vocab-extend-qlora) project (its
package is importable as `src`). This repo adds the game-specific layers: knowledge
extraction (XML Defs / decompiled C# / wiki), an entity graph, vision/mmproj, the RIMAPI
game-interaction layer, episode recording + reward, self-play training, and gameplay
evaluation.

```bash
pip install -e ../vocab-extend-qlora        # provides `import src.*`
pip install -e .                            # this package: `rimworld_agent`
# offline subset only needs: torch transformers hydra-core omegaconf lxml requests numpy pytest
```

> The new package is named `rimworld_agent` (not `src`) precisely so that `import src.*`
> unambiguously resolves to `vocab-extend-qlora`.

## Pipeline (six phases)

| Phase | Module | What it does |
|------|--------|--------------|
| 1. Knowledge | `rimworld_agent/knowledge/` | Parse XML Defs (`extract_defs`), decompiled C# (`extract_csharp`), scrape the wiki (`scrape_wiki`); mine inefficient tokens (`mine_tokens`); extend the tokenizer (`extend_tokenizer`). |
| 2. Semantic IDs | `rimworld_agent/semantic_ids/` | Build the entity graph, embed entities, train an RQ-VAE (`rqvae`, reused), assign `<SID_L*_*>` tokens (`assign_ids`), visualise clusters. |
| 3. Vision | `rimworld_agent/vision/` | Capture screenshots, train an mmproj projection (SigLIP→MLP→LLM space), encode the 4-frame history. |
| 4. Game | `rimworld_agent/game/` | RIMAPI REST client, the finite action space + token grammar, keymap, reward function, episode recorder, game loop. |
| 5. Training | `rimworld_agent/training/` | Build the multi-source dataset, QLoRA knowledge pre-training, self-play loop, merge + GGUF export. |
| 6. Eval | `rimworld_agent/eval/` | Gameplay performance, token compression, vision accuracy, planning quality. |

Every stage is a Hydra entry point:

```bash
python -m rimworld_agent.knowledge.extract_defs                 # parse Defs -> results/entities.json
python -m rimworld_agent.semantic_ids.assign_ids experiment=semantic_ids
python -m rimworld_agent.knowledge.extend_tokenizer             # extend + init embeddings
python -m rimworld_agent.training.train_qlora experiment=knowledge_pretrain
python -m rimworld_agent.vision.train_mmproj experiment=mmproj_train
python -m rimworld_agent.training.self_play experiment=self_play
python -m rimworld_agent.eval.eval_compression
```

Configuration is Hydra (`configs/base.yaml` + `configs/experiment/*.yaml`). The top-level
`mining`, `extend`, `semantic_ids`, `model`, `paths` namespaces mirror `vocab-extend-qlora`
so configs pass straight through `rimworld_agent.utils.to_veq_cfg`.

## What runs where

| Runs on CPU / offline | Needs a GPU / running game |
|-----------------------|----------------------------|
| Def/C#/wiki parsing, entity graph, token mining, SID assignment + RQ-VAE, action space, reward, episode schema, data prep, compression + planning eval, **all unit tests** | mmproj training, QLoRA training, GGUF export, live RIMAPI, screenshot capture, the game loop, self-play, gameplay/vision eval |

## Action format

```
<ACTION_START>
  <ACT:order_build>
  <PARAM:def_name=SolarGenerator>   # uses SID <SID:3-3-2>
  <PARAM:x=42>
  <PARAM:y=18>
  <REASON>Need power for the research bench</REASON>
<ACTION_END>
```

The model emits up to 5 of these per turn after a `<REASONING>...</REASONING>` block.

## Setup & docs

- `docs/RIMWORLD_SETUP.md` — install the RIMAPI mod + screenshot capture, decompile the assembly.
- `docs/TRAINING_GUIDE.md` — step-by-step from zero to a playing agent.
- `docs/ARCHITECTURE.md` — full system diagram and data flow.

## Tests

```bash
pytest            # CPU-only; torch-dependent tests skip cleanly when torch is absent
```

## License

Apache-2.0.
