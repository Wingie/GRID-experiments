# Sovereign Game Agents: Teaching Small Language Models to Play RimWorld via Tokenizer Extension, Dual Semantic IDs, and Self-Play

## Abstract

We present a framework for training a ~1.5B-parameter language model to play
[RimWorld](https://rimworldgame.com) by learning the game's codebase **at the embedding
layer** instead of reading documentation at inference time. The tokenizer is extended with
game vocabulary, and entity hierarchy is encoded as **dual Semantic IDs** via separate
RQ-VAEs: **READ SIDs (RSIDs)** capture entity taxonomy for perception/reasoning, while
**WRITE SIDs (WSIDs)** capture workflow co-occurrence for action planning — motivated by
Spotify's finding (NeurIPS 2025) that search-tuned and recommendation-tuned semantic IDs
degrade each other when sharing a codebook. Visual understanding is trained through an
**mmproj** projection aligned to RSID embeddings. The agent observes the last 4 frames,
reasons with RSIDs about colony state, and plans actions with WSIDs. Training uses a
**self-play** loop with the game's own quest/reward signal. All experiments target a single
RTX 3090.

## The key idea

This is **not primarily about playing RimWorld well**. RimWorld is the *testbed* for a
general recipe that applies to any codebase or app with a reward signal:

1. **Extend the tokenizer** with domain vocabulary — proven (AdaptiVocab / VEGAD).
2. **Dual semantic IDs for entities** via two RQ-VAEs — TIGER applied to a codebase, split
   into READ (taxonomy) and WRITE (workflow) because *reading* and *writing* optimise for
   different similarity metrics, just as search and recommendation do (Spotify, NeurIPS 2025).
3. **Train vision on your specific UI** — mmproj for a domain app, aligned to RSIDs.
4. **Plan actions with discrete diffusion** — parallel action generation + denoising.
5. **Improve by self-play** — GRPO-style filtering where the game *is* the reward model.

The **dual SID** insight is the key novelty. The same pipeline applies to testing **your**
web app, navigating **your** IDE, or automating **your** internal tools. RimWorld just ships
a built-in reward signal (quests, wealth, survival).

## READ vs WRITE semantic IDs

| Entity | READ SID (what it IS) | WRITE SID (what it's USED WITH) |
|--------|----------------------|---------------------------------|
| SolarGenerator | `Buildings→Power→Solar` | `PowerSetup→Generation→Solar` |
| Battery | `Buildings→Power→Storage` | `PowerSetup→Storage→Battery` |
| ResearchBench | `Buildings→Production→Research` | `PowerSetup→Prerequisites→Bench` |

RSIDs appear in `<REASONING>` (perception); WSIDs annotate `<ACTIONS>` (planning). The model
learns the same entity has two addresses — "I see a bed" vs "I should build a bed". Token
budget: `192` RSID + `192` WSID per-level tokens + structural + ~256 mined vocab ≈ 646 new
tokens (<0.5% of a 151k vocab). **Training order matters** (gotcha #11): READ trains from
Defs alone; WRITE needs bootstrap gameplay to mine co-occurrence first.

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
| 2. Semantic IDs | `rimworld_agent/semantic_ids/` | Build the entity graph; train the READ RQ-VAE (`rqvae_read`, taxonomy) and — after gameplay — the WRITE RQ-VAE (`rqvae_write`) from mined co-occurrence (`collect_cooccurrence`); assign dual `<RSID_L*_*>` + `<WSID_L*_*>` tokens (`assign_ids`); compare READ vs WRITE clusters (`visualize`). |
| 3. Vision | `rimworld_agent/vision/` | Capture screenshots, train an mmproj projection (SigLIP→MLP→LLM space), encode the 4-frame history. |
| 4. Game | `rimworld_agent/game/` | RIMAPI REST client, the finite action space + token grammar, keymap, reward function, episode recorder, game loop. |
| 5. Training | `rimworld_agent/training/` | Build the multi-source dataset, QLoRA knowledge pre-training, self-play loop, merge + GGUF export. |
| 6. Eval | `rimworld_agent/eval/` | Gameplay performance, token compression, vision accuracy, planning quality. |

Every stage is a Hydra entry point:

```bash
python -m rimworld_agent.knowledge.extract_defs                      # parse Defs -> results/entities.json
python -m rimworld_agent.semantic_ids.assign_ids experiment=semantic_ids   # READ SIDs (taxonomy)
python -m rimworld_agent.training.train_qlora    experiment=knowledge_pretrain  # RSIDs only
python -m rimworld_agent.training.self_play      experiment=bootstrap     # record bootstrap episodes
python -m rimworld_agent.semantic_ids.collect_cooccurrence experiment=write_rqvae
python -m rimworld_agent.semantic_ids.assign_ids experiment=write_rqvae    # adds WRITE SIDs (workflow)
python -m rimworld_agent.training.train_qlora    experiment=dual_pretrain  # retrain with RSID + WSID
python -m rimworld_agent.training.self_play      experiment=self_play
python -m rimworld_agent.eval.eval_planning                          # RSID/WSID usage + leakage
python -m rimworld_agent.eval.eval_ablation                          # dual vs single (spec §9f)
```

> **Training order is a hard dependency** (gotcha #11): READ SIDs → RSID-only pre-training →
> bootstrap gameplay → WRITE SIDs from co-occurrence → dual retrain → self-play.

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
- `docs/GAMES.md` — the multi-game framework (RimWorld + EVE Online + VideoGameBench).

## Multi-game framework

The agent is not RimWorld-specific. A small `GameBackend` protocol
(`rimworld_agent/games/base.py`) plugs three backends into the same training + eval pipeline:

- **`rimworld`** — RIMAPI client + the existing action space / reward.
- **`eve`** — EVE Online ESI REST client + SDE knowledge parser (industry / market / skill /
  contracts / navigation; needs `$EVE_ACCESS_TOKEN` for character-scoped actions).
- **`videogamebench`** — emulator-backed gym envs; the default benchmark is **capped to
  Pokémon Red and *The Legend of Zelda: The Minish Cap***. Other VGB games work via
  `get_backend("videogamebench", game=...)`.

Run the cross-game benchmark with:

```bash
python -m rimworld_agent.benchmarks.videogamebench experiment=benchmark
#   -> results/videogamebench.json (Pokémon Red + Zelda: The Minish Cap by default)
```

See `docs/GAMES.md` for the full backend contract and how to add a new game.

## Live presentation (reveal.js + Markdown + React widgets)

A demo deck under `presentation/`: reveal.js loads Markdown slides, and a React grid on the
"Live demo" slide bridges the audience to the running agent's commentary channel.
Questions submitted in the deck land in the same `QueueQuestionSource` the agent pulls from;
every `say` action is POSTed back into the deck via a WebSocket broadcast.

```bash
pip install -e .[presentation]
uvicorn presentation.server:app --reload --port 8000
# open http://127.0.0.1:8000 and navigate to the "Live demo" slide
```

Wire your agent into the deck:

```python
from presentation.server import SHARED_SOURCE, bridge_to_server
from rimworld_agent.games.commentary import CommentaryWrapper
from rimworld_agent.games.base import get_backend

backend = CommentaryWrapper(
    get_backend("rimworld"),
    source=SHARED_SOURCE,
    on_say=bridge_to_server("http://127.0.0.1:8000"),
)
```

See `presentation/README.md` for the full endpoint list and slide source layout.

## Tests

```bash
pytest            # CPU-only; torch-dependent tests skip cleanly when torch is absent
```

## License

Apache-2.0.
