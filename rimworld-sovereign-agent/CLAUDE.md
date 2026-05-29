# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`rimworld-sovereign-agent` trains a ~1.5B "sovereign" SLM to play RimWorld by learning the
game at the **embedding layer**: extended tokenizer + **dual** hierarchical **semantic IDs**
(two RQ-VAEs) + **mmproj** vision + **discrete-diffusion** action planning + **self-play**.
RimWorld is the testbed for a general recipe (the same pipeline applies to any
codebase/app with a reward signal). Targets a single RTX 3090.

**Dual semantic IDs (the key idea):** READ SIDs (`<RSID_L*_*>`, taxonomy) describe what an
entity IS — used in `<REASONING>` (perception); WRITE SIDs (`<WSID_L*_*>`, workflow) describe
what it is USED WITH — used in `<ACTIONS>` (planning). Motivated by Spotify (NeurIPS 2025):
search-tuned and rec-tuned semantic IDs degrade each other in one codebook, so we keep two.

It currently lives as a **subdirectory of GRID-experiments**, a sibling of
`vocab-extend-qlora`. It is designed to be split out later via
`git subtree split --prefix rimworld-sovereign-agent`.

## Reuse: `vocab-extend-qlora` is a dependency

The tokenizer-extension / RQ-VAE / QLoRA / GGUF-export machinery is **imported**, not
copied, from the sibling `vocab-extend-qlora` (its package is importable as `src`).

- The new package is named **`rimworld_agent`** (NOT `src`) precisely so `import src.*`
  resolves unambiguously to vocab-extend-qlora.
- `rimworld_agent.utils.ensure_veq_importable()` locates the sibling (or `$VEQ_PATH`, or an
  installed `src`) and puts it on `sys.path`. Run stages from THIS repo's directory (no
  local `src` here) so the sibling's `src` wins.
- **Config bridge:** we use Hydra/OmegaConf, but veq functions read a plain `dict` via a
  dict-only `cfg_get`. Always pass `rimworld_agent.utils.to_veq_cfg(cfg)` (=
  `OmegaConf.to_container(cfg, resolve=True)`) into any veq function. Our top-level config
  namespaces (`mining`, `extend`, `semantic_ids`, `model`, `paths`, `seed`, `export`)
  deliberately match veq's so this is a structural conversion, not a remap.

## Setup & running

```bash
pip install -e ../vocab-extend-qlora     # provides `import src.*`
pip install -e .                          # this package
```

Every stage is a Hydra entry point run as a module:

```bash
python -m rimworld_agent.knowledge.extract_defs
python -m rimworld_agent.semantic_ids.assign_ids experiment=semantic_ids
python -m rimworld_agent.training.train_qlora experiment=knowledge_pretrain
python -m rimworld_agent.training.self_play experiment=self_play
```

Config: `configs/base.yaml` (self-contained) + `configs/experiment/*.yaml`
(`# @package _global_` overlays) + `configs/{paths,hydra}/` groups. CLI overrides are
dotted (`mining.top_n=512`). New runs are new experiment YAMLs, not Python edits.

## What runs where (CRITICAL)

This repo is built so the **offline subset** is correct and tested without a GPU, the game,
or network. Do not "fix" the offline fallbacks by making them hard failures.

- **CPU / offline-verifiable:** `knowledge/extract_defs`, `semantic_ids/build_entity_graph`,
  `semantic_ids/sid_vocab`, `semantic_ids/collect_cooccurrence`, `semantic_ids/rqvae_write`
  (PPMI+SVD), `semantic_ids/assign_ids.run_pipeline` (READ + WRITE; uses veq's HashingEncoder
  fallback when CodeT5+ can't be downloaded), `game/action_space`, `game/keymap`,
  `game/reward`, `game/episode_recorder`, `training/prepare_data` (formatting),
  `eval/eval_compression`, `eval/eval_planning`, `eval/eval_ablation`, and **all of `tests/`**.
- **GPU-only (written, unverified):** `knowledge/extend_tokenizer`, `training/train_qlora`,
  `training/merge_export`, `vision/train_mmproj`.
- **Game-only (live RIMAPI):** `game/rimapi_client`, `game/game_loop`, `vision/screenshot`,
  `training/self_play`, `eval/eval_gameplay`, `eval/eval_vision`.
- **Network-only:** `knowledge/scrape_wiki`.

Metrics produced with the offline HashingEncoder / FallbackTokenizer are machinery
sanity-checks, **not** research numbers.

## Module map

`rimworld_agent/{knowledge,semantic_ids,vision,game,training,eval,games,benchmarks}/` +
`utils.py`. See `docs/ARCHITECTURE.md` for the data flow, `docs/TRAINING_GUIDE.md` for the
run order, and `docs/GAMES.md` for the multi-game framework.

## Multi-game framework

`rimworld_agent/games/base.py` defines `GameBackend` (Protocol) + a `register`/`get_backend`
registry. Bundled backends: `rimworld` (wraps existing code), `eve` (EVE Online ESI + SDE),
`videogamebench` (Pokémon Red + Zelda: The Minish Cap headline cap). The same dual RQ-VAE +
vision + self-play layers work across backends as soon as a new game conforms to the
protocol. `rimworld_agent.benchmarks.videogamebench.run_benchmark` is the policy-agnostic
cross-game runner that drives one or more backends and writes a unified per-game table.

Keep backend imports lazy on the game's heavy dep (`videogamebench` package, EVE OAuth) so
the registry's `_autoload` succeeds even when one backend is unavailable.

## Conventions & gotchas

- **Dual semantic IDs.** READ (`<RSID_*>`) and WRITE (`<WSID_*>`) are parallel token families
  from two RQ-VAEs that **share the architecture** (`rqvae.py` re-exports veq's
  `ResidualVQVAE`) but train on different embeddings: READ on structural views
  (`rqvae_read.build_read_embeddings`, optionally category-sharpened), WRITE on co-occurrence
  PPMI+SVD (`rqvae_write.build_write_embeddings`). `L=3 × K=64 ⇒ 192` per-level tokens *per
  family*. `sid_vocab.SIDVocab(prefix=...)` owns the token grammar (NOT veq's `SemanticIDVocab`,
  which is `<SID_>`-only).
- **Training order is a hard dependency (gotcha #11):** READ (Defs only) → RSID-only QLoRA →
  bootstrap gameplay (`experiment=bootstrap`, `self_play.train=false`) → mine co-occurrence →
  WRITE RQ-VAE → dual retrain (`experiment=dual_pretrain`) → self-play. `assign_ids.run_pipeline`
  assigns RSIDs to all entities and WSIDs only to entities acted-on in recorded episodes
  (others get `wsid_*: None`).
- **Watch for SID leakage (gotcha #13):** `eval_planning` reports `rsid_usage_rate` (reasoning),
  `wsid_usage_rate` (actions), and `sid_leakage_rate` (WSID in reasoning / RSID in actions).
- **SID tokens** are `add_special_tokens` (never split); RSID/WSID rows init from their
  respective RQ-VAE codebook geometry (`extend.init_method=codebook`, `extend_tokenizer.extend_dual`).
  Mined freq tokens use mean-of-subtoken init; action/vision structural tokens use the mean row.
- **Entity graph** (`build_entity_graph`) gives a category label for *measuring* READ
  hierarchical consistency (and READ contrastive sharpening), not for enforcing SID prefixes.
- **Unsloth order** (gotcha #9): 4-bit load → `resize_token_embeddings` → `get_peft_model`,
  with `modules_to_save=[embed_tokens, lm_head]`. `training/train_qlora.train` follows it.
- **`num_hierarchies` off-by-one** in the GRID parent does NOT apply here — we do not run the
  GRID SID-inference dedup step; SIDs come straight from `rqvae.encode_to_ids`.
- **Action grammar:** `<ACTION_START><ACT:..><PARAM:k=v>..<REASON>..</REASON><ACTION_END>`,
  ≤5 per turn. `game/action_space.py` is the single source of truth (vocabulary, token
  format, validity). Vision tokens: `<VISION_START>/<FRAME_SEP>/<VISION_END>`.
- **Reward** (`game/reward.py`) reads RIMAPI state deltas; the building count comes from
  `resources["buildings"]`. Weights live in `configs/base.yaml::eval.reward_weights` and
  `game/reward.py::RewardWeights`.
- **Artifacts are gitignored** (`data/`, `results/*.json`, `*.pt`, `*.gguf`); only source +
  `.gitkeep` placeholders are tracked.

## Tests

`pytest` (CPU). torch-dependent tests (`test_semantic_ids.py` RQ-VAE + SID-vocab) use
`pytest.importorskip("torch")` and `ensure_veq_importable()` so they skip cleanly when torch
or the sibling checkout is absent.
