# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`rimworld-sovereign-agent` trains a ~1.5B "sovereign" SLM to play RimWorld by learning the
game at the **embedding layer**: extended tokenizer + hierarchical **semantic IDs**
(RQ-VAE) + **mmproj** vision + **discrete-diffusion** action planning + **self-play**.
RimWorld is the testbed for a general recipe (the same pipeline applies to any
codebase/app with a reward signal). Targets a single RTX 3090.

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
  `semantic_ids/assign_ids.run_pipeline` (uses veq's HashingEncoder fallback when CodeT5+
  can't be downloaded), `game/action_space`, `game/keymap`, `game/reward`,
  `game/episode_recorder`, `training/prepare_data` (formatting), `eval/eval_compression`,
  `eval/eval_planning`, and **all of `tests/`**.
- **GPU-only (written, unverified):** `knowledge/extend_tokenizer`, `training/train_qlora`,
  `training/merge_export`, `vision/train_mmproj`.
- **Game-only (live RIMAPI):** `game/rimapi_client`, `game/game_loop`, `vision/screenshot`,
  `training/self_play`, `eval/eval_gameplay`, `eval/eval_vision`.
- **Network-only:** `knowledge/scrape_wiki`.

Metrics produced with the offline HashingEncoder / FallbackTokenizer are machinery
sanity-checks, **not** research numbers.

## Module map

`rimworld_agent/{knowledge,semantic_ids,vision,game,training,eval}/` + `utils.py`. See
`docs/ARCHITECTURE.md` for the data flow and `docs/TRAINING_GUIDE.md` for the run order.

## Conventions & gotchas

- **Semantic IDs are per-entity & independent** (TIGER-style). `L=3 × K=64 ⇒ 192` per-level
  SID tokens. The entity graph (`build_entity_graph`) gives a category label for *measuring*
  hierarchical consistency, not for enforcing SID prefixes.
- **SID tokens** are `add_special_tokens` (never split); init from the RQ-VAE codebook
  geometry (`extend.init_method=codebook`). Mined freq tokens use mean-of-subtoken init.
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
