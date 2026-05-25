# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`vocab-extend-qlora` is a research repo testing three combined ideas for
codebase-specific code SLMs on a single RTX 3090: **tokenizer-vocabulary extension**
(AdaptiVocab/VEGAD), **QLoRA fine-tuning** (Unsloth), and a **novel hierarchical
semantic-ID layer** that applies the TIGER/RQ-VAE recsys paradigm to code entities
(class/function/method → an `L`-tuple of discrete codes).

It currently lives as a **subdirectory of the GRID-experiments repo** (the root
`../CLAUDE.md` documents GRID, not this project). It is designed to be split out later
via `git subtree split --prefix vocab-extend-qlora`.

## Setup & running

```bash
pip install -r requirements.txt          # full stack (needs CUDA for training)
# CPU-only subset (mining / entity extraction / RQ-VAE / compression / tests run offline):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets pyyaml numpy sacrebleu Levenshtein scikit-learn pandas tqdm pytest
```

Every stage is a module run as `python -m src.<stage> <config> [dotted.key=value ...]`:

```bash
python -m src.mine_tokens           configs/experiments/semid_qlora.yaml
python -m src.semantic_ids.assign_ids configs/experiments/semid_qlora.yaml
python -m src.prepare_data          configs/experiments/semid_qlora.yaml
python -m src.train_qlora           configs/experiments/semid_qlora.yaml      # GPU only
python -m src.eval_compression      configs/experiments/semid_qlora.yaml
```

`scripts/run_all.sh <config> [git-url] [depth]` runs the whole pipeline.
`scripts/clone_target.sh <git-url> [depth]` clones the target codebase (`depth=full`
for git history). Tests: `pytest` (CPU-only; see below).

## Config system (read before editing configs)

This project does **not** use Hydra. `src/utils.py::load_config` implements a small
inheritance + override loader:
- `configs/base.yaml` holds every default (paths, mining, semantic_ids, extend, data,
  qlora, eval) plus the experiment toggles `extend_vocab` / `use_semantic_ids` /
  `do_finetune`.
- `configs/models/*.yaml` and `configs/experiments/*.yaml` start with
  `inherit: [base.yaml, models/<m>.yaml]`; parents are deep-merged, then the file's keys
  override. The 6 experiment configs are the ablation conditions (baseline → semid_qlora).
- CLI overrides are dotted `key=value` (e.g. `mining.top_n=256`), coerced via YAML.
- Read config values with `cfg_get(cfg, "a.b.c", default)` — never assume keys exist.

New runs are new experiment YAMLs, not Python edits.

## Pipeline & data flow

`src/` top-level scripts are the stages; `src/semantic_ids/` is the novel core.

1. **mine_tokens.py** → `results/candidates.json` (freq identifiers `freq×subtokens`,
   n-grams, optional gradient selection). `analyze_tokenizer.py` is a static compression
   estimate.
2. **semantic_ids/**: `extract_entities.py` (tree-sitter, stdlib-`ast` fallback for
   Python) → `embed_entities.py` (code encoder; optional `cochange.py` smoothing) →
   `rqvae.py` (RQ-VAE: encoder/decoder + EMA codebooks + dead-code reinit) →
   `assign_ids.py` (codes → `<SID_L{l}_{c}>` special tokens) → `inject_ids.py`
   (training formats + task objectives). `assign_ids.run_pipeline()` chains
   extract→embed→train→assign and is reused by extend/prepare/eval.
3. **extend_tokenizer.py** adds freq tokens (`add_tokens`) + SID tokens
   (`add_special_tokens`), resizes embeddings, inits new rows.
4. **prepare_data.py** → HF dataset (chunks + FIM + SID interleaving).
5. **train_qlora.py** (Unsloth), **merge_and_export.py** (GGUF), **eval_*.py**.

## Critical conventions & gotchas

- **Offline fallbacks are intentional.** When models can't be downloaded / no GPU:
  `utils.load_hf_tokenizer` → `FallbackTokenizer`, `embed_entities.get_encoder` →
  `HashingEncoder`, entity extraction → stdlib `ast`. This is what lets the CPU pipeline
  + tests run with no network. Metrics from fallback encoders are machinery
  sanity-checks, **not** research results. Don't "fix" the fallbacks by making them hard
  failures.
- **CPU-verifiable** here: mining, analysis, entity extraction, RQ-VAE, SID assignment,
  data prep, compression eval, co-change, all tests. **GPU-only (written, unverified):**
  `train_qlora.py`, `eval_completion.py`, `eval_latency.py`, the model-dependent SID
  metrics in `eval_semantic_ids.py`, `merge_and_export.py`.
- **`modules_to_save=["embed_tokens","lm_head"]`** is non-negotiable for LoRA + new
  tokens. New-token embeddings must be mean-initialised (freq) or codebook-projected
  (SID), never left random. Unsloth patches must be applied **before**
  `resize_token_embeddings`, which is **before** `get_peft_model` — `train_qlora.py`
  follows this order.
- **Semantic IDs are per-entity and independent** (TIGER-style); a method's codes are
  NOT forced to share its class's prefix. Hierarchical consistency is *measured* by
  `eval_semantic_ids.py`, not enforced. SID tokens are `add_special_tokens` (never
  split). For `L=3, K=64` that's `3*64=192` per-level tokens + structural + task tokens.
- **Co-change** (`semantic_ids.cochange.enabled`) needs a **full** clone; on a shallow
  clone (<2 commits) it logs a warning and is a no-op. It is auxiliary signal — commit
  messages / PR text are deliberately not mined into the code vocabulary.
- **Artifacts are gitignored**: `data/`, `results/*.json`, `*.gguf`. Only source +
  `results/.gitkeep` are tracked.
- **CI doesn't run while nested**: GitHub only reads root `.github/workflows`. The
  `.github/workflows/ci.yml` and `.pre-commit-config.yaml` here activate after the
  subdir is extracted to its own repo.

## Tests

`pytest` runs CPU-only unit tests (`tests/`): entity extraction, SID vocab
formatting/parsing, RQ-VAE shapes + dead-code reinit, embedding-init math, co-change
parsing/graph/smoothing, config inheritance. Each test file bootstraps `sys.path` to the
repo root; torch-dependent tests `pytest.importorskip("torch")` so they skip cleanly
when torch is absent. Init-math/RQ-VAE primitives take plain tensors so they're testable
without a downloaded model.
