# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GRID (Generative Recommendation with Semantic IDs) is a research framework for
generative recommendation. It is **fully config-driven**: there is almost no
imperative glue code. Behavior is composed by Hydra from `configs/` and every
object (datamodule, model, optimizer, callbacks, trainer) is built via
`hydra.utils.instantiate` using `_target_` keys. To change what runs, you edit
or add a config — not the entry-point Python.

## Setup & running

```bash
pip install -r requirements.txt   # large, pip-compiled, pinned for CUDA cu124; no extras/setup.py
```

There is **no `pyproject.toml`, `setup.py`, `Makefile`, lint config, or test
suite** in this repo. Don't look for `pytest`/`make` targets — run the
pipelines directly as modules:

```bash
python -m src.train     experiment=<name> [overrides...]
python -m src.inference experiment=<name> [overrides...]
```

GPU is assumed throughout (`accelerator: gpu`, `devices: -1`, DDP strategy). The
default trainer expects multi-GPU; on a single GPU pass `trainer.devices=1`,
and to run without a GPU pass `trainer.accelerator=cpu trainer.devices=1`.

### Gotcha: `.project-root` is required but absent

`src/train.py` and `src/inference.py` call
`rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)`,
which walks up the tree looking for a `.project-root` marker and **raises if it
isn't found**. This repo does not contain that file. Create an empty one at the
repo root before running anything: `touch .project-root`. It only adds the root
to `PYTHONPATH` / sets `PROJECT_ROOT`; it does not need contents.

## The three-stage pipeline

The whole project is one workflow run as a sequence of `experiment=` configs.
Each stage's output file feeds the next stage as a CLI override:

1. **LLM embeddings** — `src.inference experiment=sem_embeds_inference_flat`
   runs a HuggingFace encoder (default `google/flan-t5-xl`) over item text and
   writes `.../merged_predictions_tensor.pt`.
2. **Semantic ID learning** — `src.train experiment=rkmeans_train_flat`
   (or `rvq_train_flat`, `rqvae_train_flat`) learns codebooks from those
   embeddings; `src.inference experiment=rkmeans_inference_flat ckpt_path=...`
   emits per-item semantic IDs as a `.pt`.
3. **Generative recommendation** — `src.train experiment=tiger_train_flat`
   then `src.inference experiment=tiger_inference_flat ckpt_path=...` train and
   run the TIGER T5 encoder-decoder over user histories expressed as SID tokens.

See `README.md` for the exact override values (`embedding_path`,
`semantic_id_path`, `embedding_dim`, `num_hierarchies`, `codebook_width`).

**`num_hierarchies` off-by-one:** the SID inference stage appends one extra
digit to de-duplicate IDs, so the generative stage uses
`num_hierarchies = (SID hierarchies) + 1` (e.g. 3 → 4). This is intentional, not
a bug.

## Config architecture (read this before editing configs)

- `configs/train.yaml` / `configs/inference.yaml` are the root configs. They set
  most groups (`model`, `data_loading`, `loss`, `optim`, `eval`, ...) to `null`
  by default.
- `configs/experiment/*.yaml` are `# @package _global_` files that supply the
  **entire** run definition (model, full data-loading graph, trainer, callbacks,
  optimizer, eval). Selecting `experiment=<name>` is how a run is fully
  specified. New runs should be new experiment files, not edits to root configs.
- **Custom Hydra resolvers** in `src/utils/custom_hydra_resolvers.py` are
  imported for their side-effecting `OmegaConf.register_new_resolver` calls via
  `from src.utils.custom_hydra_resolvers import *` in both entry points. Configs
  lean on them heavily — e.g. `${extract_fields_from_list_of_dicts:...}` and
  `${create_map_from_list_of_dicts:...}` derive feature-name maps from the
  `features` list, and `${math_eval:...}` does arithmetic. If a config
  interpolation looks like a function call, its definition is here.
- Semantic IDs / embeddings are injected into configs by calling `torch.load`
  inside the YAML (`_target_: torch.load` wrapping
  `src.utils.file_utils.open_local_or_remote`), keyed off the `semantic_id_path`
  / `embedding_path` override.

## Code layout

- `src/train.py`, `src/inference.py` — Hydra entry points. Both delegate to
  `pipeline_launcher` (`src/utils/launcher_utils.py`), a context manager that
  instantiates `PipelineModules` (datamodule, model, callbacks, loggers,
  trainer) and guarantees loggers are finalized even on failure. `train.py` adds
  fit → (optional) test using the best-checkpoint path; `inference.py` calls
  `trainer.predict`. `train.py` also wraps the run in `LocalJobLauncher` for
  restart support.
- `src/models/modules/` — the LightningModules:
  - `base_module.py` `BaseModule` — shared training/val/test hooks, metric
    logging, optimizer wiring. Note: setting a custom `training_loop_function`
    switches the module to **manual optimization** (`automatic_optimization = False`).
  - `huggingface/transformer_base_module.py`, `semantic_id/tiger_generation_model.py`
    — the TIGER generative recommender (T5 encoder + T5Stack decoder, weight
    tying, constrained/prefix generation over codebooks).
  - `clustering/` — `mini_batch_kmeans.py`, `base_clustering_module.py` (the
    quantization layers used by RQ).
- `src/modules/` — higher-level pipeline modules:
  - `semantic_embedding_inference_module.py` — stage 1 (LLM → embeddings).
  - `clustering/residual_quantization.py`, `clustering/vector_quantization.py`
    — stage 2 SID learners (RQ-KMeans / RVQ / RQ-VAE).
- `src/components/` — reusable pieces wired in by config: `eval_metrics.py`
  (`SIDRetrievalEvaluator`, NDCG, Recall), `loss_functions.py`,
  `distance_functions.py`, `clustering_initializers.py` (k-means++),
  `optimizer.py`, `scheduler.py`, `training_loop_functions.py`,
  `network_blocks/hf_language_model.py`.
- `src/data/loading/` — config-driven data stack. `datamodules/sequence_datamodule.py`
  exposes `SequenceDataModule` (user-history sequences) and `ItemDataModule`
  (per-item text/embeddings). Data is read from **TFRecord** via
  `components/iterators.py::TFRecordIterator`, streamed through
  `UnboundedSequenceIterable`, and shaped by an ordered list of composable
  `components/pre_processing.py` functions plus `collate_functions.py`. Dataset
  /dataloader shapes are dataclasses in `components/interfaces.py`.
- `src/utils/` — `restart_job.py` / `restart_job_utils.py` (job restart +
  `RestartAndLoadCheckpointCallback`, resumes from latest checkpoint),
  `inference_utils.py` (`LocalPickleWriter` merges prediction shards into
  `merged_predictions_tensor.pt`), `file_utils.py`, `instantiators.py`,
  `pylogger.py` (`RankedLogger`, rank-zero logging).

## Data layout expected by configs

The configs read from these subfolders of `data_dir` (note: the folder names in
the configs differ from the README's prose):

```
{data_dir}/items        # item text + ids (stages 1–2)
{data_dir}/training      # user-history train split (stage 3)
{data_dir}/evaluation    # validation split
{data_dir}/testing       # test split
```

## Outputs

Hydra writes each run to a timestamped dir (`${hydra:runtime.output_dir}`, under
`logs/`). Checkpoints land in `<output_dir>/checkpoints`, restart metadata in
`<output_dir>/metadata`, and pickled predictions under `<output_dir>/pickle`.
The `merged_predictions_tensor.pt` you pass to the next stage lives in that run's
output dir.
