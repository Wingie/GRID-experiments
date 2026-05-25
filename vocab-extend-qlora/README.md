# Semantic IDs Meet Tokenizer Extension: Hierarchical Code Entity Representations for Codebase-Specific SLMs on Consumer GPUs

[![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-green)](https://www.python.org/)

> Research code investigating whether **tokenizer-vocabulary extension**, **QLoRA
> fine-tuning**, and a novel **hierarchical semantic-ID layer for code entities** can be
> combined to produce compact, fast, codebase-specific Small Language Models — entirely
> on a single RTX 3090 (24 GB).

## Abstract

Codebase-specific code models pay a hidden tax: a general-purpose tokenizer shatters
project identifiers (`AgentosaurusConfig`, `validate_schema`) into many sub-tokens,
inflating sequence length, decode steps, and latency. Prior work attacks this by
extending the vocabulary with frequent domain tokens and re-initialising their
embeddings from sub-token means (AdaptiVocab) or gradient-selected subsets (VEGAD),
recovering 20–28% of tokens. Separately, generative-retrieval research (TIGER) showed
that items can be represented as short sequences of **hierarchical discrete semantic
IDs** learned by an RQ-VAE — coarse category → finer category → specific item.

We unify these threads. We apply the TIGER/RQ-VAE paradigm to **code entities**: each
class, function, and method is embedded by a code encoder and quantised into an
`L`-level semantic ID (`module cluster → class cluster → method role`). These SID tokens
are added to the tokenizer alongside frequency-mined tokens, initialised from the
RQ-VAE codebook (not random), and trained into a code-generation SLM with QLoRA under a
four-objective mix (causal LM, entity→SID, SID→signature, SID-aware completion). On a
single RTX 3090 we benchmark four open code models across six conditions and report
compression, completion accuracy, inference latency, and a suite of semantic-ID quality
metrics (cluster coherence, hierarchical consistency, SID prediction, novel-entity
generalization). The repository is fully config-driven and reproducible end to end.

## Why this is new

| Prior work | Domain | Discrete IDs? | Generation? | Tokenizer ext? |
|---|---|---|---|---|
| AdaptiVocab / VEGAD | text/code | no | yes | **yes** |
| TIGER | products (recsys) | **yes (RQ-VAE)** | yes (item IDs) | no |
| CodeDSI | code **search** | yes (docids) | no (retrieval) | no |
| code2vec / code2seq | code (continuous embeds) | no | name prediction | no |
| **This work** | **code entities** | **yes (RQ-VAE)** | **yes (code)** | **yes** |

CodeDSI applied the Differentiable Search Index to code *retrieval*; code2vec learned
*continuous* embeddings; TIGER applied semantic IDs to *products*. We combine all three:
**discrete hierarchical IDs for code entities, added to the tokenizer vocabulary, and
trained into a code-generation SLM.**

## The idea in one example

```python
# Before (raw tokenization): the name alone costs many sub-tokens
class AgentosaurusConfig:        # ~5 tokens
    def load_from_yaml(self): ...   # ~6 tokens
    def validate_schema(self): ...  # ~5 tokens

# After (semantic-ID tokens): a compact, structured representation
class <SID_L1_12><SID_L2_07>:                 # 2 tokens
    def <SID_L1_12><SID_L2_07><SID_L3_03>():  # 3 tokens
    def <SID_L1_12><SID_L2_07><SID_L3_08>():  # 3 tokens
```

The model learns that `<SID_L1_12>*` entities share a module-level purpose and can
generate valid completions by predicting SID tokens autoregressively — exactly as TIGER
generates item IDs.

## Installation

```bash
# Prereqs: Python 3.10+, an NVIDIA GPU (RTX 3090 / 24 GB tested), CUDA toolkit.
pip install -r requirements.txt
# Unsloth is installed per its official, CUDA-specific instructions, e.g.:
#   pip install "unsloth[cu124] @ git+https://github.com/unslothai/unsloth.git"
```

For a CPU-only checkout (no training; mining / entity extraction / RQ-VAE / compression
analysis / tests still run):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets pyyaml numpy sacrebleu Levenshtein scikit-learn pandas tqdm pytest
```

## Quick start

```bash
# 1. Clone the codebase to learn from (any git URL; default is a sample repo).
scripts/clone_target.sh https://github.com/wingie/agentosaurus

# 2. Run the full pipeline for one experiment condition.
scripts/run_all.sh configs/experiments/semid_qlora.yaml
```

Or step by step:

```bash
python -m src.mine_tokens          configs/experiments/semid_qlora.yaml
python -m src.semantic_ids.assign_ids configs/experiments/semid_qlora.yaml
python -m src.prepare_data         configs/experiments/semid_qlora.yaml
python -m src.train_qlora          configs/experiments/semid_qlora.yaml      # GPU
python -m src.eval_compression     configs/experiments/semid_qlora.yaml
python -m src.eval_completion      configs/experiments/semid_qlora.yaml model_path=data/models/semid_qlora variant=full  # GPU
python -m src.eval_semantic_ids    configs/experiments/semid_qlora.yaml model_path=data/models/semid_qlora              # GPU
python -m src.merge_and_export     configs/experiments/semid_qlora.yaml adapter_dir=data/models/semid_qlora             # GPU
```

## Experiment conditions

Six configs in `configs/experiments/` isolate each contribution:

| Config | Vocab ext | Semantic IDs | Fine-tune |
|---|---|---|---|
| `baseline` | – | – | – |
| `qlora_only` | – | – | ✅ |
| `vocab_only` | ✅ (freq) | – | – |
| `vocab_qlora` | ✅ (freq) | – | ✅ |
| `semid_only` | ✅ (SID) | ✅ | – |
| `semid_qlora` | ✅ (freq+SID) | ✅ | ✅ |

## How it works

1. **Token mining** (`src/mine_tokens.py`) — frequent identifiers scored by
   `frequency × sub-token count`, plus frequent sub-token n-grams (AdaptiVocab-style),
   plus an optional gradient-based selector (VEGAD-style).
2. **Semantic IDs** (`src/semantic_ids/`) — tree-sitter (with a stdlib-`ast` fallback for
   Python) extracts entities; a code encoder embeds them; a small **RQ-VAE**
   (`rqvae.py`, EMA codebooks + dead-code reinit) quantises each into an `L`-tuple of
   codes; `assign_ids.py` formats them as special tokens; `inject_ids.py` builds the
   training formats and task objectives.
3. **Tokenizer extension** (`src/extend_tokenizer.py`) — adds freq tokens
   (`add_tokens`) and SID tokens (`add_special_tokens`), resizes embeddings, and
   initialises new rows via mean-of-subtokens, exponentially-weighted, or
   **codebook-projected** (SID rows seeded from the RQ-VAE codebook geometry).
4. **QLoRA training** (`src/train_qlora.py`) — Unsloth 4-bit, LoRA with
   `modules_to_save=["embed_tokens","lm_head"]`, four-objective data mix.
5. **Evaluation** (`src/eval_*.py`) — compression, completion accuracy, latency, and
   semantic-ID quality.

## Results

Result tables are produced by `notebooks/03_results.ipynb` from the per-condition JSON
in `results/`. Templates (to be populated by your runs on the target repo):

**Table 1 — Compression** (% input-token reduction vs. base tokenizer)

| Model | freq | SID | freq+SID | bytes/token |
|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

**Table 2 — Completion accuracy** (6 conditions × 4 models): EM / BLEU-4 / CodeBLEU /
edit-ratio / syntax-validity.

**Table 3 — Inference latency**: tokens/sec, TTFT, end-to-end (transformers + Ollama).

**Table 4 — Semantic-ID quality**: cluster coherence (intra vs inter), L1/L2
hierarchical consistency, SID prediction top-1/3 per level, novel-entity generalization
lift over chance.

**Table 5 — Ablations**: vocab size, init method, LoRA rank, `modules_to_save`,
selection strategy, RQ-VAE `L`/`K`.

**Figure 1** — t-SNE/UMAP of the codebook (`notebooks/02`). **Figure 2** — class→method
SID hierarchy tree.

## Hardware requirements

- **GPU:** RTX 3090 24 GB (tested target). No hardware FP8 → bf16 throughout.
- **RAM:** ~16 GB minimum. **Disk:** ~20 GB for models + data.
- **Training time (rough, 3 epochs, 3090):** 1.5B ≈ 30–60 min; 7B ≈ 3–5 h depending on
  corpus size. The RQ-VAE is tiny (<5M params) and trains in minutes (CPU-fine).

## Repository layout

```
configs/         base + per-model + per-experiment YAML (inheritance-merged)
src/             pipeline stages (mining, semantic_ids/, extend, prepare, train, eval, export)
notebooks/       token analysis, SID viz, results, ablations
scripts/         clone_target.sh, run_all.sh, export_ollama.sh
tests/           CPU-only unit tests (entity extraction, SID vocab, RQ-VAE, init math)
```

> **Note on CI:** `.github/workflows/ci.yml` and `.pre-commit-config.yaml` are included,
> but GitHub only runs workflows from a repository root. This project currently lives as
> a subdirectory of another repo, so CI activates only once it is split into its own
> repository (`git subtree split --prefix vocab-extend-qlora -b vocab-extend-qlora`).

## Related work

**Tokenizer extension**
1. AdaptiVocab — Nakash et al., 2025 — [arXiv:2503.19693](https://arxiv.org/abs/2503.19693) — [code](https://github.com/itay-nakash/AdaptiVocab). 22.9–27.9% input / 24.9–27.6% output token reduction; exponentially-weighted embedding init.
2. Vocabulary Customization for Domain-Specific LLM Deployment — 2025 — [arXiv:2509.26124](https://arxiv.org/abs/2509.26124). Up to 20% sequence shortening; token-adoption + forward-speed analysis.
3. VEGAD — Liu et al., 2024 — [arXiv:2410.01188](https://arxiv.org/abs/2410.01188). Gradient-based vocabulary subset selection; subset expansion > full expansion.
4. Medical Clinical Tokens — 2025 — [PMC12910058](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12910058/). BPE domain tokenizer for LLaMA-2.
5. Tokenizer Optimization for Domain Adaptation — 2024 — [arXiv:2402.01035](https://arxiv.org/abs/2402.01035). 32k–256k vocab size has minimal downstream impact.

**Semantic IDs / generative retrieval**
6. TIGER — Rajput et al., 2023 (NeurIPS 36). RQ-VAE hierarchical semantic IDs; autoregressive item-ID generation; first codebook = coarse category, second = finer.
7. SE-DSI — Tang et al., 2023 (KDD) — [arXiv:2305.15115](https://arxiv.org/abs/2305.15115). Semantically meaningful docids beat arbitrary integers.
8. CodeDSI — Nadeem et al., 2022 — [arXiv:2210.00328](https://arxiv.org/abs/2210.00328). DSI for code *search*; semantic-clustering docids +2–6%; numeric > character docids.
9. Spotify Semantic IDs — 2025 (Spotify Research). Joint semantic IDs for search + recommendation via RQ-KMeans.

**Code representation**
10. code2vec — Alon et al., 2019 (POPL) — [arXiv:1803.09473](https://arxiv.org/abs/1803.09473). AST path-based embeddings for method-name prediction.
11. code2seq — Alon et al., 2019 (ICLR). Encoder–decoder over AST path-contexts.
12. SeCoT — Ma et al., 2024 — [arXiv:2310.10698](https://arxiv.org/abs/2310.10698). Semantic Chain-of-Thought injecting data/control-flow.

## Citation

```bibtex
@misc{vocab_extend_qlora,
  title  = {Semantic IDs Meet Tokenizer Extension: Hierarchical Code Entity
            Representations for Codebase-Specific SLMs on Consumer GPUs},
  year   = {2025},
  note   = {https://github.com/wingie/vocab-extend-qlora}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
