"""Extend a tokenizer with mined tokens + semantic-ID tokens and resize the model's
embeddings with a meaningful initialisation.

Two complementary token sources:
  A. frequency-mined tokens  -> added with ``add_tokens`` (regular tokens).
  B. semantic-ID tokens       -> added with ``add_special_tokens`` (never split).

Initialisation strategies (``extend.init_method``):
  - random        : leave HF's default random rows (control).
  - mean          : mean of the new token's sub-token embeddings (proven for freq tokens).
  - exp_weighted  : AdaptiVocab-style exponentially weighted sub-token mean.
  - codebook      : SID tokens initialised from the RQ-VAE codebook vectors projected
                    into the model embedding space; freq tokens use ``mean``.

The init *primitives* (``mean_init_vectors`` etc.) take plain tensors so they can be
unit-tested without downloading a model.
"""

from __future__ import annotations

from pathlib import Path

from src.utils import cfg_get, get_logger

log = get_logger("extend_tokenizer")


# --------------------------------------------------------------------------- #
# Initialisation primitives (pure tensor ops; unit-testable)
# --------------------------------------------------------------------------- #
def mean_init_vectors(old_emb, subtoken_ids: list[list[int]]):
    """Mean of sub-token embeddings for each new token. ``old_emb`` is ``[V, D]``."""
    import torch

    out = torch.empty(len(subtoken_ids), old_emb.size(1), dtype=old_emb.dtype)
    for i, ids in enumerate(subtoken_ids):
        if ids:
            out[i] = old_emb[ids].mean(dim=0)
        else:
            out[i] = old_emb.mean(dim=0)
    return out


def exp_weighted_init_vectors(old_emb, subtoken_ids: list[list[int]], decay: float = 0.5):
    """AdaptiVocab-style exponentially weighted sub-token mean (earlier pieces weigh
    more). Weight of piece j is ``decay**j``, normalised across pieces.
    """
    import torch

    out = torch.empty(len(subtoken_ids), old_emb.size(1), dtype=old_emb.dtype)
    for i, ids in enumerate(subtoken_ids):
        if not ids:
            out[i] = old_emb.mean(dim=0)
            continue
        w = torch.tensor([decay**j for j in range(len(ids))], dtype=old_emb.dtype)
        w = w / w.sum()
        out[i] = (old_emb[ids] * w.unsqueeze(1)).sum(dim=0)
    return out


def codebook_projected_vectors(codebook_vectors, embed_dim: int, target_norm: float, seed: int = 0):
    """Project RQ-VAE codebook vectors ``[L, K, latent]`` into the model embedding
    space ``[L*K, embed_dim]`` via a fixed random linear map, scaled so row norms match
    the model's existing embeddings. The projection preserves the codebook geometry
    (nearby codes -> nearby init vectors) which random init destroys.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    L, K, latent = codebook_vectors.shape
    flat = codebook_vectors.reshape(L * K, latent).float()
    proj = torch.randn(latent, embed_dim, generator=g) / (latent**0.5)
    out = flat @ proj
    norms = out.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return out / norms * target_norm


# --------------------------------------------------------------------------- #
# Orchestration (needs a real HF tokenizer + model)
# --------------------------------------------------------------------------- #
def extend(tokenizer, model, freq_tokens: list[str], sid_vocab, codebook_vectors, cfg: dict) -> dict:
    """Add tokens to ``tokenizer``, resize ``model`` embeddings, and initialise the new
    rows per ``extend.init_method``. Returns a report dict. Mutates both in place.
    """
    import torch

    method = cfg_get(cfg, "extend.init_method", "mean")
    sid_tokens = sid_vocab.special_tokens() if sid_vocab is not None else []

    old_emb = model.get_input_embeddings().weight.data.clone()
    embed_dim = old_emb.size(1)
    target_norm = old_emb.norm(dim=1).mean().item()

    # Compute init vectors BEFORE adding tokens (so surfaces still tokenise to subwords).
    freq_subtokens = [tokenizer(t, add_special_tokens=False).input_ids for t in freq_tokens]
    if method == "exp_weighted":
        freq_vecs = exp_weighted_init_vectors(old_emb, freq_subtokens, cfg_get(cfg, "extend.exp_weight_decay", 0.5))
    elif method in ("mean", "codebook"):
        freq_vecs = mean_init_vectors(old_emb, freq_subtokens)
    else:  # random
        freq_vecs = None

    # SID per-level tokens get codebook-projected vectors when method == codebook.
    sid_vecs = None
    if sid_tokens and method == "codebook" and codebook_vectors is not None:
        per_level = codebook_projected_vectors(codebook_vectors, embed_dim, target_norm)
        # structural + task tokens have no codebook entry -> mean of all embeddings
        n_extra = len(sid_tokens) - per_level.size(0)
        extra = old_emb.mean(dim=0, keepdim=True).repeat(max(0, n_extra), 1)
        sid_vecs = torch.cat([per_level, extra], dim=0)[: len(sid_tokens)]

    num_freq = tokenizer.add_tokens(freq_tokens)
    num_sid = tokenizer.add_special_tokens({"additional_special_tokens": sid_tokens}) if sid_tokens else 0
    model.resize_token_embeddings(len(tokenizer))

    new_emb = model.get_input_embeddings().weight.data
    # Newly added rows are appended at the end, in add order: freq first, then sid.
    base = old_emb.size(0)
    if freq_vecs is not None and num_freq:
        new_emb[base : base + len(freq_tokens)] = freq_vecs.to(new_emb.dtype)
    if sid_vecs is not None and num_sid:
        start = base + len(freq_tokens)
        new_emb[start : start + len(sid_tokens)] = sid_vecs.to(new_emb.dtype)

    # mirror to output embeddings when they are not tied to the input embeddings
    out_emb = model.get_output_embeddings()
    if out_emb is not None and out_emb.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        out_emb.weight.data[base:] = new_emb[base:].clone()

    report = {
        "init_method": method,
        "num_freq_added": int(num_freq),
        "num_sid_added": int(num_sid),
        "old_vocab_size": int(base),
        "new_vocab_size": int(len(tokenizer)),
        "embed_dim": int(embed_dim),
    }
    log.info("Extended tokenizer: +%d freq, +%d sid (init=%s)", num_freq, num_sid, method)
    return report


def build_token_lists(cfg: dict):
    """Run mining (+ optional SID pipeline) and return
    ``(freq_tokens, sid_vocab, codebook_vectors)``.
    """
    from src.mine_tokens import mine, select_top_n

    freq_tokens: list[str] = []
    if cfg_get(cfg, "extend_vocab", False):
        ranked = mine(cfg)
        freq_tokens = select_top_n(ranked, cfg_get(cfg, "mining.top_n", 128))

    sid_vocab = None
    codebook_vectors = None
    if cfg_get(cfg, "semantic_ids.enabled", False):
        from src.semantic_ids.assign_ids import run_pipeline

        _e, _emb, model, sid_vocab, _assign = run_pipeline(cfg)
        codebook_vectors = model.codebook_vectors()
    return freq_tokens, sid_vocab, codebook_vectors


def load_model_and_tokenizer(cfg: dict):
    """Load the base model + tokenizer for extension/eval (needs download/GPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg_get(cfg, "model.id")
    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if cfg_get(cfg, "qlora.precision", "bf16") == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return tok, model


def main() -> None:
    import argparse

    from src.utils import ensure_dir, load_config, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    freq_tokens, sid_vocab, codebook_vectors = build_token_lists(cfg)
    tokenizer, model = load_model_and_tokenizer(cfg)
    report = extend(tokenizer, model, freq_tokens, sid_vocab, codebook_vectors, cfg)

    out_dir = ensure_dir(Path(cfg_get(cfg, "paths.model_out_dir", "data/models")) / "extended")
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    write_json(report, Path(cfg_get(cfg, "paths.results_dir", "results")) / "extend_report.json")
    log.info("Saved extended model + tokenizer -> %s", out_dir)


if __name__ == "__main__":
    main()
