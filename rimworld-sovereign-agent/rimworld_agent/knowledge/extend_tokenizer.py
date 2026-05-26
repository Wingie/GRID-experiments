"""Extend the base model's tokenizer with mined RimWorld tokens + DUAL semantic-ID tokens
(RSID + WSID) + action/vision structural tokens, and initialise the new embedding rows
meaningfully (project spec §3b, token budget ~646 new tokens).

Initialisation reuses vocab-extend-qlora's primitives:
  * frequency tokens     -> mean of their sub-token embeddings;
  * RSID per-level tokens -> READ codebook vectors projected into embedding space;
  * WSID per-level tokens -> WRITE codebook vectors projected into embedding space;
  * structural/action/vision tokens -> the mean embedding (no codebook entry).

Both SID families use the SAME projection geometry helper so nearby codes -> nearby init
vectors. `modules_to_save=[embed_tokens, lm_head]` downstream lets LoRA train the new rows.
Loading the base model needs a GPU/network; the assembly logic is otherwise CPU-safe.
"""

from __future__ import annotations

from pathlib import Path

from rimworld_agent.utils import (
    cfg_get,
    ensure_dir,
    ensure_veq_importable,
    get_logger,
    to_veq_cfg,
    write_json,
)

log = get_logger("extend_tokenizer")


def build_game_token_lists(cfg):
    """Return ``(freq_tokens, dual_result)`` where dual_result has the READ/WRITE vocabs +
    codebooks needed to initialise the new rows.
    """
    if not ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None)):
        raise RuntimeError("vocab-extend-qlora must be importable.")
    from src.mine_tokens import select_top_n  # type: ignore

    from rimworld_agent.knowledge.mine_tokens import mine_game_tokens

    freq_tokens: list[str] = []
    if cfg_get(cfg, "extend_vocab", True):
        ranked = mine_game_tokens(cfg)
        freq_tokens = select_top_n(ranked, cfg_get(cfg, "mining.top_n", 256))

    dual = None
    if cfg_get(cfg, "semantic_ids.enabled", True):
        from rimworld_agent.semantic_ids.assign_ids import run_pipeline

        dual = run_pipeline(cfg)
    return freq_tokens, dual


def load_model_and_tokenizer(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg_get(cfg, "model.id", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if cfg_get(cfg, "training.qlora.precision", "bf16") == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return tok, model


def extend_dual(tokenizer, model, freq_tokens, dual, cfg) -> dict:
    """Add freq + dual-SID + action/vision tokens and initialise the new rows in place."""
    import torch

    ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None))
    from src.extend_tokenizer import (  # type: ignore
        codebook_projected_vectors,
        exp_weighted_init_vectors,
        mean_init_vectors,
    )

    from rimworld_agent.game.action_space import action_special_tokens
    from rimworld_agent.semantic_ids.sid_vocab import TASK_TOKENS
    from rimworld_agent.vision.state_encoder import vision_special_tokens

    old_emb = model.get_input_embeddings().weight.data.clone()
    embed_dim = old_emb.size(1)
    target_norm = old_emb.norm(dim=1).mean().item()
    mean_vec = old_emb.mean(dim=0, keepdim=True)
    method = cfg_get(cfg, "extend.init_method", "codebook")

    # --- frequency tokens ----------------------------------------------------
    freq_subtoks = [tokenizer(t, add_special_tokens=False).input_ids for t in freq_tokens]
    if method == "exp_weighted":
        freq_vecs = exp_weighted_init_vectors(old_emb, freq_subtoks, cfg_get(cfg, "extend.exp_weight_decay", 0.5))
    else:
        freq_vecs = mean_init_vectors(old_emb, freq_subtoks)

    # --- assemble special tokens + their init vectors, in add order ----------
    special_tokens: list[str] = []
    special_vecs: list = []

    def _add_sid_family(vocab, codebook):
        nonlocal special_tokens, special_vecs
        per_level = vocab.per_level_tokens()
        special_tokens += per_level
        if method == "codebook" and codebook is not None:
            proj = codebook_projected_vectors(codebook, embed_dim, target_norm)  # [L*K, D]
            special_vecs.append(proj)
        else:
            special_vecs.append(mean_vec.repeat(len(per_level), 1))
        struct = vocab.structural_tokens()
        special_tokens += struct
        special_vecs.append(mean_vec.repeat(len(struct), 1))

    _add_sid_family(dual.read_vocab, dual.read_rqvae.codebook_vectors() if dual.read_rqvae else None)
    if dual.write_vocab is not None:
        _add_sid_family(dual.write_vocab, dual.write_rqvae.codebook_vectors() if dual.write_rqvae else None)

    extra = TASK_TOKENS + action_special_tokens() + vision_special_tokens()
    special_tokens += extra
    special_vecs.append(mean_vec.repeat(len(extra), 1))
    special_init = torch.cat(special_vecs, dim=0) if special_vecs else None

    # --- add, resize, write rows --------------------------------------------
    base = old_emb.size(0)
    num_freq = tokenizer.add_tokens(freq_tokens)
    num_special = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    new_emb = model.get_input_embeddings().weight.data

    if num_freq:
        new_emb[base : base + len(freq_tokens)] = freq_vecs.to(new_emb.dtype)
    if num_special and special_init is not None:
        start = base + len(freq_tokens)
        new_emb[start : start + len(special_tokens)] = special_init.to(new_emb.dtype)

    out_emb = model.get_output_embeddings()
    if out_emb is not None and out_emb.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        out_emb.weight.data[base:] = new_emb[base:].clone()

    report = {
        "init_method": method,
        "num_freq_added": int(num_freq),
        "num_special_added": int(num_special),
        "num_rsid_tokens": len(dual.read_vocab.per_level_tokens()),
        "num_wsid_tokens": len(dual.write_vocab.per_level_tokens()) if dual.write_vocab else 0,
        "old_vocab_size": int(base),
        "new_vocab_size": int(len(tokenizer)),
    }
    log.info("extended tokenizer: +%d freq, +%d special (RSID+WSID+action+vision)", num_freq, num_special)
    return report


def extend_tokenizer(cfg) -> dict:
    freq_tokens, dual = build_game_token_lists(cfg)
    if dual is None:
        raise RuntimeError("semantic_ids.enabled must be true to build the dual SID vocabularies.")
    tokenizer, model = load_model_and_tokenizer(cfg)
    report = extend_dual(tokenizer, model, freq_tokens, dual, cfg)

    out_dir = ensure_dir(Path(cfg_get(cfg, "paths.model_out_dir", "data/models")) / "extended")
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    write_json(report, Path(cfg_get(cfg, "paths.results_dir", "results")) / "extend_report.json")
    log.info("saved extended model + tokenizer -> %s", out_dir)
    return report


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        extend_tokenizer(cfg)

    _run()


if __name__ == "__main__":
    main()
