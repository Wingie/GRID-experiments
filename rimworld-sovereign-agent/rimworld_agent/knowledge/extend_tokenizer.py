"""Extend the base model's tokenizer with mined RimWorld tokens + semantic-ID tokens and
re-initialise the new embedding rows meaningfully.

This delegates the heavy lifting to vocab-extend-qlora's ``src.extend_tokenizer.extend``
(which adds tokens, resizes embeddings, and initialises new rows). We only assemble the
game-specific inputs:
  * frequency tokens  -> from :mod:`rimworld_agent.knowledge.mine_tokens`;
  * SID vocabulary + RQ-VAE codebook vectors -> from
    :mod:`rimworld_agent.semantic_ids.assign_ids`.

Per spec §3b, SID tokens are initialised from the RQ-VAE codebook geometry (``codebook``
init) and frequency tokens from their mean sub-token embedding. ``modules_to_save`` must
include ``embed_tokens``/``lm_head`` downstream so LoRA can train the new rows.
Requires a GPU/network to load the base model; the assembly logic is otherwise CPU-safe.
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
    """Return ``(freq_tokens, sid_vocab, codebook_vectors)`` for the extension step."""
    if not ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None)):
        raise RuntimeError("vocab-extend-qlora must be importable.")
    from src.mine_tokens import select_top_n  # type: ignore

    from rimworld_agent.knowledge.mine_tokens import mine_game_tokens

    freq_tokens: list[str] = []
    if cfg_get(cfg, "extend_vocab", True):
        ranked = mine_game_tokens(cfg)
        freq_tokens = select_top_n(ranked, cfg_get(cfg, "mining.top_n", 256))

    sid_vocab = None
    codebook_vectors = None
    if cfg_get(cfg, "semantic_ids.enabled", True):
        from rimworld_agent.semantic_ids.assign_ids import run_pipeline

        result = run_pipeline(cfg)
        sid_vocab = result.sid_vocab
        codebook_vectors = result.rqvae.codebook_vectors()
    return freq_tokens, sid_vocab, codebook_vectors


def load_model_and_tokenizer(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg_get(cfg, "model.id", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if cfg_get(cfg, "training.qlora.precision", "bf16") == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return tok, model


def extend_tokenizer(cfg) -> dict:
    """Run the full extension and persist the extended model + tokenizer."""
    ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None))
    from src.extend_tokenizer import extend  # type: ignore

    freq_tokens, sid_vocab, codebook_vectors = build_game_token_lists(cfg)
    tokenizer, model = load_model_and_tokenizer(cfg)
    report = extend(tokenizer, model, freq_tokens, sid_vocab, codebook_vectors, to_veq_cfg(cfg))

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
