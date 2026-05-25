"""Merge LoRA adapters into the base weights and export to GGUF for Ollama.

The extended tokenizer is saved alongside the merged model so GGUF conversion picks up
the new vocabulary automatically (no extra config needed). Requires Unsloth (preferred)
or a peft + llama.cpp toolchain; GPU recommended. Not runnable on CPU here.

Usage:
    python -m src.merge_and_export configs/experiments/semid_qlora.yaml \
        adapter_dir=data/models/semid_qlora
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import cfg_get, ensure_dir, get_logger, load_config

log = get_logger("merge_and_export")


def export_unsloth(cfg: dict, adapter_dir: str, quant: str) -> str:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=cfg_get(cfg, "model.max_seq_len", 2048),
        load_in_4bit=False,
    )
    out_dir = str(ensure_dir(Path(cfg_get(cfg, "paths.model_out_dir", "data/models")) / f"{cfg.get('name','run')}_merged"))
    # 16-bit merge keeps the trained embedding rows intact.
    model.save_pretrained_merged(out_dir, tokenizer, save_method="merged_16bit")
    # GGUF for Ollama / llama.cpp.
    model.save_pretrained_gguf(out_dir, tokenizer, quantization_method=quant)
    log.info("Exported merged model + GGUF (%s) -> %s", quant, out_dir)
    return out_dir


def export_peft(cfg: dict, adapter_dir: str) -> str:
    """Fallback merge via peft (no GGUF; convert separately with llama.cpp)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = cfg_get(cfg, "model.id")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16)
    # ensure base embeddings match the extended tokenizer before loading adapters
    base.resize_token_embeddings(len(tokenizer))
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    out_dir = str(ensure_dir(Path(cfg_get(cfg, "paths.model_out_dir", "data/models")) / f"{cfg.get('name','run')}_merged"))
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    log.info(
        "Merged via peft -> %s. Convert to GGUF with llama.cpp: "
        "`python convert_hf_to_gguf.py %s --outfile model.gguf`",
        out_dir,
        out_dir,
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    adapter_dir = cfg_get(cfg, "adapter_dir") or str(
        Path(cfg_get(cfg, "paths.model_out_dir", "data/models")) / cfg.get("name", "run")
    )
    quant = cfg_get(cfg, "export.quant", "q4_k_m")
    try:
        export_unsloth(cfg, adapter_dir, quant)
    except Exception as exc:
        log.warning("Unsloth export failed (%s); falling back to peft merge.", exc)
        export_peft(cfg, adapter_dir)


if __name__ == "__main__":
    main()
