"""QLoRA fine-tuning with Unsloth (2x speed / ~50% less VRAM on the RTX 3090).

Pipeline per experiment config:
  1. load the base model 4-bit via Unsloth (this applies Unsloth's patches FIRST);
  2. if the experiment extends the vocabulary, add freq + SID tokens and resize the
     embeddings (gotcha: must happen AFTER Unsloth patching, BEFORE get_peft_model);
  3. attach LoRA adapters with ``modules_to_save=["embed_tokens","lm_head"]`` so the new
     embedding rows are actually trained (non-negotiable for vocab extension);
  4. fine-tune on the prepared dataset; the multi-objective mix is realised by the
     dataset composition in ``prepare_data.py`` (objective_weights + sid_task_fraction).

Requires a CUDA GPU + Unsloth; not runnable on CPU.

Usage:
    python -m src.train_qlora configs/experiments/semid_qlora.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import cfg_get, ensure_dir, get_logger, load_config, set_seed, write_json

log = get_logger("train_qlora")


def _load_dataset(cfg: dict):
    """Load the prepared dataset, building it on the fly if absent."""
    dataset_dir = cfg_get(cfg, "paths.dataset_dir", "data/dataset")
    try:
        from datasets import load_from_disk

        return load_from_disk(dataset_dir)
    except Exception:
        log.info("No saved dataset at %s; building it now.", dataset_dir)
        from datasets import Dataset, DatasetDict

        from src.prepare_data import build_dataset

        train, evalset = build_dataset(cfg)
        return DatasetDict(
            {"train": Dataset.from_list(train), "eval": Dataset.from_list(evalset)}
        )


def train(cfg: dict) -> dict:
    import torch
    from unsloth import FastLanguageModel

    set_seed(cfg_get(cfg, "seed", 42))
    model_id = cfg_get(cfg, "model.id")
    max_seq = cfg_get(cfg, "model.max_seq_len", 2048)
    precision = cfg_get(cfg, "qlora.precision", "bf16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    # 1. Unsloth load (patches applied here).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq,
        dtype=dtype,
        load_in_4bit=cfg_get(cfg, "qlora.load_in_4bit", True),
    )

    # 2. Vocabulary extension (after patching, before PEFT).
    extend_report = {}
    if cfg_get(cfg, "extend_vocab", False) or cfg_get(cfg, "semantic_ids.enabled", False):
        from src.extend_tokenizer import build_token_lists, extend

        freq_tokens, sid_vocab, codebook_vectors = build_token_lists(cfg)
        extend_report = extend(tokenizer, model, freq_tokens, sid_vocab, codebook_vectors, cfg)

    # 3. LoRA adapters. modules_to_save trains the (possibly new) embeddings + head.
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg_get(cfg, "qlora.lora_r", 32),
        lora_alpha=cfg_get(cfg, "qlora.lora_alpha", 64),
        lora_dropout=cfg_get(cfg, "qlora.lora_dropout", 0.0),
        target_modules=cfg_get(cfg, "qlora.target_modules", "all-linear"),
        modules_to_save=cfg_get(cfg, "extend.modules_to_save", ["embed_tokens", "lm_head"]),
        use_gradient_checkpointing="unsloth"
        if cfg_get(cfg, "qlora.gradient_checkpointing", True)
        else False,
        random_state=cfg_get(cfg, "seed", 42),
    )

    # 4. Trainer.
    from transformers import TrainingArguments
    from trl import SFTTrainer

    dataset = _load_dataset(cfg)
    out_dir = str(ensure_dir(Path(cfg_get(cfg, "paths.model_out_dir", "data/models")) / cfg.get("name", "run")))

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg_get(cfg, "qlora.batch_size", 4),
        gradient_accumulation_steps=cfg_get(cfg, "qlora.grad_accum", 4),
        num_train_epochs=cfg_get(cfg, "qlora.epochs", 3),
        learning_rate=cfg_get(cfg, "qlora.lr", 2e-4),
        warmup_ratio=cfg_get(cfg, "qlora.warmup_ratio", 0.05),
        lr_scheduler_type=cfg_get(cfg, "qlora.lr_scheduler", "cosine"),
        optim=cfg_get(cfg, "qlora.optim", "paged_adamw_8bit"),
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        logging_steps=10,
        save_strategy="epoch",
        report_to=["wandb"] if cfg_get(cfg, "logging.use_wandb", False) else [],
        seed=cfg_get(cfg, "seed", 42),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        dataset_text_field="text",
        max_seq_length=max_seq,
        args=args,
    )
    train_result = trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    report = {
        "config_name": cfg.get("name"),
        "model": model_id,
        "output_dir": out_dir,
        "extend": extend_report,
        "train_loss": float(train_result.training_loss),
    }
    write_json(report, Path(cfg_get(cfg, "paths.results_dir", "results")) / f"train_{cfg.get('name','run')}.json")
    log.info("Training complete -> %s", out_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if not cfg_get(cfg, "do_finetune", True):
        log.warning("do_finetune is false for %s; nothing to train.", cfg.get("name"))
        return
    train(cfg)


if __name__ == "__main__":
    main()
