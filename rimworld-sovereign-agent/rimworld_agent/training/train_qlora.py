"""QLoRA fine-tuning of the base SLM on the RimWorld training mix (project spec §7a).

Order is critical (vocab-extend-qlora convention, gotcha #9): Unsloth 4-bit load + patches
-> extend tokenizer (mined tokens + SID tokens + action/vision special tokens) -> resize
embeddings -> ``get_peft_model`` with ``modules_to_save=["embed_tokens","lm_head"]`` ->
SFTTrainer. The new-token rows are initialised meaningfully (mean for freq tokens,
codebook-projected for SID tokens) by vocab-extend-qlora's ``extend``.

GPU-only; written-but-unverified in this environment.
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

log = get_logger("train_qlora")


def _all_special_tokens(cfg) -> list[str]:
    """Action + vision structural tokens added on top of the SID tokens veq registers."""
    from rimworld_agent.game.action_space import action_special_tokens
    from rimworld_agent.vision.state_encoder import vision_special_tokens

    extra = action_special_tokens() + vision_special_tokens()
    return extra


def train(cfg) -> dict:
    ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None))
    from src.extend_tokenizer import extend  # type: ignore

    from rimworld_agent.knowledge.extend_tokenizer import build_game_token_lists
    from rimworld_agent.training.prepare_data import build_examples, save_dataset

    veq = to_veq_cfg(cfg)
    model_id = cfg_get(cfg, "model.id", "Qwen/Qwen2.5-Coder-1.5B-Instruct")

    # 1. Unsloth 4-bit load (patches BEFORE resize_token_embeddings).
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=cfg_get(cfg, "training.max_seq_len", 4096),
        load_in_4bit=cfg_get(cfg, "training.qlora.load_in_4bit", True),
        dtype=None,
    )

    # 2. Extend tokenizer with game vocab + SID tokens (reuses veq.extend).
    freq_tokens, sid_vocab, codebook_vectors = build_game_token_lists(cfg)
    extend_report = extend(tokenizer, model, freq_tokens, sid_vocab, codebook_vectors, veq)
    # Register action/vision structural tokens too (mean-initialised by HF resize default;
    # extend() already handled freq+SID rows).
    added = tokenizer.add_special_tokens({"additional_special_tokens": _all_special_tokens(cfg)})
    if added:
        model.resize_token_embeddings(len(tokenizer))

    # 3. LoRA with embed/lm_head trainable so the new rows learn.
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg_get(cfg, "training.qlora.lora_r", 32),
        lora_alpha=cfg_get(cfg, "training.qlora.lora_alpha", 64),
        lora_dropout=cfg_get(cfg, "training.qlora.lora_dropout", 0.0),
        target_modules=cfg_get(cfg, "training.qlora.target_modules", "all-linear"),
        modules_to_save=cfg_get(cfg, "training.qlora.modules_to_save", ["embed_tokens", "lm_head"]),
        use_gradient_checkpointing=cfg_get(cfg, "training.qlora.gradient_checkpointing", True),
    )

    # 4. Data + SFT.
    examples = build_examples(cfg)
    ds_dir = save_dataset(examples, cfg_get(cfg, "paths.dataset_dir", "data/dataset"))
    from datasets import load_from_disk
    from trl import SFTConfig, SFTTrainer

    dataset = load_from_disk(str(ds_dir))
    sft_cfg = SFTConfig(
        output_dir=cfg_get(cfg, "paths.model_out_dir", "data/models") + "/qlora",
        per_device_train_batch_size=cfg_get(cfg, "training.qlora.batch_size", 4),
        gradient_accumulation_steps=cfg_get(cfg, "training.qlora.grad_accum", 4),
        learning_rate=cfg_get(cfg, "training.qlora.lr", 2e-4),
        num_train_epochs=cfg_get(cfg, "training.qlora.epochs", 3),
        warmup_ratio=cfg_get(cfg, "training.qlora.warmup_ratio", 0.05),
        lr_scheduler_type=cfg_get(cfg, "training.qlora.lr_scheduler", "cosine"),
        optim=cfg_get(cfg, "training.qlora.optim", "paged_adamw_8bit"),
        bf16=cfg_get(cfg, "training.qlora.precision", "bf16") == "bf16",
        logging_steps=10,
        dataset_text_field="text",
        max_seq_length=cfg_get(cfg, "training.max_seq_len", 4096),
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=sft_cfg)
    train_result = trainer.train()

    out_dir = ensure_dir(Path(cfg_get(cfg, "paths.model_out_dir", "data/models")) / "qlora")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(out_dir)
    report = {
        "train_loss": float(train_result.training_loss),
        "extend_report": extend_report,
        "num_examples": len(examples),
        "added_special_tokens": int(added),
    }
    write_json(report, Path(cfg_get(cfg, "paths.results_dir", "results")) / "train_report.json")
    return report


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        train(cfg)

    _run()


if __name__ == "__main__":
    main()
