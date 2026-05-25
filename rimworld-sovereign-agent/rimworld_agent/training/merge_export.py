"""Merge the QLoRA adapter into the base weights and export to GGUF for Ollama/llama.cpp.

Thin wrapper over vocab-extend-qlora's ``src.merge_and_export`` (Unsloth
``save_pretrained_merged`` / ``save_pretrained_gguf`` with a peft + llama.cpp fallback).
GPU-only.
"""

from __future__ import annotations

from pathlib import Path

from rimworld_agent.utils import cfg_get, ensure_veq_importable, get_logger, to_veq_cfg

log = get_logger("merge_export")


def merge_and_export(cfg) -> str:
    """Merge + export the trained adapter. Returns the output directory."""
    ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None))
    from src.merge_and_export import export_peft, export_unsloth  # type: ignore

    veq = to_veq_cfg(cfg)
    adapter_dir = cfg_get(cfg, "adapter_dir", None) or str(
        Path(cfg_get(cfg, "paths.model_out_dir", "data/models")) / "qlora"
    )
    quant = cfg_get(cfg, "export.quant", "q4_k_m")
    try:
        return export_unsloth(veq, adapter_dir, quant)
    except Exception as exc:
        log.warning("Unsloth export failed (%s); falling back to peft merge.", exc)
        return export_peft(veq, adapter_dir)


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        merge_and_export(cfg)

    _run()


if __name__ == "__main__":
    main()
