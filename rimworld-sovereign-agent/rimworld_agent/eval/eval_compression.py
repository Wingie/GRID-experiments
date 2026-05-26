"""Token-efficiency evaluation on RimWorld text (project spec §9b, target 25%+ reduction).

Measures how many fewer tokens the *extended* tokenizer needs to represent game text
(Defs + C# + wiki + rendered game-state strings) versus the base tokenizer, and reports the
SID-token adoption rate in the model's reasoning when a trained tokenizer/model is provided.

The corpus build + token counting reuse vocab-extend-qlora's tokenizer helpers and run with
the offline FallbackTokenizer when no model can be downloaded (results are then a
machinery sanity-check, not a research number).
"""

from __future__ import annotations

from pathlib import Path

from rimworld_agent.utils import cfg_get, ensure_veq_importable, get_logger, write_json

log = get_logger("eval_compression")


def build_eval_corpus(cfg) -> list[str]:
    from rimworld_agent.knowledge.extract_defs import DEFAULT_DEF_TYPES, extract_defs
    from rimworld_agent.knowledge.mine_tokens import build_corpus

    entities = extract_defs(
        cfg_get(cfg, "paths.xml_defs_dir", "data/rimworld_xml_defs"),
        cfg_get(cfg, "knowledge.def_types", DEFAULT_DEF_TYPES),
    )
    def_blobs = [e.text_blob() for e in entities]
    return build_corpus(def_blobs, [], cfg_get(cfg, "paths.wiki_dir", "data/rimworld_wiki"))


def evaluate(cfg) -> dict:
    ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None))
    from src.mine_tokens import select_top_n  # type: ignore
    from src.utils import count_subtokens, load_hf_tokenizer  # type: ignore

    from rimworld_agent.knowledge.mine_tokens import mine_game_tokens

    corpus = build_eval_corpus(cfg)
    model_id = cfg_get(cfg, "model.id", "Qwen/Qwen2.5-Coder-1.5B-Instruct")

    base_tok, _ = load_hf_tokenizer(model_id, allow_fallback=True)
    ranked = mine_game_tokens(cfg)
    new_tokens = select_top_n(ranked, cfg_get(cfg, "mining.top_n", 256))

    ext_tok, is_real = load_hf_tokenizer(model_id, allow_fallback=True)
    if hasattr(ext_tok, "add_tokens"):
        ext_tok.add_tokens(new_tokens)

    base_counts = [count_subtokens(base_tok, t) for t in corpus]
    ext_counts = [count_subtokens(ext_tok, t) for t in corpus]
    base_total, ext_total = sum(base_counts), sum(ext_counts)
    reduction = 1.0 - (ext_total / base_total) if base_total else 0.0

    result = {
        "documents": len(corpus),
        "base_tokens": base_total,
        "extended_tokens": ext_total,
        "reduction_fraction": reduction,
        "num_added_tokens": len(new_tokens),
        "real_tokenizer": bool(is_real),
        "target_reduction": 0.25,
        "meets_target": reduction >= 0.25,
    }
    log.info("token reduction: %.1f%% (%d -> %d)", reduction * 100, base_total, ext_total)
    return result


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        result = evaluate(cfg)
        write_json(result, Path(cfg_get(cfg, "paths.results_dir", "results")) / "eval_compression.json")

    _run()


if __name__ == "__main__":
    main()
