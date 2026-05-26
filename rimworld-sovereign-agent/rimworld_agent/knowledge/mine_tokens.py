"""Mine tokenizer-inefficient RimWorld terms and score them for vocabulary extension.

RimWorld text is full of multi-word identifiers that tokenize badly and recur thousands of
times (``ThingDef``, ``ResearchProjectDef``, ``HediffDefOf``, ``StatDefOf.MoveSpeed``) plus
wiki concepts (``mechanoid cluster``, ``psychic drone``). We build a corpus from the parsed
Defs, the C# entities, and the scraped wiki, then reuse vocab-extend-qlora's frequency
scorers (``score = frequency * current_subtoken_count``) to pick the top-N additions.

Reuse: this is a thin domain wrapper over ``src.mine_tokens`` (vocab-extend-qlora).
"""

from __future__ import annotations

from pathlib import Path

from rimworld_agent.utils import (
    cfg_get,
    ensure_veq_importable,
    get_logger,
    to_veq_cfg,
    write_json,
)

log = get_logger("mine_tokens")


def build_corpus(
    entities_blobs: list[str],
    csharp_blobs: list[str],
    wiki_dir: str | Path | None,
) -> list[str]:
    """Assemble the text corpus mining runs over."""
    texts = list(entities_blobs) + list(csharp_blobs)
    if wiki_dir:
        for md in sorted(Path(wiki_dir).glob("*.md")):
            texts.append(md.read_text(encoding="utf8", errors="replace"))
    return [t for t in texts if t.strip()]


def mine_game_tokens(cfg) -> list[dict]:
    """Score RimWorld term candidates. Returns a ranked list of ``{token, score, ...}``."""
    if not ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None)):
        raise RuntimeError("vocab-extend-qlora must be importable (see utils.ensure_veq_importable).")

    from src.mine_tokens import mine_identifiers, mine_ngrams  # type: ignore
    from src.utils import load_hf_tokenizer  # type: ignore

    from rimworld_agent.knowledge.extract_defs import DEFAULT_DEF_TYPES, extract_defs

    veq = to_veq_cfg(cfg)

    # Defs corpus
    xml_dir = cfg_get(cfg, "paths.xml_defs_dir", "data/rimworld_xml_defs")
    def_types = cfg_get(cfg, "knowledge.def_types", DEFAULT_DEF_TYPES)
    entities = extract_defs(xml_dir, def_types)
    def_blobs = [e.text_blob() for e in entities]

    # C# corpus (optional; needs the csharp extra)
    csharp_blobs: list[str] = []
    if cfg_get(cfg, "knowledge.mine_csharp", True):
        try:
            from rimworld_agent.knowledge.extract_csharp import extract_csharp

            csharp_blobs = [
                e.text_blob()
                for e in extract_csharp(cfg_get(cfg, "paths.csharp_source_dir", "data/rimworld_source"))
            ]
        except Exception as exc:  # tree-sitter missing or no source
            log.warning("skipping C# corpus: %s", exc)

    corpus = build_corpus(def_blobs, csharp_blobs, cfg_get(cfg, "paths.wiki_dir", "data/rimworld_wiki"))
    log.info("mining over %d documents", len(corpus))

    tokenizer, is_real = load_hf_tokenizer(
        cfg_get(cfg, "model.id", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        allow_fallback=cfg_get(cfg, "knowledge.allow_offline_tokenizer", True),
    )
    if not is_real:
        log.warning("using offline FallbackTokenizer; scores are a sanity-check, not final.")

    ranked: dict[str, dict] = {}
    ranked.update(
        mine_identifiers(corpus, tokenizer, cfg_get(veq, "mining.identifier_min_frequency", 5))
    )
    ranked.update(
        mine_ngrams(
            corpus,
            tokenizer,
            cfg_get(veq, "mining.ngram_sizes", [2, 3]),
            cfg_get(veq, "mining.ngram_min_frequency", 25),
        )
    )
    out = sorted(ranked.values(), key=lambda d: d.get("score", 0), reverse=True)
    return out


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        ranked = mine_game_tokens(cfg)
        top_n = cfg_get(cfg, "mining.top_n", 256)
        out = cfg_get(cfg, "paths.candidates_file", "results/candidates.json")
        write_json({"top_n": top_n, "candidates": ranked[: top_n * 2], "all_count": len(ranked)}, out)
        log.info("mined %d candidates; top tokens: %s", len(ranked), [c["token"] for c in ranked[:10]])

    _run()


if __name__ == "__main__":
    main()
