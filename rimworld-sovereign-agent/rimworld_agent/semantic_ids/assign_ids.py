"""Assign hierarchical semantic IDs to RimWorld entities and define the SID vocabulary.

After the RQ-VAE is trained on entity embeddings, every Def/class is mapped to an
``L``-tuple of codes and rendered as special tokens. We reuse vocab-extend-qlora's
:class:`SemanticIDVocab` (so the token grammar ``<SID_L{l}_{c}>`` + structural/task tokens
is byte-for-byte identical to the code-model work) and add the RimWorld-specific
extract -> graph -> embed -> train -> assign pipeline.

Each entity is encoded independently (TIGER-style); the entity-graph category is attached
to each assignment only so ``eval_semantic_ids`` can *measure* hierarchical consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rimworld_agent.utils import cfg_get, ensure_veq_importable, get_logger, to_veq_cfg, write_json

log = get_logger("assign_ids")


def _vocab_cls():
    ensure_veq_importable()
    from src.semantic_ids.assign_ids import SemanticIDVocab  # type: ignore

    return SemanticIDVocab


@dataclass
class SIDPipelineResult:
    entities: list
    embeddings: np.ndarray
    rqvae: Any
    sid_vocab: Any
    assignments: dict[str, dict]
    graph: Any


def assign_ids(entities: list, embeddings: np.ndarray, rqvae, vocab, graph=None) -> dict[str, dict]:
    """Map each ``def_name`` -> {codes, sid_sequence, sid_inline, category, ...}."""
    import torch

    codes = rqvae.encode_to_ids(torch.from_numpy(np.asarray(embeddings, np.float32)))
    codes = codes.cpu().numpy().tolist()
    out: dict[str, dict] = {}
    for entity, code_seq in zip(entities, codes):
        out[entity.def_name] = {
            "def_name": entity.def_name,
            "def_type": entity.def_type,
            "label": entity.label,
            "parent_def": entity.parent_def,
            "category": graph.category_of(entity.def_name) if graph else None,
            "subcategory": graph.subcategory_of(entity.def_name) if graph else None,
            "codes": code_seq,
            "sid_sequence": vocab.format_sequence(code_seq),
            "sid_inline": vocab.format_inline(code_seq),
        }
    log.info("assigned semantic IDs to %d entities", len(out))
    return out


def run_pipeline(cfg) -> SIDPipelineResult:
    """extract Defs -> build graph -> embed -> train RQ-VAE -> assign SIDs."""
    ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None))
    from src.semantic_ids.rqvae import train_rqvae  # type: ignore

    from rimworld_agent.knowledge.extract_defs import DEFAULT_DEF_TYPES, extract_defs
    from rimworld_agent.semantic_ids.build_entity_graph import build_entity_graph
    from rimworld_agent.semantic_ids.embed_entities import embed_entities

    entities = extract_defs(
        cfg_get(cfg, "paths.xml_defs_dir", "data/rimworld_xml_defs"),
        cfg_get(cfg, "knowledge.def_types", DEFAULT_DEF_TYPES),
    )
    if not entities:
        raise RuntimeError("No entities extracted; check paths.xml_defs_dir.")

    csharp_entities: list = []
    if cfg_get(cfg, "semantic_ids.use_csharp_view", True):
        try:
            from rimworld_agent.knowledge.extract_csharp import extract_csharp, link_defs_to_classes

            csharp_entities = extract_csharp(cfg_get(cfg, "paths.csharp_source_dir", "data/rimworld_source"))
            link_defs_to_classes(csharp_entities, entities)
        except Exception as exc:
            log.warning("no C# view: %s", exc)

    graph = build_entity_graph(entities)
    embeddings, _ = embed_entities(entities, cfg, csharp_entities)
    rqvae = train_rqvae(embeddings, to_veq_cfg(cfg))
    vocab = _vocab_cls().from_cfg(to_veq_cfg(cfg))
    assignments = assign_ids(entities, embeddings, rqvae, vocab, graph)
    return SIDPipelineResult(entities, embeddings, rqvae, vocab, assignments, graph)


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None))
        from src.semantic_ids.rqvae import save_rqvae  # type: ignore

        result = run_pipeline(cfg)
        save_rqvae(result.rqvae, cfg_get(cfg, "paths.rqvae_ckpt", "data/models/rqvae.pt"))
        out = cfg_get(cfg, "paths.sid_assignments_file", "results/sid_assignments.json")
        write_json(
            {
                "levels": result.sid_vocab.levels,
                "codebook_size": result.sid_vocab.codebook_size,
                "num_entities": len(result.assignments),
                "codebook_usage": result.rqvae.codebook_usage(),
                "assignments": result.assignments,
            },
            out,
        )
        log.info("wrote %d SID assignments -> %s", len(result.assignments), out)

    _run()


if __name__ == "__main__":
    main()
