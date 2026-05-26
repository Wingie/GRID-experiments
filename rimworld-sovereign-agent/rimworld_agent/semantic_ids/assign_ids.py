"""Assign dual semantic IDs to RimWorld entities: a READ SID (RSID, taxonomy) for every
entity and — when bootstrap gameplay exists — a WRITE SID (WSID, workflow) for entities the
agent has acted on (project spec §3b).

Pipeline: extract Defs -> build entity graph -> READ embeddings -> READ RQ-VAE -> RSIDs.
If recorded episodes exist: mine co-occurrence -> WRITE embeddings -> WRITE RQ-VAE -> WSIDs.

Both codebooks share the `ResidualVQVAE` architecture (reused from vocab-extend-qlora); they
differ only in the embeddings they quantise. RSIDs/WSIDs use parallel token families
(`<RSID_L*_*>` / `<WSID_L*_*>`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rimworld_agent.utils import cfg_get, ensure_veq_importable, get_logger, write_json

log = get_logger("assign_ids")


@dataclass
class DualSIDResult:
    entities: list
    graph: Any
    # READ
    read_embeddings: np.ndarray
    read_rqvae: Any
    read_vocab: Any
    # WRITE (optional until gameplay exists)
    write_rqvae: Any = None
    write_vocab: Any = None
    cooccurrence: Any = None
    # combined per-entity assignments
    assignments: dict[str, dict] = None


def _codes_for(rqvae, embeddings: np.ndarray) -> list[list[int]]:
    import torch

    codes = rqvae.encode_to_ids(torch.from_numpy(np.asarray(embeddings, np.float32)))
    return codes.cpu().numpy().tolist()


def run_pipeline(cfg, with_write: bool | None = None) -> DualSIDResult:
    ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None))

    from rimworld_agent.knowledge.extract_defs import DEFAULT_DEF_TYPES, extract_defs
    from rimworld_agent.semantic_ids.build_entity_graph import build_entity_graph
    from rimworld_agent.semantic_ids.rqvae_read import build_read_embeddings, read_vocab, train_read_rqvae

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

    # --- READ (always) -------------------------------------------------------
    read_emb = build_read_embeddings(entities, cfg, csharp_entities, graph)
    read_rqvae = train_read_rqvae(read_emb, cfg)
    rvocab = read_vocab(cfg)
    read_codes = _codes_for(read_rqvae, read_emb)

    assignments: dict[str, dict] = {}
    for entity, rcodes in zip(entities, read_codes):
        assignments[entity.def_name] = {
            "def_name": entity.def_name,
            "def_type": entity.def_type,
            "label": entity.label,
            "parent_def": entity.parent_def,
            "category": graph.category_of(entity.def_name),
            "subcategory": graph.subcategory_of(entity.def_name),
            "read_codes": rcodes,
            "rsid_sequence": rvocab.format_sequence(rcodes),
            "rsid_inline": rvocab.format_inline(rcodes),
            "write_codes": None,
            "wsid_sequence": None,
            "wsid_inline": None,
        }

    # --- WRITE (only if gameplay exists) ------------------------------------
    write_rqvae = wvocab = cooc = None
    want_write = cfg_get(cfg, "semantic_ids.write.enabled", True) if with_write is None else with_write
    if want_write:
        from rimworld_agent.semantic_ids.collect_cooccurrence import collect_cooccurrence
        from rimworld_agent.semantic_ids.rqvae_write import build_write_embeddings, train_write_rqvae, write_vocab

        cooc = collect_cooccurrence(
            cfg_get(cfg, "paths.episodes_dir", "data/episodes"),
            window=cfg_get(cfg, "semantic_ids.write.cooccurrence_window", 5),
        )
        if cooc.def_names:
            dim = cfg_get(cfg, "semantic_ids.write.rqvae.latent_dim", 256)
            write_emb = build_write_embeddings(cooc, dim, cfg_get(cfg, "seed", 42))
            write_rqvae = train_write_rqvae(write_emb, cfg)
            wvocab = write_vocab(cfg)
            write_codes = _codes_for(write_rqvae, write_emb)
            for def_name, wcodes in zip(cooc.def_names, write_codes):
                if def_name in assignments:
                    assignments[def_name]["write_codes"] = wcodes
                    assignments[def_name]["wsid_sequence"] = wvocab.format_sequence(wcodes)
                    assignments[def_name]["wsid_inline"] = wvocab.format_inline(wcodes)
            log.info("assigned WSIDs to %d acted-on entities", len(cooc.def_names))
        else:
            log.warning("no gameplay episodes yet -> WRITE codebook skipped (RSIDs only).")

    return DualSIDResult(
        entities=entities,
        graph=graph,
        read_embeddings=read_emb,
        read_rqvae=read_rqvae,
        read_vocab=rvocab,
        write_rqvae=write_rqvae,
        write_vocab=wvocab,
        cooccurrence=cooc,
        assignments=assignments,
    )


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None))
        from src.semantic_ids.rqvae import save_rqvae  # type: ignore

        result = run_pipeline(cfg)
        save_rqvae(result.read_rqvae, cfg_get(cfg, "paths.read_rqvae_ckpt", "data/models/rqvae_read.pt"))
        if result.write_rqvae is not None:
            save_rqvae(result.write_rqvae, cfg_get(cfg, "paths.write_rqvae_ckpt", "data/models/rqvae_write.pt"))
        n_wsid = sum(1 for a in result.assignments.values() if a["write_codes"] is not None)
        write_json(
            {
                "read": {"levels": result.read_vocab.levels, "codebook_size": result.read_vocab.codebook_size},
                "write": (
                    {"levels": result.write_vocab.levels, "codebook_size": result.write_vocab.codebook_size}
                    if result.write_vocab else None
                ),
                "num_entities": len(result.assignments),
                "num_with_wsid": n_wsid,
                "read_codebook_usage": result.read_rqvae.codebook_usage(),
                "write_codebook_usage": result.write_rqvae.codebook_usage() if result.write_rqvae else None,
                "assignments": result.assignments,
            },
            cfg_get(cfg, "paths.sid_assignments_file", "results/sid_assignments.json"),
        )
        log.info("wrote dual SID assignments (%d entities, %d with WSID)", len(result.assignments), n_wsid)

    _run()


if __name__ == "__main__":
    main()
