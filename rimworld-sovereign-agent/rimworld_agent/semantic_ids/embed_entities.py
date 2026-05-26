"""Dense embeddings for RimWorld entities, built from multiple views.

Each entity is embedded from a weighted combination of:
  * ``def``    — the resolved XML Def text (label + description + fields);
  * ``label``  — the short label + description (the natural-language view);
  * ``csharp`` — the linked C# class blob (engine semantics), when available;
  * ``wiki``   — the matching wiki page, when available.

We reuse vocab-extend-qlora's encoder loader (``src.semantic_ids.embed_entities.get_encoder``,
default ``Salesforce/codet5p-110m-embedding`` per spec §3b, with a deterministic offline
HashingEncoder fallback) and only swap in game-specific views.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np

from rimworld_agent.utils import cfg_get, ensure_veq_importable, get_logger, to_veq_cfg

log = get_logger("embed_entities")

DEFAULT_VIEW_WEIGHTS = {"def": 0.5, "label": 0.2, "csharp": 0.2, "wiki": 0.1}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def build_csharp_index(csharp_entities: Iterable) -> dict[str, str]:
    """Map class name -> its text blob for the ``csharp`` view lookup."""
    return {e.name: e.text_blob() for e in csharp_entities}


def build_wiki_index(wiki_dir: str | Path | None) -> dict[str, str]:
    """Map a slugified wiki page name -> its markdown text."""
    index: dict[str, str] = {}
    if not wiki_dir:
        return index
    for md in Path(wiki_dir).glob("*.md"):
        index[_slug(md.stem)] = md.read_text(encoding="utf8", errors="replace")
    return index


def entity_views(entity, csharp_index: dict[str, str], wiki_index: dict[str, str]) -> dict[str, str]:
    fields = getattr(entity, "fields", {}) or {}
    # Find a linked C# class via common Class-bearing fields.
    csharp_text = ""
    for key in ("thingClass", "compClass", "workerClass", "Class"):
        val = fields.get(key)
        if isinstance(val, str):
            short = val.rsplit(".", 1)[-1]
            if short in csharp_index:
                csharp_text = csharp_index[short]
                break
    wiki_text = wiki_index.get(_slug(entity.label)) or wiki_index.get(_slug(entity.def_name)) or ""
    return {
        "def": entity.text_blob(),
        "label": f"{entity.label}. {entity.description}".strip(),
        "csharp": csharp_text,
        "wiki": wiki_text[:4000],
    }


def embed_entities(
    entities: list,
    cfg,
    csharp_entities: Iterable | None = None,
) -> tuple[np.ndarray, bool]:
    """Return an ``[N, D]`` L2-normalised embedding matrix row-aligned to ``entities``."""
    ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None))
    from src.semantic_ids.embed_entities import get_encoder  # type: ignore

    veq = to_veq_cfg(cfg)
    encoder, is_real = get_encoder(veq)
    if not is_real:
        log.warning("using HashingEncoder; SID clusters are a sanity-check, not final.")

    weights = dict(cfg_get(cfg, "semantic_ids.view_weights", DEFAULT_VIEW_WEIGHTS))
    active = [v for v, w in weights.items() if w > 0]

    csharp_index = build_csharp_index(csharp_entities or [])
    wiki_index = build_wiki_index(cfg_get(cfg, "paths.wiki_dir", "data/rimworld_wiki"))

    flat_texts: list[str] = []
    for e in entities:
        views = entity_views(e, csharp_index, wiki_index)
        for v in active:
            flat_texts.append(views[v] or e.label or e.def_name)

    dim = cfg_get(cfg, "semantic_ids.embedding_dim", 256)
    if not flat_texts:
        return np.zeros((0, dim), np.float32), is_real

    flat_emb = encoder.encode(flat_texts)
    dim = flat_emb.shape[1]
    flat_emb = flat_emb.reshape(len(entities), len(active), dim)
    w = np.array([weights[v] for v in active], dtype=np.float32)
    w = w / w.sum()
    combined = (flat_emb * w[None, :, None]).sum(axis=1)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / np.clip(norms, 1e-8, None)
    log.info("embedded %d entities -> dim %d (real_encoder=%s)", len(entities), dim, is_real)
    return combined.astype(np.float32), is_real
