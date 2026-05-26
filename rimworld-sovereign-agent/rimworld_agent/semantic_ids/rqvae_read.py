"""READ RQ-VAE — "what IS this entity?" (taxonomy / structural similarity).

READ semantic IDs (RSIDs) cluster entities that are the same *kind* of thing: SolarGenerator
and WindTurbine share an RSID prefix because both are power buildings. We build structural
embeddings from the entity's def/label/C#/wiki views (`embed_entities`) and optionally sharpen
them with a supervised, category-centroid pull (a lightweight stand-in for the contrastive
objective in spec §3b — entities in the same entity-graph category are drawn together before
quantisation). The shared `ResidualVQVAE` then produces the L-tuple codes.

Offline-verifiable (uses vocab-extend-qlora's HashingEncoder fallback when CodeT5+ can't be
downloaded).
"""

from __future__ import annotations

import numpy as np

from rimworld_agent.semantic_ids.sid_vocab import SIDVocab, read_vocab_from_cfg
from rimworld_agent.utils import cfg_get, ensure_veq_importable, get_logger, to_veq_cfg

log = get_logger("rqvae_read")


def _veq_cfg_for_read(cfg) -> dict:
    """A vocab-extend-qlora cfg dict whose ``semantic_ids.rqvae`` is the READ config."""
    veq = to_veq_cfg(cfg)
    read_rqvae = cfg_get(cfg, "semantic_ids.read.rqvae", None)
    if read_rqvae is not None:
        veq.setdefault("semantic_ids", {})["rqvae"] = dict(read_rqvae)
    return veq


def contrastive_sharpen(embeddings: np.ndarray, categories: list[str], strength: float) -> np.ndarray:
    """Pull each embedding toward its category centroid: ``e' = (1-s)*e + s*mean_category``.

    A cheap supervised-contrastive surrogate that emphasises taxonomy before quantisation.
    """
    if strength <= 0:
        return embeddings
    out = embeddings.copy()
    cats = np.array(categories)
    for c in set(categories):
        mask = cats == c
        centroid = embeddings[mask].mean(axis=0, keepdims=True)
        out[mask] = (1 - strength) * embeddings[mask] + strength * centroid
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.clip(norms, 1e-8, None)).astype(np.float32)


def build_read_embeddings(entities: list, cfg, csharp_entities=None, graph=None) -> np.ndarray:
    """Structural embeddings for the READ codebook, optionally category-sharpened."""
    from rimworld_agent.semantic_ids.embed_entities import embed_entities

    embeddings, _ = embed_entities(entities, cfg, csharp_entities)
    strength = cfg_get(cfg, "semantic_ids.read.contrastive_strength", 0.3)
    if graph is not None and strength > 0:
        cats = [f"{graph.category_of(e.def_name)}/{graph.subcategory_of(e.def_name)}" for e in entities]
        embeddings = contrastive_sharpen(embeddings, cats, strength)
    return embeddings


def train_read_rqvae(embeddings: np.ndarray, cfg):
    ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None))
    from src.semantic_ids.rqvae import train_rqvae  # type: ignore

    log.info("training READ RQ-VAE on %s structural embeddings", embeddings.shape)
    return train_rqvae(embeddings, _veq_cfg_for_read(cfg))


def read_vocab(cfg) -> SIDVocab:
    return read_vocab_from_cfg(cfg)
