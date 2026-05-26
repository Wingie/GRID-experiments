"""WRITE RQ-VAE — "what is this entity USED WITH?" (workflow / co-occurrence similarity).

WRITE semantic IDs (WSIDs) cluster entities that appear in the same action sequences:
SolarGenerator and Battery share a WSID prefix because they are built together. We turn the
co-occurrence matrix (`collect_cooccurrence`) into workflow embeddings via PPMI + truncated
SVD, then quantise with the *shared* `ResidualVQVAE`. Entities never acted on in the bootstrap
episodes get a near-zero embedding and fall into a default code (they have no workflow signal
yet — re-run after more self-play, per gotcha #11).

The WRITE codebook can only be trained AFTER bootstrap gameplay exists.
"""

from __future__ import annotations

import numpy as np

from rimworld_agent.semantic_ids.collect_cooccurrence import CooccurrenceData
from rimworld_agent.semantic_ids.sid_vocab import SIDVocab, write_vocab_from_cfg
from rimworld_agent.utils import cfg_get, ensure_veq_importable, get_logger, to_veq_cfg

log = get_logger("rqvae_write")


def _veq_cfg_for_write(cfg, latent_dim: int) -> dict:
    veq = to_veq_cfg(cfg)
    write_rqvae = dict(cfg_get(cfg, "semantic_ids.write.rqvae", {}) or {})
    veq.setdefault("semantic_ids", {})["rqvae"] = write_rqvae
    veq["semantic_ids"]["embedding_dim"] = latent_dim
    return veq


def build_write_embeddings(cooc: CooccurrenceData, dim: int = 256, seed: int = 42) -> np.ndarray:
    """PPMI + truncated SVD of the co-occurrence matrix -> ``[N, dim]`` workflow embeddings."""
    ppmi = cooc.ppmi()
    n = ppmi.shape[0]
    if n == 0:
        return np.zeros((0, dim), np.float32)
    k = min(dim, max(1, n - 1))
    # Truncated SVD via numpy (entity counts are small: hundreds-to-thousands).
    try:
        u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
        emb = u[:, :k] * s[:k]
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(seed)
        emb = ppmi @ rng.standard_normal((n, k)).astype(np.float32)
    if emb.shape[1] < dim:  # pad to requested dim so the RQ-VAE input is fixed-size
        emb = np.pad(emb, ((0, 0), (0, dim - emb.shape[1])))
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / np.clip(norms, 1e-8, None)).astype(np.float32)


def train_write_rqvae(embeddings: np.ndarray, cfg):
    ensure_veq_importable(cfg_get(cfg, "paths.veq_path", None))
    from src.semantic_ids.rqvae import train_rqvae  # type: ignore

    log.info("training WRITE RQ-VAE on %s workflow embeddings", embeddings.shape)
    return train_rqvae(embeddings, _veq_cfg_for_write(cfg, embeddings.shape[1]))


def write_vocab(cfg) -> SIDVocab:
    return write_vocab_from_cfg(cfg)
