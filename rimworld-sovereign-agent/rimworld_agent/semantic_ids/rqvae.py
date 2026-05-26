"""Shared RQ-VAE architecture for semantic IDs — reused from vocab-extend-qlora
(``src.semantic_ids.rqvae``).

This is the **shared base** for both codebooks in the dual-RQ-VAE design (project spec §3b,
gotcha #12): the READ RQ-VAE (`rqvae_read`) and WRITE RQ-VAE (`rqvae_write`) use the *same*
TIGER-style residual quantiser (EMA codebooks + dead-code reinit, ``L=3 × K=64``) but train
on different embeddings — structural similarity for READ, workflow co-occurrence for WRITE.
We re-export the quantiser so both training scripts share one implementation.
"""

from __future__ import annotations

from rimworld_agent.utils import ensure_veq_importable

if not ensure_veq_importable():
    raise ImportError(
        "vocab-extend-qlora is not importable; install it with "
        "`pip install -e ../vocab-extend-qlora` or set $VEQ_PATH."
    )

from src.semantic_ids.rqvae import (  # type: ignore  # noqa: E402,F401
    EMAVectorQuantizer,
    ResidualVQVAE,
    RQVAEConfig,
    load_rqvae,
    save_rqvae,
    train_rqvae,
)

__all__ = [
    "EMAVectorQuantizer",
    "ResidualVQVAE",
    "RQVAEConfig",
    "load_rqvae",
    "save_rqvae",
    "train_rqvae",
]
