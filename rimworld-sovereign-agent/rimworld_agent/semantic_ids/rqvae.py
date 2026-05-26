"""Hierarchical RQ-VAE for game-entity semantic IDs — reused verbatim from
vocab-extend-qlora (``src.semantic_ids.rqvae``).

The RimWorld project deliberately does not reimplement the quantiser: the TIGER-style
residual VQ-VAE with EMA codebooks + dead-code reinitialisation already lives in
vocab-extend-qlora and is unit-tested there. We re-export the public symbols so the rest
of this package can ``from rimworld_agent.semantic_ids.rqvae import ResidualVQVAE`` without
caring where the implementation physically lives.

Config mapping (project spec §3b): ``levels=3`` (Category -> SubCategory -> Item),
``codebook_size=64`` (=> 3*64=192 per-level SID tokens), ``latent_dim=256``.
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
