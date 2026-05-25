"""Semantic-ID layer: extract code entities, embed them, learn hierarchical discrete
IDs with an RQ-VAE, and inject those IDs into training data.

This applies the TIGER (Rajput et al., 2023) generative-retrieval paradigm to code:
a class/function/method is represented by a short sequence of discrete codes
(coarse module cluster -> finer class cluster -> method role) rather than an
arbitrary name that fragments into subword tokens.
"""

from src.semantic_ids.extract_entities import CodeEntity, extract_entities

__all__ = ["CodeEntity", "extract_entities"]
