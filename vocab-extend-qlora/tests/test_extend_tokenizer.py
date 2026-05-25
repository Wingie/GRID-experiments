"""CPU-only tests for the embedding-initialisation primitives used by the tokenizer
extension. These operate on plain tensors so they need no model download.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from src.extend_tokenizer import (  # noqa: E402
    codebook_projected_vectors,
    exp_weighted_init_vectors,
    mean_init_vectors,
)


def test_mean_init_is_subtoken_average():
    torch = pytest.importorskip("torch")

    old_emb = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    # token A -> subtokens [0, 2]; token B -> subtoken [4-index? use 4 rows] -> [1]
    vecs = mean_init_vectors(old_emb, [[0, 2], [1]])
    assert torch.allclose(vecs[0], (old_emb[0] + old_emb[2]) / 2)
    assert torch.allclose(vecs[1], old_emb[1])


def test_mean_init_empty_falls_back_to_global_mean():
    torch = pytest.importorskip("torch")

    old_emb = torch.randn(6, 8)
    vecs = mean_init_vectors(old_emb, [[]])
    assert torch.allclose(vecs[0], old_emb.mean(dim=0))


def test_exp_weighted_prioritises_earlier_subtokens():
    torch = pytest.importorskip("torch")

    old_emb = torch.zeros(3, 2)
    old_emb[0] = torch.tensor([1.0, 0.0])
    old_emb[1] = torch.tensor([0.0, 1.0])
    vecs = exp_weighted_init_vectors(old_emb, [[0, 1]], decay=0.5)
    # weights normalise to [2/3, 1/3] so the first subtoken dominates
    assert vecs[0][0] > vecs[0][1]
    assert torch.allclose(vecs[0].sum(), torch.tensor(1.0), atol=1e-5)


def test_codebook_projection_shape_and_norm():
    torch = pytest.importorskip("torch")

    L, K, latent, embed_dim = 3, 16, 24, 32
    codebook = torch.randn(L, K, latent)
    target_norm = 0.7
    out = codebook_projected_vectors(codebook, embed_dim, target_norm)
    assert out.shape == (L * K, embed_dim)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.full_like(norms, target_norm), atol=1e-4)


def test_codebook_projection_is_deterministic():
    torch = pytest.importorskip("torch")

    codebook = torch.randn(2, 8, 16)
    a = codebook_projected_vectors(codebook, 12, 1.0, seed=123)
    b = codebook_projected_vectors(codebook, 12, 1.0, seed=123)
    assert torch.allclose(a, b)
