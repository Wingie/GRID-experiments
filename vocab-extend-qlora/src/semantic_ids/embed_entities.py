"""Generate a dense embedding for each code entity.

Default encoder is a pretrained code model (``Salesforce/codet5p-110m-embedding``),
which is *separate* from the target SLM: it runs once to produce embeddings and is
then discarded. Each entity is embedded from a weighted combination of views
(signature / docstring / body / file-path) so both semantics and structure inform the
resulting vector.

When the encoder cannot be downloaded (offline / no GPU) and
``semantic_ids.allow_offline_encoder_fallback`` is set, a deterministic feature-hashing
encoder is used instead so the rest of the pipeline (and the tests) still run.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from src.semantic_ids.extract_entities import CodeEntity
from src.utils import cfg_get, get_logger

log = get_logger("embed_entities")


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
class HashingEncoder:
    """Deterministic offline encoder: token-trigram feature hashing -> fixed-dim,
    L2-normalised vectors. Not semantically meaningful at the level of a real code
    model, but stable and dependency-free for exercising the downstream RQ-VAE.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _hash(self, token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = text.split()
            for j in range(len(toks)):
                tri = " ".join(toks[j : j + 3])
                idx = self._hash(tri)
                out[i, idx] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-8, None)


class TransformerEncoder:
    """Wraps a HuggingFace code-embedding model with mean pooling."""

    def __init__(self, model_id: str, max_length: int = 512, batch_size: int = 32):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size

    def _pool(self, out, mask):
        # codet5p-embedding returns a pooled vector directly; otherwise mean-pool.
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        last = out.last_hidden_state
        mask = mask.unsqueeze(-1).type_as(last)
        return (last * mask).sum(1) / mask.sum(1).clamp(min=1e-8)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        vecs = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                enc = self.tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                out = self.model(**enc)
                pooled = out if torch.is_tensor(out) else self._pool(out, enc["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                vecs.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(vecs, axis=0)


def get_encoder(cfg: dict):
    """Return ``(encoder, is_real)`` honouring the offline fallback setting."""
    model_id = cfg_get(cfg, "semantic_ids.encoder_model")
    dim = cfg_get(cfg, "semantic_ids.embedding_dim", 256)
    try:
        enc = TransformerEncoder(
            model_id,
            max_length=cfg_get(cfg, "semantic_ids.encoder_max_length", 512),
            batch_size=cfg_get(cfg, "semantic_ids.encoder_batch_size", 32),
        )
        return enc, True
    except Exception as exc:
        if not cfg_get(cfg, "semantic_ids.allow_offline_encoder_fallback", True):
            raise
        log.warning("Encoder %s unavailable (%s); using HashingEncoder.", model_id, exc)
        return HashingEncoder(dim=dim), False


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def entity_views(entity: CodeEntity) -> dict[str, str]:
    return {
        "signature": entity.signature or entity.name,
        "docstring": entity.docstring or "",
        "body": entity.body or "",
        "path": entity.file_path or "",
    }


def embed_entities(entities: list[CodeEntity], cfg: dict) -> tuple[np.ndarray, bool]:
    """Return an ``[N, D]`` matrix of L2-normalised entity embeddings, row-aligned to
    ``entities``, plus whether a real encoder was used.
    """
    encoder, is_real = get_encoder(cfg)
    weights: dict[str, float] = dict(cfg_get(cfg, "semantic_ids.view_weights", {}))
    if not weights:
        weights = {"signature": 0.4, "docstring": 0.2, "body": 0.3, "path": 0.1}

    # Encode every (entity, view-with-weight>0) pair, then weighted-average per entity.
    active_views = [v for v, w in weights.items() if w > 0]
    flat_texts: list[str] = []
    for entity in entities:
        views = entity_views(entity)
        for v in active_views:
            flat_texts.append(views[v] or entity.name)

    if not flat_texts:
        return np.zeros((0, cfg_get(cfg, "semantic_ids.embedding_dim", 256)), np.float32), is_real

    flat_emb = encoder.encode(flat_texts)
    dim = flat_emb.shape[1]
    n_views = len(active_views)
    flat_emb = flat_emb.reshape(len(entities), n_views, dim)
    w = np.array([weights[v] for v in active_views], dtype=np.float32)
    w = w / w.sum()
    combined = (flat_emb * w[None, :, None]).sum(axis=1)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / np.clip(norms, 1e-8, None)
    log.info("Embedded %d entities -> dim %d (real_encoder=%s)", len(entities), dim, is_real)
    return combined.astype(np.float32), is_real


def main() -> None:
    import argparse
    from pathlib import Path

    from src.semantic_ids.extract_entities import extract_entities
    from src.utils import ensure_dir, iter_repo_files, load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    files = iter_repo_files(cfg_get(cfg, "paths.target_repo"), cfg_get(cfg, "languages", ["py"]))
    entities = extract_entities(files)
    emb, _ = embed_entities(entities, cfg)
    out_dir = ensure_dir(cfg_get(cfg, "paths.data_dir", "data"))
    np.save(Path(out_dir) / "entity_embeddings.npy", emb)
    log.info("Saved embeddings %s -> %s", emb.shape, out_dir)


if __name__ == "__main__":
    main()
