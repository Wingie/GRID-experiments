"""Assign hierarchical semantic IDs to code entities and define the SID token vocabulary.

After the RQ-VAE is trained, every entity is mapped to an ``L``-tuple of codes
``[c1, ..., cL]`` and formatted into special tokens. We add the *per-level* code tokens
(``<SID_L1_00>`` ... ``<SID_L{L}_{K-1}>``), the structural tokens
(``<SID_START>``, ``<SID_END>``, ``<SID_SEP>``), and the task tokens to the tokenizer.

Design choice: each entity is encoded independently (TIGER-style), so a method's codes
are *not* forced to share its class's prefix. Whether they do is an emergent property
measured by ``eval_semantic_ids.py`` (hierarchical consistency), not an enforced one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.semantic_ids.extract_entities import CodeEntity
from src.utils import cfg_get, get_logger

log = get_logger("assign_ids")


def code_width(codebook_size: int) -> int:
    return max(2, len(str(codebook_size - 1)))


def sid_token(level_1indexed: int, code: int, width: int) -> str:
    return f"<SID_L{level_1indexed}_{code:0{width}d}>"


@dataclass
class SemanticIDVocab:
    """The full set of semantic-ID tokens and helpers to format/parse them."""

    levels: int
    codebook_size: int
    structural_tokens: tuple[str, ...] = ("<SID_START>", "<SID_END>", "<SID_SEP>")
    task_tokens: tuple[str, ...] = (
        "<PREDICT_SID>",
        "<SID_TO_SIG>",
        "<FIM_PREFIX>",
        "<FIM_MIDDLE>",
        "<FIM_SUFFIX>",
    )

    @property
    def width(self) -> int:
        return code_width(self.codebook_size)

    def per_level_tokens(self) -> list[str]:
        """All code tokens, ordered by (level, code) so index = level*K + code."""
        return [
            sid_token(level + 1, code, self.width)
            for level in range(self.levels)
            for code in range(self.codebook_size)
        ]

    def special_tokens(self) -> list[str]:
        """Every SID-related special token to register with the tokenizer."""
        return self.per_level_tokens() + list(self.structural_tokens) + list(self.task_tokens)

    def token_to_level_code(self, token: str) -> tuple[int, int] | None:
        """Parse ``<SID_L{l}_{c}>`` -> (level0, code), else None."""
        if not (token.startswith("<SID_L") and token.endswith(">")):
            return None
        try:
            body = token[len("<SID_L") : -1]
            lvl_str, code_str = body.split("_", 1)
            return int(lvl_str) - 1, int(code_str)
        except (ValueError, IndexError):
            return None

    def format_sequence(self, codes: list[int], structural: bool = True) -> str:
        """Render a code tuple as ``<SID_START><SID_L1_..><SID_SEP>...<SID_END>``."""
        toks = [sid_token(i + 1, c, self.width) for i, c in enumerate(codes)]
        if not structural:
            return "".join(toks)
        out = ["<SID_START>"]
        for i, t in enumerate(toks):
            out.append(t)
            if i < len(toks) - 1:
                out.append("<SID_SEP>")
        out.append("<SID_END>")
        return "".join(out)

    def format_inline(self, codes: list[int]) -> str:
        """Compact name-replacement form (no structural wrappers)."""
        return self.format_sequence(codes, structural=False)

    @classmethod
    def from_cfg(cls, cfg: dict) -> "SemanticIDVocab":
        return cls(
            levels=cfg_get(cfg, "semantic_ids.rqvae.levels", 3),
            codebook_size=cfg_get(cfg, "semantic_ids.rqvae.codebook_size", 64),
            structural_tokens=tuple(
                cfg_get(cfg, "semantic_ids.structural_tokens",
                        ["<SID_START>", "<SID_END>", "<SID_SEP>"])
            ),
            task_tokens=tuple(
                cfg_get(cfg, "semantic_ids.task_tokens",
                        ["<PREDICT_SID>", "<SID_TO_SIG>", "<FIM_PREFIX>", "<FIM_MIDDLE>", "<FIM_SUFFIX>"])
            ),
        )


def assign_ids(
    entities: list[CodeEntity], embeddings: np.ndarray, model, vocab: SemanticIDVocab
) -> dict[str, dict]:
    """Map each entity uid -> {codes, sid_sequence, sid_inline, name, kind, parent}."""
    import torch

    codes = model.encode_to_ids(torch.from_numpy(np.asarray(embeddings, np.float32)))
    codes = codes.cpu().numpy().tolist()
    assignments: dict[str, dict] = {}
    for entity, code_seq in zip(entities, codes):
        assignments[entity.uid] = {
            "name": entity.name,
            "kind": entity.kind,
            "parent": entity.parent,
            "file_path": entity.file_path,
            "codes": code_seq,
            "sid_sequence": vocab.format_sequence(code_seq),
            "sid_inline": vocab.format_inline(code_seq),
        }
    log.info("Assigned semantic IDs to %d entities", len(assignments))
    return assignments


def run_pipeline(cfg: dict):
    """Extract -> embed -> train RQ-VAE -> assign. Returns (entities, embeddings,
    model, vocab, assignments). Shared by inject_ids and extend_tokenizer.
    """
    from src.semantic_ids.embed_entities import embed_entities
    from src.semantic_ids.extract_entities import extract_entities
    from src.semantic_ids.rqvae import train_rqvae
    from src.utils import iter_repo_files

    files = list(iter_repo_files(cfg_get(cfg, "paths.target_repo"), cfg_get(cfg, "languages", ["py"])))
    entities = extract_entities(files)
    if not entities:
        raise RuntimeError("No code entities extracted; check paths.target_repo and languages.")
    embeddings, _ = embed_entities(entities, cfg)
    model = train_rqvae(embeddings, cfg)
    vocab = SemanticIDVocab.from_cfg(cfg)
    assignments = assign_ids(entities, embeddings, model, vocab)
    return entities, embeddings, model, vocab, assignments


def main() -> None:
    import argparse
    from pathlib import Path

    from src.semantic_ids.rqvae import save_rqvae
    from src.utils import load_config, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    _entities, _emb, model, vocab, assignments = run_pipeline(cfg)
    save_rqvae(model, cfg_get(cfg, "paths.rqvae_ckpt", "data/models/rqvae.pt"))
    out = Path(cfg_get(cfg, "paths.sid_assignments_file", "results/sid_assignments.json"))
    write_json(
        {
            "levels": vocab.levels,
            "codebook_size": vocab.codebook_size,
            "num_entities": len(assignments),
            "codebook_usage": model.codebook_usage(),
            "assignments": assignments,
        },
        out,
    )
    log.info("Wrote %d SID assignments -> %s", len(assignments), out)


if __name__ == "__main__":
    main()
