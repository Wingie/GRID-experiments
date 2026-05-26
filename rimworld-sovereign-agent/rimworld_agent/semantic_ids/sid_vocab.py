"""Prefixed semantic-ID vocabulary, parameterised by prefix (``RSID`` vs ``WSID``).

vocab-extend-qlora's ``SemanticIDVocab`` hard-codes the ``<SID_*>`` prefix; the dual-codebook
design needs two parallel token families — READ (``<RSID_L{l}_{c}>``) and WRITE
(``<WSID_L{l}_{c}>``) — so we keep a thin local vocab here. The RQ-VAE *quantiser* itself is
still the reused vocab-extend-qlora ``ResidualVQVAE``; only the token grammar lives here.

For ``L=3, K=64`` each family contributes ``3*64=192`` per-level tokens + structural tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


def code_width(codebook_size: int) -> int:
    return max(2, len(str(codebook_size - 1)))


@dataclass
class SIDVocab:
    prefix: str  # "RSID" (read/taxonomy) or "WSID" (write/workflow)
    levels: int
    codebook_size: int

    @property
    def width(self) -> int:
        return code_width(self.codebook_size)

    def token(self, level_1indexed: int, code: int) -> str:
        return f"<{self.prefix}_L{level_1indexed}_{code:0{self.width}d}>"

    @property
    def start(self) -> str:
        return f"<{self.prefix}_START>"

    @property
    def end(self) -> str:
        return f"<{self.prefix}_END>"

    @property
    def sep(self) -> str:
        return f"<{self.prefix}_SEP>"

    def structural_tokens(self) -> list[str]:
        return [self.start, self.end, self.sep]

    def per_level_tokens(self) -> list[str]:
        """All code tokens ordered by (level, code) so index = level*K + code."""
        return [
            self.token(level + 1, code)
            for level in range(self.levels)
            for code in range(self.codebook_size)
        ]

    def special_tokens(self) -> list[str]:
        return self.per_level_tokens() + self.structural_tokens()

    def token_to_level_code(self, tok: str) -> tuple[int, int] | None:
        head = f"<{self.prefix}_L"
        if not (tok.startswith(head) and tok.endswith(">")):
            return None
        try:
            body = tok[len(head) : -1]
            lvl_str, code_str = body.split("_", 1)
            return int(lvl_str) - 1, int(code_str)
        except (ValueError, IndexError):
            return None

    def format_sequence(self, codes: list[int], structural: bool = True) -> str:
        toks = [self.token(i + 1, c) for i, c in enumerate(codes)]
        if not structural:
            return "".join(toks)
        out = [self.start]
        for i, t in enumerate(toks):
            out.append(t)
            if i < len(toks) - 1:
                out.append(self.sep)
        out.append(self.end)
        return "".join(out)

    def format_inline(self, codes: list[int]) -> str:
        return self.format_sequence(codes, structural=False)


# Task tokens are shared across both families (not prefixed) — they mark the training
# objective, not a code family.
TASK_TOKENS = ["<PREDICT_SID>", "<SID_TO_SIG>", "<FIM_PREFIX>", "<FIM_MIDDLE>", "<FIM_SUFFIX>"]


def read_vocab_from_cfg(cfg) -> SIDVocab:
    from rimworld_agent.utils import cfg_get

    return SIDVocab(
        prefix="RSID",
        levels=cfg_get(cfg, "semantic_ids.read.rqvae.levels", cfg_get(cfg, "semantic_ids.rqvae.levels", 3)),
        codebook_size=cfg_get(cfg, "semantic_ids.read.rqvae.codebook_size", cfg_get(cfg, "semantic_ids.rqvae.codebook_size", 64)),
    )


def write_vocab_from_cfg(cfg) -> SIDVocab:
    from rimworld_agent.utils import cfg_get

    return SIDVocab(
        prefix="WSID",
        levels=cfg_get(cfg, "semantic_ids.write.rqvae.levels", 3),
        codebook_size=cfg_get(cfg, "semantic_ids.write.rqvae.codebook_size", 64),
    )
