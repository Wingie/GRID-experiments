"""Phase 1: mine candidate tokens from the target repository.

Three strategies (configurable via ``mining.strategies``):

  A. identifiers  - regex-extract identifiers (class/function/variable/import names),
                    score = frequency x current sub-token count. A name that the base
                    tokenizer shatters into many pieces and that appears often is the
                    highest-value addition. (AdaptiVocab motivation.)
  B. ngrams       - frequent multi-token sequences (n=2,3) that recur in the tokenised
                    corpus, e.g. ``self.config.``, ``async def``, ``=> {``.
  C. gradient     - (optional, VEGAD-style) rank a candidate pool by the gradient
                    magnitude their grouped sub-tokens receive on a small forward pass.

Writes ``results/candidates.json``: one row per candidate with
``token, frequency, current_subtokens, score, strategy``.

Usage:
    python -m src.mine_tokens configs/experiments/vocab_qlora.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.utils import (
    cfg_get,
    count_subtokens,
    get_logger,
    iter_repo_files,
    load_config,
    load_hf_tokenizer,
    set_seed,
    write_json,
)

log = get_logger("mine_tokens")

# Identifiers across the supported languages. Includes dotted access and a few
# code-specific operators so multi-character glyphs survive.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def mine_identifiers(
    texts: Iterable[str], tokenizer, min_frequency: int
) -> dict[str, dict]:
    """Strategy A: frequent identifiers scored by frequency x sub-token count."""
    freq: Counter[str] = Counter()
    for text in texts:
        freq.update(_IDENT_RE.findall(text))

    candidates: dict[str, dict] = {}
    for token, count in freq.items():
        if count < min_frequency or len(token) < 3:
            continue
        sub = count_subtokens(tokenizer, token)
        if sub <= 1:
            continue  # already a single token; adding it saves nothing
        candidates[token] = {
            "token": token,
            "frequency": count,
            "current_subtokens": sub,
            "score": count * sub,
            "strategy": "identifiers",
        }
    return candidates


def mine_ngrams(
    texts: Iterable[str], tokenizer, sizes: list[int], min_frequency: int
) -> dict[str, dict]:
    """Strategy B: frequent contiguous sub-token n-grams (AdaptiVocab-style).

    We n-gram over the *tokeniser's* pieces (not raw chars) so the resulting merged
    tokens correspond to real decode-step savings.
    """
    ngram_freq: Counter[tuple[str, ...]] = Counter()
    for text in texts:
        pieces = tokenizer.tokenize(text)
        for n in sizes:
            for i in range(len(pieces) - n + 1):
                ngram_freq[tuple(pieces[i : i + n])] += 1

    candidates: dict[str, dict] = {}
    for pieces, count in ngram_freq.items():
        if count < min_frequency:
            continue
        # Reconstruct surface form; HF sub-tokens often carry a leading space marker.
        surface = "".join(p.replace("Ġ", " ").replace("▁", " ") for p in pieces)
        surface = surface.strip()
        if len(surface) < 3 or _IDENT_RE.fullmatch(surface):
            # pure identifiers are covered by strategy A; keep operators/phrases here
            continue
        n = len(pieces)
        candidates[surface] = {
            "token": surface,
            "frequency": count,
            "current_subtokens": n,
            "score": count * n,
            "strategy": "ngrams",
        }
    return candidates


def mine_gradient(
    texts: list[str], candidates: dict[str, dict], cfg: dict
) -> dict[str, dict]:
    """Strategy C (optional): re-rank a candidate pool by gradient signal.

    Runs a few forward/backward passes of the base model on repo samples and measures
    the gradient L2 norm flowing into the sub-token embeddings that make up each
    candidate. High signal => the model "cares" about that region of token space, so
    promoting it to a single token is more likely to help. Needs the model weights;
    silently degrades to a no-op (returns the input scores) if torch/model is absent.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        log.warning("gradient strategy unavailable (torch/transformers missing); skipping.")
        return candidates

    model_id = cfg_get(cfg, "model.id")
    num_samples = cfg_get(cfg, "mining.gradient.num_samples", 256)
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        model.eval()
    except Exception as exc:
        log.warning("gradient strategy: could not load %s (%s); skipping.", model_id, exc)
        return candidates

    embed = model.get_input_embeddings()
    grad_accum = torch.zeros(embed.num_embeddings)
    sample = texts[:num_samples]
    for text in sample:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids
        embed.weight.grad = None
        out = model(ids, labels=ids)
        out.loss.backward()
        if embed.weight.grad is not None:
            grad_accum.index_add_(
                0, ids.flatten(), embed.weight.grad[ids.flatten()].norm(dim=-1)
            )

    for cand in candidates.values():
        piece_ids = tok.encode(cand["token"], add_special_tokens=False)
        cand["gradient_signal"] = float(sum(grad_accum[i].item() for i in piece_ids))
        cand["score"] = cand["score"] * (1.0 + cand["gradient_signal"])
        cand["strategy"] = cand["strategy"] + "+gradient"
    return candidates


def mine(cfg: dict) -> list[dict]:
    set_seed(cfg_get(cfg, "seed", 42))
    repo = cfg_get(cfg, "paths.target_repo")
    languages = cfg_get(cfg, "languages", ["py"])
    files = list(iter_repo_files(repo, languages))
    log.info("Read %d source files from %s", len(files), repo)
    texts = [f.text for f in files]

    tokenizer, is_real = load_hf_tokenizer(cfg_get(cfg, "model.id"))
    if not is_real:
        log.warning("Using fallback tokenizer; sub-token counts are approximate.")

    strategies = cfg_get(cfg, "mining.strategies", ["identifiers", "ngrams"])
    candidates: dict[str, dict] = {}
    if "identifiers" in strategies:
        candidates.update(
            mine_identifiers(
                texts, tokenizer, cfg_get(cfg, "mining.identifier_min_frequency", 5)
            )
        )
    if "ngrams" in strategies:
        # n-gram surface forms may collide with identifiers; identifiers win.
        ngram_cands = mine_ngrams(
            texts,
            tokenizer,
            cfg_get(cfg, "mining.ngram_sizes", [2, 3]),
            cfg_get(cfg, "mining.ngram_min_frequency", 25),
        )
        for k, v in ngram_cands.items():
            candidates.setdefault(k, v)
    if "gradient" in strategies and cfg_get(cfg, "mining.gradient.enabled", False):
        candidates = mine_gradient(texts, candidates, cfg)

    ranked = sorted(candidates.values(), key=lambda c: c["score"], reverse=True)
    log.info("Mined %d unique candidate tokens", len(ranked))
    return ranked


def select_top_n(candidates: list[dict], n: int) -> list[str]:
    """Return the surface forms of the top-``n`` candidates by score."""
    return [c["token"] for c in candidates[:n]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    ranked = mine(cfg)
    out = Path(cfg_get(cfg, "paths.candidates_file", "results/candidates.json"))
    write_json(
        {
            "config_name": cfg.get("name", Path(args.config).stem),
            "model": cfg_get(cfg, "model.id"),
            "top_n": cfg_get(cfg, "mining.top_n", 128),
            "num_candidates": len(ranked),
            "candidates": ranked,
        },
        out,
    )
    log.info("Wrote %d candidates -> %s", len(ranked), out)


if __name__ == "__main__":
    main()
