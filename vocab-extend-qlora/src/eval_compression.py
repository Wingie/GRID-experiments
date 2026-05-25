"""Measure tokenizer compression from the vocabulary extension.

For each source file: ``tokens_before`` (base tokenizer) vs ``tokens_after`` (extended
tokenizer), compression ratio, and bytes/token. Aggregates mean/median/p95 overall and
per language, and breaks the saving down into freq-tokens-only, SID-tokens-only, and
both — to test whether the two token sources are complementary or redundant.

CPU-runnable when the base tokenizer can be loaded. With the offline fallback tokenizer
(which cannot add tokens) it degrades to a surface-replacement estimate.

Usage:
    python -m src.eval_compression configs/experiments/semid_qlora.yaml
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from src.extend_tokenizer import build_token_lists
from src.utils import cfg_get, get_logger, iter_repo_files, load_config, load_hf_tokenizer, write_json

log = get_logger("eval_compression")


def _extended_tokenizer(model_id: str, tokens: list[str]):
    """Load a fresh tokenizer copy and add ``tokens`` to it (real tokenizers only)."""
    tok, is_real = load_hf_tokenizer(model_id, allow_fallback=False)
    tok.add_tokens(tokens)
    return tok


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "median": 0, "p95": 0}
    s = sorted(values)
    return {
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 4),
    }


def evaluate(cfg: dict) -> dict:
    model_id = cfg_get(cfg, "model.id")
    files = list(iter_repo_files(cfg_get(cfg, "paths.target_repo"), cfg_get(cfg, "languages", ["py"])))
    freq_tokens, sid_vocab, _cb = build_token_lists(cfg)
    sid_tokens = sid_vocab.special_tokens() if sid_vocab is not None else []

    base, real = load_hf_tokenizer(model_id)
    variants = {"base": base}
    if real:
        try:
            if freq_tokens:
                variants["freq"] = _extended_tokenizer(model_id, freq_tokens)
            if sid_tokens:
                variants["sid"] = _extended_tokenizer(model_id, sid_tokens)
            variants["both"] = _extended_tokenizer(model_id, freq_tokens + sid_tokens)
        except Exception as exc:
            log.warning("Could not build extended tokenizers (%s); base only.", exc)

    counts: dict[str, int] = defaultdict(int)
    ratios_per_lang: dict[str, list[float]] = defaultdict(list)
    bytes_total = 0
    per_file_ratio: list[float] = []
    for rf in files:
        nbytes = len(rf.text.encode("utf-8"))
        bytes_total += nbytes
        file_counts = {name: len(tok.tokenize(rf.text)) for name, tok in variants.items()}
        for name, c in file_counts.items():
            counts[name] += c
        if file_counts.get("both") and file_counts["base"]:
            r = file_counts["both"] / file_counts["base"]
            per_file_ratio.append(r)
            ratios_per_lang[rf.language].append(r)

    def ratio(name: str) -> float:
        return round(counts[name] / counts["base"], 4) if counts.get("base") else 1.0

    summary = {
        "model": model_id,
        "real_tokenizer": real,
        "num_files": len(files),
        "num_freq_tokens": len(freq_tokens),
        "num_sid_tokens": len(sid_tokens),
        "tokens": dict(counts),
        "compression_ratio": {name: ratio(name) for name in variants if name != "base"},
        "reduction_pct": {
            name: round(100 * (1 - ratio(name)), 2) for name in variants if name != "base"
        },
        "bytes_per_token_after": round(bytes_total / max(1, counts.get("both", counts["base"])), 3),
        "per_file_ratio": _percentiles(per_file_ratio),
        "per_language_reduction_pct": {
            lang: round(100 * (1 - statistics.fmean(rs)), 2)
            for lang, rs in ratios_per_lang.items()
            if rs
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    summary = evaluate(cfg)
    out = Path(cfg_get(cfg, "paths.results_dir", "results")) / f"compression_{cfg.get('name','run')}.json"
    write_json(summary, out)
    log.info("Compression report -> %s", out)


if __name__ == "__main__":
    main()
