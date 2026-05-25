"""Compare token counts before vs. after a candidate vocabulary extension.

This is a *static* analysis (no model needed): it tokenises the corpus with the base
tokenizer, then simulates the effect of the mined candidates by greedily replacing
their surface forms with a single token, and reports the reduction. Useful for picking
``mining.top_n`` before committing to a (slow) embedding resize + fine-tune.

Usage:
    python -m src.analyze_tokenizer configs/experiments/vocab_qlora.yaml
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from src.mine_tokens import mine, select_top_n
from src.utils import (
    cfg_get,
    get_logger,
    iter_repo_files,
    load_config,
    load_hf_tokenizer,
    write_json,
)

log = get_logger("analyze_tokenizer")


def _simulate_after(text: str, base_tokens: int, candidate_surfaces: list[str]) -> int:
    """Approximate the post-extension token count for a single file.

    Each occurrence of a candidate surface form collapses from its sub-token count to
    a single token. We compute the saving as: base_tokens - sum(occurrences * (subtoks - 1)).
    """
    saved = 0
    for surface in candidate_surfaces:
        if not surface:
            continue
        occ = text.count(surface)
        if occ:
            # The candidate originally cost len(surface)//~4 + boundary pieces; we
            # approximate sub-token cost as the number of base pieces it spans.
            saved += occ  # at least one piece saved per occurrence (single-token now)
    return max(1, base_tokens - saved)


def analyze(cfg: dict) -> dict:
    repo = cfg_get(cfg, "paths.target_repo")
    languages = cfg_get(cfg, "languages", ["py"])
    files = list(iter_repo_files(repo, languages))

    tokenizer, is_real = load_hf_tokenizer(cfg_get(cfg, "model.id"))
    ranked = mine(cfg)
    top_n = cfg_get(cfg, "mining.top_n", 128)
    surfaces = select_top_n(ranked, top_n)

    per_lang_before: dict[str, int] = defaultdict(int)
    per_lang_after: dict[str, int] = defaultdict(int)
    total_before = total_after = 0
    for f in files:
        before = len(tokenizer.tokenize(f.text))
        after = _simulate_after(f.text, before, surfaces)
        total_before += before
        total_after += after
        per_lang_before[f.language] += before
        per_lang_after[f.language] += after

    ratio = (total_after / total_before) if total_before else 1.0
    summary = {
        "model": cfg_get(cfg, "model.id"),
        "real_tokenizer": is_real,
        "top_n": top_n,
        "num_files": len(files),
        "tokens_before": total_before,
        "tokens_after": total_after,
        "compression_ratio": round(ratio, 4),
        "reduction_pct": round(100 * (1 - ratio), 2),
        "per_language": {
            lang: {
                "before": per_lang_before[lang],
                "after": per_lang_after[lang],
                "reduction_pct": round(
                    100 * (1 - per_lang_after[lang] / max(1, per_lang_before[lang])), 2
                ),
            }
            for lang in per_lang_before
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    summary = analyze(cfg)
    out = Path(cfg_get(cfg, "paths.results_dir", "results")) / "tokenizer_analysis.json"
    write_json(summary, out)
    log.info(
        "Estimated reduction: %.2f%% (%d -> %d tokens) -> %s",
        summary["reduction_pct"],
        summary["tokens_before"],
        summary["tokens_after"],
        out,
    )


if __name__ == "__main__":
    main()
