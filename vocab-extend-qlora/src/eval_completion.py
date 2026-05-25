"""Code-completion accuracy on held-out files.

Masks a function/method body in held-out files, generates a completion with the model
variant under test, and scores it with exact-match, BLEU-4, edit-distance ratio, and
syntax validity (plus CodeBLEU when the ``codebleu`` package is installed). The same
script is run once per condition (base / qlora / vocab / vocab+qlora / semid / full);
``notebooks/03_results.ipynb`` aggregates the per-condition JSON into the comparison
table.

Requires a GPU + model weights; not runnable on CPU here.

Usage:
    python -m src.eval_completion configs/experiments/semid_qlora.yaml \
        model_path=data/models/semid_qlora variant=full
"""

from __future__ import annotations

import argparse
import ast
import random
from pathlib import Path

from src.semantic_ids.extract_entities import extract_entities
from src.utils import cfg_get, get_logger, iter_repo_files, load_config, set_seed, write_json

log = get_logger("eval_completion")


def _held_out_entities(cfg: dict):
    """Pick held-out function/method entities to mask (deterministic by seed)."""
    files = list(iter_repo_files(cfg_get(cfg, "paths.target_repo"), cfg_get(cfg, "languages", ["py"])))
    entities = [e for e in extract_entities(files) if e.kind in ("function", "method") and e.body]
    rng = random.Random(cfg_get(cfg, "seed", 42))
    rng.shuffle(entities)
    k = int(len(entities) * cfg_get(cfg, "data.eval_fraction", 0.1))
    held = entities[:k]
    return held[: cfg_get(cfg, "eval.num_completion_samples", 200)]


def _syntax_valid(code: str, language: str) -> bool:
    if language != "python":
        return bool(code.strip())  # only python has a cheap stdlib check
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _bleu(pred: str, ref: str) -> float:
    try:
        from sacrebleu import sentence_bleu

        return sentence_bleu(pred, [ref]).score / 100.0
    except Exception:
        return 0.0


def _edit_ratio(pred: str, ref: str) -> float:
    try:
        import Levenshtein

        return Levenshtein.ratio(pred, ref)
    except Exception:
        return 0.0


def _codebleu(pred: str, ref: str, language: str) -> float | None:
    try:
        from codebleu import calc_codebleu

        return calc_codebleu([ref], [pred], lang=language)["codebleu"]
    except Exception:
        return None


def _load_model(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = cfg_get(cfg, "model_path") or cfg_get(cfg, "model.id")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return tok, model


def evaluate(cfg: dict) -> dict:
    import torch

    set_seed(cfg_get(cfg, "seed", 42))
    tok, model = _load_model(cfg)
    held = _held_out_entities(cfg)
    max_new = cfg_get(cfg, "eval.completion_max_new_tokens", 256)

    per_example = []
    for e in held:
        prompt = e.signature + "\n"  # ask the model to produce the body
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        gen = tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        ref = e.body
        per_example.append(
            {
                "exact_match": float(gen.strip() == ref.strip()),
                "bleu": _bleu(gen, ref),
                "edit_ratio": _edit_ratio(gen, ref),
                "syntax_valid": float(_syntax_valid(prompt + gen, e.language)),
                "codebleu": _codebleu(gen, ref, e.language),
            }
        )

    def avg(key: str) -> float:
        vals = [x[key] for x in per_example if x[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "config_name": cfg.get("name"),
        "variant": cfg_get(cfg, "variant", cfg.get("name", "unknown")),
        "model_path": cfg_get(cfg, "model_path") or cfg_get(cfg, "model.id"),
        "num_samples": len(per_example),
        "exact_match": avg("exact_match"),
        "bleu": avg("bleu"),
        "edit_ratio": avg("edit_ratio"),
        "syntax_valid": avg("syntax_valid"),
        "codebleu": avg("codebleu"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    summary = evaluate(cfg)
    variant = summary["variant"]
    out = Path(cfg_get(cfg, "paths.results_dir", "results")) / f"completion_{variant}.json"
    write_json(summary, out)
    log.info("Completion eval (%s): EM=%.3f BLEU=%.3f -> %s", variant, summary["exact_match"], summary["bleu"], out)


if __name__ == "__main__":
    main()
