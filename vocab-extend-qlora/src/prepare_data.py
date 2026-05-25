"""Convert the target repo into a training dataset.

  - file-level chunks with a ``# File: <path>`` context header;
  - sliding window for files longer than ``data.max_seq_len`` (approximate, line-based
    so it runs without a tokenizer);
  - 90/10 train/eval split, stratified by file extension;
  - optional Fill-in-the-Middle (FIM) rendering for completion training;
  - when ``semantic_ids.enabled``, interleave code chunks with SID task examples at
    ``data.sid_task_fraction``.

Saves a HuggingFace ``DatasetDict`` to ``paths.dataset_dir`` (falls back to JSONL when
``datasets`` is unavailable).
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from src.utils import RepoFile, cfg_get, ensure_dir, get_logger, iter_repo_files, load_config

log = get_logger("prepare_data")


def chunk_file(rf: RepoFile, max_chars: int, header: bool) -> list[str]:
    """Split a file into <=``max_chars`` chunks on line boundaries (sliding window)."""
    prefix = f"# File: {rf.rel_path}\n" if header else ""
    lines = rf.text.splitlines(keepends=True)
    chunks, buf, size = [], [prefix], len(prefix)
    for line in lines:
        if size + len(line) > max_chars and len(buf) > 1:
            chunks.append("".join(buf))
            buf, size = [prefix], len(prefix)
        buf.append(line)
        size += len(line)
    if len(buf) > 1:
        chunks.append("".join(buf))
    return chunks


def make_fim(text: str, rng: random.Random) -> str:
    """Render a chunk as PSM-order Fill-in-the-Middle."""
    n = len(text)
    if n < 16:
        return text
    a, b = sorted(rng.sample(range(n), 2))
    prefix, middle, suffix = text[:a], text[a:b], text[b:]
    return f"<FIM_PREFIX>{prefix}<FIM_SUFFIX>{suffix}<FIM_MIDDLE>{middle}"


def build_code_examples(cfg: dict) -> list[dict]:
    rng = random.Random(cfg_get(cfg, "seed", 42))
    files = list(iter_repo_files(cfg_get(cfg, "paths.target_repo"), cfg_get(cfg, "languages", ["py"])))
    max_chars = cfg_get(cfg, "data.max_seq_len", 2048) * 4  # ~4 chars/token heuristic
    header = cfg_get(cfg, "data.path_context_header", True)
    use_fim = cfg_get(cfg, "data.use_fim", True)
    fim_rate = cfg_get(cfg, "data.fim_rate", 0.5)

    examples: list[dict] = []
    for rf in files:
        for chunk in chunk_file(rf, max_chars, header):
            text = make_fim(chunk, rng) if (use_fim and rng.random() < fim_rate) else chunk
            examples.append({"text": text, "source": "code", "task": "next_token", "ext": rf.ext})
    log.info("Built %d code chunks from %d files", len(examples), len(files))
    return examples


def build_sid_examples(cfg: dict) -> list[dict]:
    """Build SID task examples (objectives 2-4 + SID-annotated formats)."""
    from src.semantic_ids.assign_ids import run_pipeline
    from src.semantic_ids.inject_ids import make_examples

    entities, _emb, _model, vocab, assignments = run_pipeline(cfg)
    sid_examples = make_examples(entities, assignments, vocab, cfg)
    return [
        {"text": ex.text, "source": f"sid_{ex.task}", "task": ex.task, "ext": "sid"}
        for ex in sid_examples
    ]


def _weighted_sample(examples: list[dict], weights: dict[str, float], rng: random.Random) -> list[dict]:
    """Resample task examples to roughly match ``data.objective_weights``."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_task[ex["task"]].append(ex)
    total = len(examples)
    out: list[dict] = []
    for task, exs in by_task.items():
        w = weights.get(task, 0.0)
        target = int(total * w)
        if target <= 0 or not exs:
            continue
        out.extend(rng.choices(exs, k=target))
    return out or examples


def stratified_split(examples: list[dict], eval_fraction: float, rng: random.Random):
    by_ext: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_ext[ex["ext"]].append(ex)
    train, evalset = [], []
    for exs in by_ext.values():
        rng.shuffle(exs)
        k = max(1, int(len(exs) * eval_fraction)) if len(exs) > 1 else 0
        evalset.extend(exs[:k])
        train.extend(exs[k:])
    rng.shuffle(train)
    rng.shuffle(evalset)
    return train, evalset


def build_dataset(cfg: dict):
    rng = random.Random(cfg_get(cfg, "seed", 42))
    code = build_code_examples(cfg)
    examples = code

    if cfg_get(cfg, "semantic_ids.enabled", False):
        sid = build_sid_examples(cfg)
        sid = _weighted_sample(sid, dict(cfg_get(cfg, "data.objective_weights", {})), rng)
        # interleave so SID examples are ~sid_task_fraction of the total
        frac = cfg_get(cfg, "data.sid_task_fraction", 0.4)
        n_sid_target = int(len(code) * frac / max(1e-6, 1 - frac))
        if sid:
            sid = rng.choices(sid, k=n_sid_target) if n_sid_target > len(sid) else rng.sample(sid, n_sid_target)
        examples = code + sid
        rng.shuffle(examples)
        log.info("Interleaved %d code + %d sid examples", len(code), len(sid))

    train, evalset = stratified_split(examples, cfg_get(cfg, "data.eval_fraction", 0.1), rng)
    return train, evalset


def save_dataset(train: list[dict], evalset: list[dict], out_dir: str):
    out = ensure_dir(out_dir)
    try:
        from datasets import Dataset, DatasetDict

        ds = DatasetDict(
            {"train": Dataset.from_list(train), "eval": Dataset.from_list(evalset)}
        )
        ds.save_to_disk(str(out))
        log.info("Saved HF DatasetDict (train=%d, eval=%d) -> %s", len(train), len(evalset), out)
    except Exception as exc:
        log.warning("datasets unavailable (%s); writing JSONL instead.", exc)
        import json

        for name, rows in (("train", train), ("eval", evalset)):
            with open(Path(out) / f"{name}.jsonl", "w") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    train, evalset = build_dataset(cfg)
    save_dataset(train, evalset, cfg_get(cfg, "paths.dataset_dir", "data/dataset"))


if __name__ == "__main__":
    main()
