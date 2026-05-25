"""Rewrite training data with semantic IDs, producing the data formats and task
objectives the model learns during QLoRA.

Formats (see ``data.sid_format_ratios``):
  1. raw          - unmodified entity source (general code fluency).
  2. sid_header   - entity source preceded by a comment carrying its SID
                    (teach: "this entity IS <SID...>" before the implementation).
  3. sid_replace  - entity name replaced by its inline SID (bidirectional SID<->impl).

Objectives (see ``data.objective_weights``):
  1. next_token     - plain causal-LM over code (covered by formats 1-3).
  2. entity_sid     - given a body, predict its SID.
  3. sid_signature  - given a SID, predict the signature.
  4. sid_completion - given surrounding sibling SIDs, complete the implementation (FIM).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.semantic_ids.assign_ids import SemanticIDVocab
from src.semantic_ids.extract_entities import CodeEntity
from src.utils import cfg_get, get_logger

log = get_logger("inject_ids")

_COMMENT_PREFIX = {
    "python": "#",
    "ruby": "#",
    "go": "//",
    "rust": "//",
    "java": "//",
    "javascript": "//",
    "typescript": "//",
    "tsx": "//",
    "solidity": "//",
}


@dataclass
class TrainingExample:
    text: str
    task: str  # next_token | entity_sid | sid_signature | sid_completion
    format: str  # raw | sid_header | sid_replace | task


def comment_prefix(language: str) -> str:
    return _COMMENT_PREFIX.get(language, "#")


# --------------------------------------------------------------------------- #
# Formats
# --------------------------------------------------------------------------- #
def format_raw(entity: CodeEntity) -> str:
    return entity.body


def format_sid_header(entity: CodeEntity, sid_sequence: str) -> str:
    pre = comment_prefix(entity.language)
    return f"{pre} {sid_sequence}\n{entity.body}"


def format_sid_replace(entity: CodeEntity, sid_inline: str) -> str:
    """Replace the entity's own name with its inline SID (first occurrence in body)."""
    if entity.name and entity.name in entity.body:
        return entity.body.replace(entity.name, sid_inline, 1)
    return entity.body


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #
def obj_entity_sid(entity: CodeEntity, sid_sequence: str) -> str:
    return f"<PREDICT_SID> {entity.body}\n{sid_sequence}"


def obj_sid_signature(entity: CodeEntity, sid_inline: str) -> str:
    return f"<SID_TO_SIG>{sid_inline} {entity.signature}"


def obj_sid_completion(entity: CodeEntity, assignments: dict, sibling_sids: list[str]) -> str:
    """FIM-style: show the class context + sibling SIDs, then complete this body."""
    a = assignments[entity.uid]
    pre = comment_prefix(entity.language)
    siblings = ", ".join(sibling_sids) if sibling_sids else "(none)"
    header = f"{pre} class {a['parent'] or ''} methods: {siblings}\n" if a.get("parent") else ""
    return (
        f"{header}<FIM_PREFIX>{entity.signature}<FIM_SUFFIX><FIM_MIDDLE>{entity.body}"
    )


def make_examples(
    entities: list[CodeEntity],
    assignments: dict[str, dict],
    vocab: SemanticIDVocab,
    cfg: dict,
) -> list[TrainingExample]:
    """Build the full set of SID-aware training examples (formats + task objectives)."""
    fmt_ratios = dict(cfg_get(cfg, "data.sid_format_ratios", {"raw": 0.7, "sid_header": 0.2, "sid_replace": 0.1}))

    # sibling SID map: parent class -> list of member inline SIDs
    siblings: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        a = assignments.get(e.uid)
        if a and a.get("parent"):
            siblings[f"{e.file_path}::{a['parent']}"].append(a["sid_inline"])

    examples: list[TrainingExample] = []
    for e in entities:
        a = assignments.get(e.uid)
        if a is None:
            continue
        seq, inline = a["sid_sequence"], a["sid_inline"]

        # Formats: assign each entity to one format by ratio (deterministic by hash).
        bucket = (hash(e.uid) % 1000) / 1000.0
        cum = 0.0
        chosen = "raw"
        for fmt, r in fmt_ratios.items():
            cum += r
            if bucket < cum:
                chosen = fmt
                break
        if chosen == "raw":
            examples.append(TrainingExample(format_raw(e), "next_token", "raw"))
        elif chosen == "sid_header":
            examples.append(TrainingExample(format_sid_header(e, seq), "next_token", "sid_header"))
        elif chosen == "sid_replace":
            examples.append(TrainingExample(format_sid_replace(e, inline), "next_token", "sid_replace"))

        # Task objectives 2-4 (one of each per entity; sampled/weighted in prepare_data).
        examples.append(TrainingExample(obj_entity_sid(e, seq), "entity_sid", "task"))
        examples.append(TrainingExample(obj_sid_signature(e, inline), "sid_signature", "task"))
        sib = [s for s in siblings.get(f"{e.file_path}::{a['parent']}", []) if s != inline]
        examples.append(
            TrainingExample(obj_sid_completion(e, assignments, sib), "sid_completion", "task")
        )

    log.info("Built %d SID training examples from %d entities", len(examples), len(entities))
    return examples


def main() -> None:
    import argparse
    from pathlib import Path

    from src.semantic_ids.assign_ids import run_pipeline
    from src.utils import load_config, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    entities, _emb, _model, vocab, assignments = run_pipeline(cfg)
    examples = make_examples(entities, assignments, vocab, cfg)
    out = Path(cfg_get(cfg, "paths.data_dir", "data")) / "sid_examples.json"
    write_json([e.__dict__ for e in examples], out)
    log.info("Wrote %d examples -> %s", len(examples), out)


if __name__ == "__main__":
    main()
