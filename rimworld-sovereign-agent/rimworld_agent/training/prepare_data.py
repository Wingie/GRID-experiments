"""Build the multi-source training dataset (project spec §6).

Sources and their default mix (spec §6a):
  code understanding (FIM)   : xml_defs, csharp_source, wiki_text
  semantic-ID training       : entity_to_sid, sid_to_entity, sid_hierarchy
  gameplay                   : episode_replay, action_prediction, reasoning_only

FIM rendering reuses vocab-extend-qlora's ``src.prepare_data.make_fim``. The gameplay
formats are defined here (spec §6c). The result is an interleaved list of ``{text, source}``
records saved as a HF dataset (JSONL fallback when ``datasets`` is unavailable).

The formatting functions are pure string logic and are offline-verifiable; only the final
HF ``save`` step and the FIM helper need the extra deps.
"""

from __future__ import annotations

import random
from pathlib import Path

from rimworld_agent.game.action_space import format_actions
from rimworld_agent.game.episode_recorder import GameState, load_episode
from rimworld_agent.utils import cfg_get, ensure_dir, ensure_veq_importable, get_logger

log = get_logger("prepare_data")

DEFAULT_MIX = {
    "xml_defs": 0.12,
    "csharp_source": 0.08,
    "wiki_text": 0.10,
    # READ SID training
    "entity_to_rsid": 0.04,
    "rsid_to_entity": 0.04,
    "rsid_hierarchy": 0.02,
    # WRITE SID training
    "entity_to_wsid": 0.04,
    "wsid_sequence": 0.04,
    "wsid_to_action": 0.02,
    # gameplay
    "episode_replay": 0.30,
    "action_prediction": 0.10,
    "reasoning_only": 0.10,
}


# --------------------------------------------------------------------------- #
# Gameplay formats (spec §6c): RSIDs describe what is SEEN, WSIDs plan what to DO.
# --------------------------------------------------------------------------- #
def render_game_state(state: GameState, visible_rsids: list[str] | None = None) -> str:
    colonists = ", ".join(
        f"{c.name} ({c.job}, mood {int(c.mood * 100)}%)" for c in state.colonists
    )
    resources = ", ".join(f"{k} {int(v)}" for k, v in state.resources.items())
    research = ", ".join(f"{k} {int(v * 100)}%" for k, v in state.research_progress.items())
    quests = ", ".join(f"{q.get('name')} ({q.get('progress')})" for q in state.quests_active)
    threats = ", ".join(t.get("name", "?") for t in state.threats) or "None"
    rsids = " ".join(visible_rsids or [])
    return (
        "<GAME_STATE>\n"
        f"  Season: {state.season}, Day {state.day}\n"
        f"  Colonists: {colonists}\n"
        f"  Resources: {resources}\n"
        f"  Research: {research}\n"
        f"  Active Quests: {quests}\n"
        f"  Visible entities (READ): {rsids}\n"
        f"  Threats: {threats}\n"
        "</GAME_STATE>"
    )


def _render_actions(step) -> str:
    """Render the action block, annotating each action with its WRITE SID when available."""
    wsids = list(getattr(step, "action_wsids", []))
    lines = ["<ACTIONS>"]
    for i, a in enumerate(step.actions, 1):
        param_str = ", ".join(f"{k}={v}" for k, v in a.params.items())
        wsid = f"{wsids[i - 1]}: " if i - 1 < len(wsids) else ""
        lines.append(f"  <ACTION_{i}>{a.action}({param_str}) — {wsid}{a.reason}</ACTION_{i}>")
    lines.append("</ACTIONS>")
    return "\n".join(lines)


def format_episode_replay(step) -> str:
    state = render_game_state(step.game_state, step.visible_rsids)
    screenshots = "<SCREENSHOTS>\n  <VISION_START>" + "<FRAME_SEP>".join(
        Path(s).name for s in step.screenshots
    ) + "<VISION_END>\n</SCREENSHOTS>"
    reasoning = f"<REASONING>\n  {step.reasoning}\n</REASONING>"
    return "\n".join([state, screenshots, reasoning, _render_actions(step)])


def format_action_prediction(step) -> str:
    """State (with RSIDs) -> actions (with WSIDs); teaches planning from structured state."""
    return "\n".join([render_game_state(step.game_state, step.visible_rsids), _render_actions(step)])


def format_reasoning_only(step) -> str:
    return "\n".join(
        [render_game_state(step.game_state, step.visible_rsids), f"<REASONING>\n  {step.reasoning}\n</REASONING>"]
    )


# --------------------------------------------------------------------------- #
# Dual semantic-ID task formats (spec §6a)
# --------------------------------------------------------------------------- #
def format_entity_to_rsid(a: dict) -> str:
    return f"{a['def_type']} {a['def_name']}: {a['label']} -> {a['rsid_sequence']}"


def format_rsid_to_entity(a: dict) -> str:
    return f"{a['rsid_sequence']} -> {a['def_name']}, {a.get('subcategory', '')}, {a['label']}"


def format_rsid_hierarchy(a: dict, siblings: list[str]) -> str:
    prefix = a["rsid_inline"].rsplit("<RSID_L", 1)[0] if "<RSID_L" in a["rsid_inline"] else ""
    return f"{prefix}* members: {', '.join(siblings[:8])}"


def format_entity_to_wsid(a: dict) -> str:
    return f"{a['def_name']} in build context -> {a['wsid_sequence']}"


def format_wsid_to_action(a: dict) -> str:
    return f"{a['wsid_sequence']} -> order_build({a['def_name']})"


# --------------------------------------------------------------------------- #
# Code FIM (reuse veq)
# --------------------------------------------------------------------------- #
def _fim_examples_from_dir(directory: str | Path, glob: str, fim_rate: float, rng: random.Random) -> list[str]:
    ensure_veq_importable()
    from src.prepare_data import make_fim  # type: ignore

    out: list[str] = []
    for path in Path(directory).rglob(glob):
        text = path.read_text(encoding="utf8", errors="replace")
        for chunk in _chunk(text, 1500):
            out.append(make_fim(chunk, rng) if rng.random() < fim_rate else chunk)
    return out


def _chunk(text: str, max_chars: int) -> list[str]:
    lines, buf, size, chunks = text.splitlines(keepends=True), [], 0, []
    for ln in lines:
        buf.append(ln)
        size += len(ln)
        if size >= max_chars:
            chunks.append("".join(buf))
            buf, size = [], 0
    if buf:
        chunks.append("".join(buf))
    return chunks


# --------------------------------------------------------------------------- #
# Build + save
# --------------------------------------------------------------------------- #
def build_examples(cfg) -> list[dict]:
    rng = random.Random(cfg_get(cfg, "seed", 42))
    mix = dict(cfg_get(cfg, "training.training_mix", DEFAULT_MIX))
    fim_rate = cfg_get(cfg, "training.fim_rate", 0.5)
    examples: list[dict] = []

    def add(source: str, texts: list[str]) -> None:
        for t in texts:
            if t and t.strip():
                examples.append({"text": t, "source": source})

    # Code/wiki FIM
    add("xml_defs", _fim_examples_from_dir(cfg_get(cfg, "paths.xml_defs_dir", "data/rimworld_xml_defs"), "*.xml", fim_rate, rng))
    add("csharp_source", _fim_examples_from_dir(cfg_get(cfg, "paths.csharp_source_dir", "data/rimworld_source"), "*.cs", fim_rate, rng))
    add("wiki_text", _fim_examples_from_dir(cfg_get(cfg, "paths.wiki_dir", "data/rimworld_wiki"), "*.md", fim_rate, rng))

    # Dual semantic-ID tasks (needs precomputed assignments)
    sid_file = Path(cfg_get(cfg, "paths.sid_assignments_file", "results/sid_assignments.json"))
    if sid_file.exists():
        import json

        data = json.loads(sid_file.read_text())
        assigns = list(data["assignments"].values())
        # READ tasks (all entities)
        add("entity_to_rsid", [format_entity_to_rsid(a) for a in assigns])
        add("rsid_to_entity", [format_rsid_to_entity(a) for a in assigns])
        by_sub: dict[str, list[str]] = {}
        for a in assigns:
            by_sub.setdefault(a.get("subcategory", "?"), []).append(a["def_name"])
        add("rsid_hierarchy", [format_rsid_hierarchy(a, by_sub.get(a.get("subcategory", "?"), [])) for a in assigns])
        # WRITE tasks (only entities that have a WSID)
        with_wsid = [a for a in assigns if a.get("wsid_sequence")]
        add("entity_to_wsid", [format_entity_to_wsid(a) for a in with_wsid])
        add("wsid_to_action", [format_wsid_to_action(a) for a in with_wsid])

    # Gameplay episodes
    ep_dir = Path(cfg_get(cfg, "paths.episodes_dir", "data/episodes"))
    replay, action_pred, reasoning, wsid_seq = [], [], [], []
    for ep_path in ep_dir.glob("*/episode.json"):
        ep = load_episode(ep_path)
        for step in ep.steps:
            replay.append(format_episode_replay(step))
            action_pred.append(format_action_prediction(step))
            reasoning.append(format_reasoning_only(step))
            if step.action_wsids:  # the workflow that actually occurred this step
                wsid_seq.append(" then ".join(step.action_wsids))
    add("episode_replay", replay)
    add("action_prediction", action_pred)
    add("reasoning_only", reasoning)
    add("wsid_sequence", wsid_seq)

    return _resample_to_mix(examples, mix, rng)


def _resample_to_mix(examples: list[dict], mix: dict[str, float], rng: random.Random) -> list[dict]:
    """Resample per-source so source proportions approximate ``mix`` (sum normalised)."""
    by_source: dict[str, list[dict]] = {}
    for ex in examples:
        by_source.setdefault(ex["source"], []).append(ex)
    if not by_source:
        return []
    total_weight = sum(mix.get(s, 0.0) for s in by_source) or 1.0
    # Size the dataset by the most-represented source's natural count.
    base = max(len(v) for v in by_source.values())
    out: list[dict] = []
    for source, items in by_source.items():
        target = int(base * (mix.get(source, 0.0) / total_weight) * len(by_source))
        if target <= 0 or not items:
            continue
        out.extend(rng.choices(items, k=target))
    rng.shuffle(out)
    return out


def save_dataset(examples: list[dict], out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    try:
        from datasets import Dataset

        ds = Dataset.from_list(examples)
        ds.save_to_disk(str(out_dir))
    except Exception as exc:  # datasets missing -> JSONL fallback
        import json

        log.warning("datasets unavailable (%s); writing JSONL", exc)
        with open(Path(out_dir) / "train.jsonl", "w") as fh:
            for ex in examples:
                fh.write(json.dumps(ex) + "\n")
    return Path(out_dir)


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        examples = build_examples(cfg)
        out = save_dataset(examples, cfg_get(cfg, "paths.dataset_dir", "data/dataset"))
        log.info("built %d training examples -> %s", len(examples), out)

    _run()


if __name__ == "__main__":
    main()
