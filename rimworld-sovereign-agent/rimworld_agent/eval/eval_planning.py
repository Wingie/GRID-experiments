"""Planning-quality evaluation (project spec §9d).

Given model outputs for a set of game states, measure:
  * action validity      — fraction of emitted actions that parse and pass structural checks;
  * count compliance     — fraction of turns that emit 1–5 actions;
  * reason coverage       — fraction of actions that carry a non-empty reason;
  * SID usage             — fraction of turns whose reasoning references SID tokens.

The scoring functions are pure and offline-testable; generating the outputs needs the model
(GPU), so :func:`evaluate` accepts either a callable generator or precomputed outputs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from rimworld_agent.game.action_space import MAX_ACTIONS_PER_TURN, parse_actions
from rimworld_agent.utils import cfg_get, get_logger, write_json

log = get_logger("eval_planning")

_SID_RE = re.compile(r"<SID_L\d+_\d+>|<SID:[\d-]+>")


def score_output(text: str) -> dict:
    """Score a single model turn (reasoning + action blocks)."""
    actions = parse_actions(text)
    valid = [a for a in actions if a.is_valid()[0]]
    with_reason = [a for a in actions if a.reason.strip()]
    reasoning = text.split("<ACTION_START>", 1)[0]
    return {
        "num_actions": len(actions),
        "num_valid": len(valid),
        "action_validity": (len(valid) / len(actions)) if actions else 0.0,
        "count_ok": 1 <= len(actions) <= MAX_ACTIONS_PER_TURN,
        "reason_coverage": (len(with_reason) / len(actions)) if actions else 0.0,
        "uses_sid": bool(_SID_RE.search(reasoning)),
    }


def aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {"turns": 0}
    n = len(scores)
    return {
        "turns": n,
        "mean_action_validity": sum(s["action_validity"] for s in scores) / n,
        "count_compliance": sum(s["count_ok"] for s in scores) / n,
        "mean_reason_coverage": sum(s["reason_coverage"] for s in scores) / n,
        "sid_usage_rate": sum(s["uses_sid"] for s in scores) / n,
    }


def evaluate(cfg, generate: Callable[[str], str] | None = None, prompts: list[str] | None = None,
             outputs: list[str] | None = None) -> dict:
    if outputs is None:
        if generate is None or prompts is None:
            raise ValueError("provide either `outputs`, or both `generate` and `prompts`.")
        outputs = [generate(p) for p in prompts]
    result = aggregate([score_output(o) for o in outputs])
    log.info("planning: validity=%.2f sid_usage=%.2f", result.get("mean_action_validity", 0),
             result.get("sid_usage_rate", 0))
    return result


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        # Live eval would load the model and a held-out set of states; here we score any
        # precomputed outputs under results/planning_outputs.txt if present.
        path = Path(cfg_get(cfg, "paths.results_dir", "results")) / "planning_outputs.txt"
        outputs = path.read_text().split("\n===\n") if path.exists() else []
        result = evaluate(cfg, outputs=outputs)
        write_json(result, Path(cfg_get(cfg, "paths.results_dir", "results")) / "eval_planning.json")

    _run()


if __name__ == "__main__":
    main()
