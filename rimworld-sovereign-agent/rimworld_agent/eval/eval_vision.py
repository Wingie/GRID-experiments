"""Screenshot-understanding evaluation (project spec §9c).

Given a held-out set of screenshots with ground-truth state (from RIMAPI), probe whether the
model can identify: which buildings are visible, the season/weather, whether colonists are
idle/working/fighting, and whether a threat is on screen. Each probe is a constrained
question; accuracy is the fraction of correct answers.

GPU + a labelled screenshot set required; written-but-unverified here. The probe
definitions and the scoring are kept separate so the scoring is reusable/testable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from rimworld_agent.utils import cfg_get, get_logger, write_json

log = get_logger("eval_vision")

PROBES = {
    "buildings_visible": "Which buildings are visible on screen? List their names.",
    "season": "What is the current season?",
    "colonist_activity": "Are the colonists idle, working, or fighting?",
    "threat_on_screen": "Is there a threat visible on screen? Answer yes or no.",
}


def _match(probe: str, answer: str, truth: dict) -> bool:
    a = answer.lower()
    if probe == "season":
        return str(truth.get("season", "")).lower() in a
    if probe == "threat_on_screen":
        has_threat = bool(truth.get("threats"))
        return ("yes" in a) == has_threat
    if probe == "buildings_visible":
        names = [b.lower() for b in truth.get("visible_buildings", [])]
        return any(n in a for n in names) if names else "none" in a
    if probe == "colonist_activity":
        jobs = {c.get("job", "").lower() for c in truth.get("colonists", [])}
        label = "fighting" if "fighting" in jobs or truth.get("threats") else ("working" if jobs - {"idle"} else "idle")
        return label in a
    return False


def evaluate(cfg, answer_fn: Callable[[list[str], str], str]) -> dict:
    """``answer_fn(frame_paths, question) -> answer``. Iterates the labelled screenshot set."""
    shot_dir = Path(cfg_get(cfg, "paths.screenshots_dir", "data/screenshots"))
    records = sorted(shot_dir.glob("*.json"))
    per_probe = {p: [] for p in PROBES}
    for rec in records:
        truth = json.loads(rec.read_text())
        frames = [str(rec.with_suffix(".png"))]
        for probe, question in PROBES.items():
            ans = answer_fn(frames, question)
            per_probe[probe].append(_match(probe, ans, truth))
    result = {p: (sum(v) / len(v) if v else 0.0) for p, v in per_probe.items()}
    result["overall"] = sum(result.values()) / len(result) if result else 0.0
    log.info("vision accuracy: %s", {k: round(v, 2) for k, v in result.items()})
    return result


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        log.warning("eval_vision needs a model + labelled screenshots; see TRAINING_GUIDE.md")

    _run()


if __name__ == "__main__":
    main()
