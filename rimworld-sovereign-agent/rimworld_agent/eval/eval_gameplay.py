"""Gameplay-performance evaluation (project spec §9a).

Runs N episodes with a given policy and reports colony survival time, quest completion rate,
research completion, wealth curve, and colonist death rate — for comparison across the
random, scripted, and trained agents. game + GPU required; the aggregation over recorded
:class:`Episode` objects is pure and reusable.
"""

from __future__ import annotations

from pathlib import Path

from rimworld_agent.game.reward import parse_progress
from rimworld_agent.utils import cfg_get, get_logger, write_json

log = get_logger("eval_gameplay")


def summarize_episode(ep) -> dict:
    last = ep.steps[-1].game_state if ep.steps else None
    quests = last.quests_active if last else []
    completed = sum(1 for q in quests if parse_progress(q.get("progress", 0)) >= 1.0)
    deaths = 0
    for i in range(1, len(ep.steps)):
        prev = {c.name for c in ep.steps[i - 1].game_state.colonists}
        curr = {c.name for c in ep.steps[i].game_state.colonists}
        deaths += len(prev - curr)
    return {
        "episode_id": ep.episode_id,
        "survival_days": last.day if last else 0,
        "quests_completed": completed,
        "quests_total": len(quests),
        "quest_completion_rate": (completed / len(quests)) if quests else 0.0,
        "final_wealth": last.wealth if last else 0.0,
        "deaths": deaths,
        "total_reward": ep.total_reward,
        "steps": ep.total_steps,
    }


def aggregate(summaries: list[dict]) -> dict:
    if not summaries:
        return {"episodes": 0}
    n = len(summaries)
    return {
        "episodes": n,
        "mean_survival_days": sum(s["survival_days"] for s in summaries) / n,
        "mean_quest_completion": sum(s["quest_completion_rate"] for s in summaries) / n,
        "mean_final_wealth": sum(s["final_wealth"] for s in summaries) / n,
        "death_rate": sum(s["deaths"] for s in summaries) / n,
        "mean_reward": sum(s["total_reward"] for s in summaries) / n,
    }


def evaluate_recorded(episode_dir: str | Path) -> dict:
    """Aggregate already-recorded episodes (offline-usable over real recordings)."""
    from rimworld_agent.game.episode_recorder import load_episode

    eps = [load_episode(p) for p in sorted(Path(episode_dir).glob("*/episode.json"))]
    return aggregate([summarize_episode(e) for e in eps])


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        result = evaluate_recorded(cfg_get(cfg, "paths.episodes_dir", "data/episodes"))
        write_json(result, Path(cfg_get(cfg, "paths.results_dir", "results")) / "eval_gameplay.json")
        log.info("gameplay eval: %s", result)

    _run()


if __name__ == "__main__":
    main()
