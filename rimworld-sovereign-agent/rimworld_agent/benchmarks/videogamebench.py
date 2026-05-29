"""VideoGameBench benchmark runner — play N episodes per game, aggregate per-game scores
into one comparison table.

The runner is policy-agnostic: it takes any callable ``policy(observation) -> (reasoning,
actions)`` (the same shape :mod:`rimworld_agent.game.game_loop` expects) and drives the
:class:`GameBackend` protocol. Default targets are the headline games (Pokémon Red + Zelda:
The Minish Cap, capped per current spec); extra game ids can be passed via config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rimworld_agent.game.action_space import MAX_ACTIONS_PER_TURN, Action
from rimworld_agent.games.base import GameBackend, Observation, get_backend
from rimworld_agent.games.videogamebench import HEADLINE_GAMES
from rimworld_agent.utils import cfg_get, ensure_dir, get_logger, write_json

log = get_logger("benchmarks.vgb")


Policy = Callable[[Observation], tuple[str, list[Action]]]


@dataclass
class EpisodeSummary:
    game: str
    total_reward: float
    steps: int
    terminated: bool
    progress: float


def play_one(backend: GameBackend, policy: Policy, max_steps: int, seed: int | None) -> EpisodeSummary:
    obs = backend.reset(seed=seed)
    total = 0.0
    info_progress = 0.0
    for step in range(max_steps):
        _reasoning, actions = policy(obs)
        actions = actions[: MAX_ACTIONS_PER_TURN]
        for a in actions:
            backend.execute(a)
        next_obs = backend.observe()
        breakdown = backend.reward(obs, next_obs, actions)
        total += breakdown.get("total", 0.0)
        info_progress = float(next_obs.metadata.get("info", {}).get("progress", info_progress))
        obs = next_obs
        terminated = getattr(obs.state, "terminated", False)
        if terminated:
            return EpisodeSummary(backend.name, total, step + 1, True, info_progress)
    return EpisodeSummary(backend.name, total, max_steps, False, info_progress)


def run_benchmark(
    policy: Policy,
    games: list[str] = list(HEADLINE_GAMES),
    n_episodes: int = 5,
    max_steps: int = 1000,
    seed: int | None = 0,
    backend_kwargs: dict[str, dict] | None = None,
) -> dict:
    """Run ``n_episodes`` per game and aggregate the per-game summaries."""
    backend_kwargs = backend_kwargs or {}
    table: dict[str, dict] = {}
    for game in games:
        kwargs = backend_kwargs.get(game, {})
        try:
            backend = get_backend(game, **kwargs)
        except Exception as exc:
            log.warning("skipping %s: %s", game, exc)
            table[game] = {"error": str(exc)}
            continue
        try:
            summaries = [play_one(backend, policy, max_steps, (seed or 0) + i) for i in range(n_episodes)]
        finally:
            backend.close()
        rewards = [s.total_reward for s in summaries]
        progress = [s.progress for s in summaries]
        terms = [s.terminated for s in summaries]
        table[game] = {
            "n_episodes": n_episodes,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "best_reward": max(rewards) if rewards else 0.0,
            "termination_rate": sum(terms) / len(terms) if terms else 0.0,
            "mean_progress": sum(progress) / len(progress) if progress else 0.0,
        }
        log.info("%s: %s", game, table[game])
    return {"games": table, "n_games": len(games), "headline": list(HEADLINE_GAMES)}


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        # Use a random policy by default; downstream loads the real model.
        from rimworld_agent.training.self_play import RandomPolicy

        games = list(cfg_get(cfg, "benchmark.games", HEADLINE_GAMES))
        result = run_benchmark(
            RandomPolicy(),
            games=games,
            n_episodes=cfg_get(cfg, "benchmark.n_episodes", 5),
            max_steps=cfg_get(cfg, "benchmark.max_steps", 1000),
            seed=cfg_get(cfg, "seed", 0),
        )
        out = ensure_dir(Path(cfg_get(cfg, "paths.results_dir", "results"))) / "videogamebench.json"
        write_json(result, out)
        log.info("wrote VGB benchmark -> %s", out)

    _run()


if __name__ == "__main__":
    main()
