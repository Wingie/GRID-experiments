"""VideoGameBench benchmark runner — play N episodes per game, aggregate per-game scores
into one comparison table.

The runner is policy-agnostic: it takes any callable ``policy(observation) -> (reasoning,
actions)`` (the same shape :mod:`rimworld_agent.game.game_loop` expects) and drives the
:class:`GameBackend` protocol. Default targets are the headline games (Pokémon Red + Zelda:
The Minish Cap, capped per current spec); extra game ids can be passed via config.

Commentary mode: when ``commentary_questions`` is provided, each backend is wrapped with
:class:`CommentaryWrapper` so the user can ask questions while the agent plays. Per-game
question sources are surfaced through ``Observation.metadata["user_question"]``; policies
that answer with the ``say`` action contribute to a commentary-quality reward stream that is
aggregated alongside the gameplay reward in the per-game table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rimworld_agent.game.action_space import MAX_ACTIONS_PER_TURN, Action
from rimworld_agent.games.base import GameBackend, Observation, get_backend
from rimworld_agent.games.commentary import CommentaryWrapper, FileQuestionSource, QuestionSource
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
    # Commentary metrics (zero when commentary is disabled).
    say_count: int = 0
    commentary_reward: float = 0.0
    questions_received: int = 0


def _maybe_wrap_for_commentary(
    backend: GameBackend, questions_path: str | Path | None
) -> tuple[GameBackend, CommentaryWrapper | None]:
    """Wrap ``backend`` with :class:`CommentaryWrapper` when a questions file is given."""
    if not questions_path:
        return backend, None
    source: QuestionSource = FileQuestionSource(path=Path(questions_path))
    wrapper = CommentaryWrapper(backend, source=source)
    return wrapper, wrapper


def play_one(backend: GameBackend, policy: Policy, max_steps: int, seed: int | None,
             wrapper: CommentaryWrapper | None = None) -> EpisodeSummary:
    obs = backend.reset(seed=seed)
    total = 0.0
    info_progress = 0.0
    say_count = 0
    commentary_reward = 0.0
    questions_received = 0
    seen_questions: set[str] = set()
    for step in range(max_steps):
        if obs.metadata and obs.metadata.get("user_question"):
            q = obs.metadata["user_question"]
            if q not in seen_questions:
                questions_received += 1
                seen_questions.add(q)
        _reasoning, actions = policy(obs)
        actions = actions[: MAX_ACTIONS_PER_TURN]
        say_count += sum(1 for a in actions if a.action == "say")
        for a in actions:
            backend.execute(a)
        next_obs = backend.observe()
        breakdown = backend.reward(obs, next_obs, actions)
        total += breakdown.get("total", 0.0)
        if "commentary" in breakdown:
            commentary_reward += float(breakdown["commentary"])
        info_progress = float(next_obs.metadata.get("info", {}).get("progress", info_progress))
        obs = next_obs
        terminated = getattr(obs.state, "terminated", False)
        if terminated:
            return EpisodeSummary(backend.name, total, step + 1, True, info_progress,
                                   say_count, commentary_reward, questions_received)
    return EpisodeSummary(backend.name, total, max_steps, False, info_progress,
                           say_count, commentary_reward, questions_received)


def run_benchmark(
    policy: Policy,
    games: list[str] = list(HEADLINE_GAMES),
    n_episodes: int = 5,
    max_steps: int = 1000,
    seed: int | None = 0,
    backend_kwargs: dict[str, dict] | None = None,
    commentary_questions: dict[str, str | Path] | None = None,
) -> dict:
    """Run ``n_episodes`` per game and aggregate the per-game summaries.

    ``commentary_questions`` maps a game id to a file of user questions (one per line); when
    provided, that backend is wrapped with :class:`CommentaryWrapper` and commentary metrics
    flow into the per-game table.
    """
    backend_kwargs = backend_kwargs or {}
    commentary_questions = commentary_questions or {}
    table: dict[str, dict] = {}
    for game in games:
        kwargs = backend_kwargs.get(game, {})
        try:
            backend = get_backend(game, **kwargs)
        except Exception as exc:
            log.warning("skipping %s: %s", game, exc)
            table[game] = {"error": str(exc)}
            continue
        backend, wrapper = _maybe_wrap_for_commentary(backend, commentary_questions.get(game))
        try:
            summaries = [play_one(backend, policy, max_steps, (seed or 0) + i, wrapper) for i in range(n_episodes)]
        finally:
            backend.close()
        rewards = [s.total_reward for s in summaries]
        progress = [s.progress for s in summaries]
        terms = [s.terminated for s in summaries]
        row = {
            "n_episodes": n_episodes,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "best_reward": max(rewards) if rewards else 0.0,
            "termination_rate": sum(terms) / len(terms) if terms else 0.0,
            "mean_progress": sum(progress) / len(progress) if progress else 0.0,
        }
        if wrapper is not None:
            answered = sum(s.say_count for s in summaries)
            received = sum(s.questions_received for s in summaries)
            row["commentary"] = {
                "questions_received": received,
                "answered": answered,
                "answer_rate": (answered / received) if received else 0.0,
                "mean_commentary_reward": (
                    sum(s.commentary_reward for s in summaries) / len(summaries) if summaries else 0.0
                ),
            }
        table[game] = row
        log.info("%s: %s", game, row)
    return {"games": table, "n_games": len(games), "headline": list(HEADLINE_GAMES),
            "commentary_enabled": bool(commentary_questions)}


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        # Use a random policy by default; downstream loads the real model.
        from rimworld_agent.training.self_play import RandomPolicy

        games = list(cfg_get(cfg, "benchmark.games", HEADLINE_GAMES))
        commentary = {}
        if cfg_get(cfg, "benchmark.commentary.enabled", False):
            commentary = dict(cfg_get(cfg, "benchmark.commentary.questions", {}) or {})
        result = run_benchmark(
            RandomPolicy(),
            games=games,
            n_episodes=cfg_get(cfg, "benchmark.n_episodes", 5),
            max_steps=cfg_get(cfg, "benchmark.max_steps", 1000),
            seed=cfg_get(cfg, "seed", 0),
            commentary_questions=commentary,
        )
        out = ensure_dir(Path(cfg_get(cfg, "paths.results_dir", "results"))) / "videogamebench.json"
        write_json(result, out)
        log.info("wrote VGB benchmark -> %s", out)

    _run()


if __name__ == "__main__":
    main()
