"""Offline tests for the VideoGameBench adapter + cross-game benchmark runner.

The actual ``videogamebench`` package is an optional dep; here we only test:
  * the static headline-game constants and the action-space mapper;
  * the benchmark runner driving the toy backend (registered in ``test_games_registry``).
"""

import pytest

from rimworld_agent.benchmarks.videogamebench import run_benchmark
from rimworld_agent.game.action_space import Action
from rimworld_agent.games.base import Observation
from rimworld_agent.games.videogamebench import (
    GAME_POKEMON_RED,
    GAME_ZELDA_MINISH_CAP,
    HEADLINE_GAMES,
    _action_space_for_buttons,
)

# Make sure the toy backend is registered.
from tests import test_games_registry  # noqa: F401


def test_headline_games_capped_to_pokemon_and_zelda():
    assert HEADLINE_GAMES == (GAME_POKEMON_RED, GAME_ZELDA_MINISH_CAP)


def test_action_space_for_buttons():
    space = _action_space_for_buttons(("A", "B", "UP"))
    assert space["press_a"]["button"] == "A"
    assert space["press_up"]["button"] == "UP"
    assert space["wait"]["type"] == "noop"


def _fixed_policy(_obs: Observation):
    return "tick once", [Action("tick")]


def test_run_benchmark_against_toy_backend():
    result = run_benchmark(_fixed_policy, games=["toy"], n_episodes=3, max_steps=5, seed=0)
    assert result["n_games"] == 1
    row = result["games"]["toy"]
    assert row["n_episodes"] == 3
    # Each step the toy backend increments ticks by 1, so reward sums to max_steps per episode.
    assert row["mean_reward"] == pytest.approx(5.0)
    assert row["best_reward"] == pytest.approx(5.0)


def test_run_benchmark_skips_unimportable_backends():
    # videogamebench package not installed -> backend factory raises; runner records error
    # and keeps going.
    result = run_benchmark(_fixed_policy, games=["pokemon_red", "toy"], n_episodes=1, max_steps=1)
    assert "error" in result["games"]["pokemon_red"]
    assert "mean_reward" in result["games"]["toy"]
