"""Offline tests for the game-backend protocol and registry."""

from pathlib import Path

import pytest

from rimworld_agent.games.base import (
    Action,
    ExecutionResult,
    GameBackend,
    Observation,
    get_backend,
    list_backends,
    register,
)


class _Toy:
    """Minimal backend used to exercise the protocol + registry without any game deps."""

    name = "toy"

    def __init__(self, **kw):
        self.kw = kw
        self._ticks = 0

    def action_space(self):
        return {"tick": {"type": "noop"}, "wait": {"type": "noop"}}

    def knowledge_dirs(self):
        return {"data": Path("data/toy")}

    def observe(self):
        return Observation(state={"ticks": self._ticks})

    def execute(self, action):
        self._ticks += 1
        return ExecutionResult(ok=True, result={"ticks": self._ticks})

    def reward(self, prev, curr, actions):
        return {"total": float(curr.state["ticks"] - prev.state["ticks"])}

    def reset(self, seed=None):
        self._ticks = 0
        return self.observe()

    def close(self):
        pass


@register("toy")
def _factory(**kw):
    return _Toy(**kw)


def test_toy_backend_satisfies_protocol():
    b = get_backend("toy", flag=True)
    assert isinstance(b, GameBackend)
    assert b.name == "toy"
    assert "tick" in b.action_space()
    assert b.kw == {"flag": True}


def test_registry_lists_built_in_backends():
    names = list_backends()
    for expected in ("toy", "rimworld", "eve", "videogamebench", "pokemon_red", "zelda_minish_cap", "kiosk"):
        assert expected in names, expected


def test_observe_execute_reward_roundtrip():
    b = get_backend("toy")
    prev = b.reset()
    res = b.execute(Action("tick"))
    assert res.ok and res.result["ticks"] == 1
    curr = b.observe()
    breakdown = b.reward(prev, curr, [Action("tick")])
    assert breakdown["total"] == 1.0


def test_get_backend_unknown_raises():
    with pytest.raises(KeyError):
        get_backend("not_a_real_backend_xyz")
