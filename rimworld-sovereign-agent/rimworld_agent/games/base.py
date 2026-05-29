"""Game-backend protocol and registry — the multi-game core.

A ``GameBackend`` is a uniform interface over a specific game (RimWorld via RIMAPI, EVE
Online via ESI, a VideoGameBench gym env, ...). Every backend exposes:
  * ``name``                                    -- short identifier (``"rimworld"``);
  * ``action_space() -> dict[str, dict]``       -- the finite action vocabulary;
  * ``observe() -> Observation``                -- current state + screenshots + visible entities;
  * ``execute(action) -> ExecutionResult``      -- run one action;
  * ``reward(prev, curr, actions) -> dict``     -- score the transition;
  * ``knowledge_dirs() -> dict[str, Path]``     -- where the offline knowledge lives;
  * ``reset(seed)`` / ``close()``               -- episode lifecycle.

The shared :class:`Action` from ``rimworld_agent.game.action_space`` is reused as the action
payload across backends; what *changes* per game is the action_space dict (vocabulary +
parameter schemas) and the underlying execute mechanism.

This module is pure types + a tiny registry; importable without any game dep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from rimworld_agent.game.action_space import Action

__all__ = ["Action", "Observation", "ExecutionResult", "GameBackend", "register", "get_backend", "list_backends"]


@dataclass
class Observation:
    """Generic observation: structured state + recent screenshots + perceived entities.

    ``state`` is backend-specific (a RimWorld ``GameState``, an EVE character snapshot, a VGB
    raw obs array, ...). The agent's mmproj/state encoder treats it opaquely.
    """

    state: Any
    screenshots: list[str] = field(default_factory=list)
    visible_entities: list[str] = field(default_factory=list)  # def_names / type ids
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    ok: bool
    error: str | None = None
    result: Any = None


@runtime_checkable
class GameBackend(Protocol):
    name: str

    def action_space(self) -> dict[str, dict]:
        ...

    def knowledge_dirs(self) -> dict[str, Path]:
        ...

    def observe(self) -> Observation:
        ...

    def execute(self, action: Action) -> ExecutionResult:
        ...

    def reward(self, prev: Observation, curr: Observation, actions: list[Action]) -> dict[str, float]:
        ...

    def reset(self, seed: int | None = None) -> Observation:
        ...

    def close(self) -> None:
        ...


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, Callable[..., GameBackend]] = {}


def register(name: str):
    """Decorator: register a backend factory under ``name`` (lazy — no game deps imported)."""

    def _decorator(factory: Callable[..., GameBackend]) -> Callable[..., GameBackend]:
        _REGISTRY[name] = factory
        return factory

    return _decorator


def get_backend(name: str, **kwargs) -> GameBackend:
    """Build the backend registered under ``name``. ``kwargs`` are forwarded to its factory."""
    if name not in _REGISTRY:
        # Trigger the backends' module import so their @register decorators run.
        _autoload()
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_backends() -> list[str]:
    _autoload()
    return sorted(_REGISTRY)


def _autoload() -> None:
    """Import the bundled backend modules so their factories self-register."""
    import importlib

    for mod in ("rimworld", "eve", "videogamebench", "kiosk"):
        try:
            importlib.import_module(f"rimworld_agent.games.{mod}")
        except Exception:
            # A missing optional dep on one backend should not break the others.
            pass
