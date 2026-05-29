"""VideoGameBench :class:`GameBackend` — adapt an emulator-backed VGB gym env (Pokémon,
*The Legend of Zelda: The Minish Cap*, etc.) to the multi-game protocol.

VideoGameBench ships gym-like environments wrapping Game Boy / GBA / DOS emulators. We treat
each game as one backend instance keyed by ``game``. Two headline targets are exposed as
presets:

  * ``pokemon_red``               — Pokémon Red (GB) — the canonical LLM/VLM gameplay eval;
  * ``zelda_minish_cap``          — The Legend of Zelda: The Minish Cap (GBA).

Other VGB games (Pokémon Crystal, Donkey Kong Country, Doom, etc.) are usable by passing
their VGB game id directly via the ``game`` factory kwarg.

Requires the ``videogamebench`` extra. The factory degrades gracefully (clear error) when
the package isn't installed.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rimworld_agent.games.base import Action, ExecutionResult, Observation, register
from rimworld_agent.utils import get_logger

log = get_logger("backend.videogamebench")

GAME_POKEMON_RED = "pokemon_red"
GAME_POKEMON_CRYSTAL = "pokemon_crystal"
GAME_ZELDA_MINISH_CAP = "zelda_minish_cap"
GAME_ZELDA_LINKS_AWAKENING = "zelda_links_awakening"

# Headline targets the README + benchmark default to (capped per user spec).
HEADLINE_GAMES: tuple[str, ...] = (GAME_POKEMON_RED, GAME_ZELDA_MINISH_CAP)


# Standard GB/GBA button layout. The button -> env-action-index mapping is filled in from
# the env's spec at construction time (different VGB envs use different orderings).
_BUTTONS = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT", "L", "R")


def _action_space_for_buttons(buttons: tuple[str, ...]) -> dict[str, dict]:
    return {f"press_{b.lower()}": {"type": "button", "button": b} for b in buttons} | {
        "wait": {"type": "noop"},
    }


@dataclass
class VGBState:
    """Snapshot a VGB env returns each step (raw image + the env's structured info)."""

    frame_path: str        # PNG path the latest rendered frame was written to
    info: dict[str, Any]   # whatever info dict the env emitted (lives, score, progress flags)
    last_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False


class VideoGameBenchBackend:
    name = "videogamebench"

    def __init__(self, game: str = GAME_POKEMON_RED, render_mode: str = "rgb_array",
                 frame_dir: str | Path | None = None, **env_kwargs):
        try:
            import videogamebench as vgb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "VideoGameBench backend needs `pip install videogamebench` "
                "(the optional `videogamebench` extra)."
            ) from exc

        self.game = game
        self.env = vgb.make(game, render_mode=render_mode, **env_kwargs)
        self.frame_dir = Path(frame_dir) if frame_dir else Path(tempfile.mkdtemp(prefix=f"vgb_{game}_"))
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self._frame_idx = 0
        self._buttons = self._detect_buttons()
        self._button_to_action = self._map_buttons_to_actions()
        self._state = VGBState(frame_path="", info={})

    def _detect_buttons(self) -> tuple[str, ...]:
        meta = getattr(self.env, "buttons", None) or getattr(self.env.unwrapped, "buttons", None)
        return tuple(meta) if meta else _BUTTONS

    def _map_buttons_to_actions(self) -> dict[str, int]:
        meta = getattr(self.env, "button_to_action", None) or getattr(self.env.unwrapped, "button_to_action", None)
        if isinstance(meta, dict):
            return {b: int(meta[b]) for b in meta}
        # Fallback: assume action space is Discrete(N) ordered as ``self._buttons``.
        return {b: i for i, b in enumerate(self._buttons)}

    def action_space(self) -> dict[str, dict]:
        return _action_space_for_buttons(self._buttons)

    def knowledge_dirs(self) -> dict[str, Path]:
        return {"roms": Path("data/videogamebench") / self.game}

    def _save_frame(self) -> str:
        from PIL import Image

        arr = self.env.render()
        path = self.frame_dir / f"frame_{self._frame_idx:06d}.png"
        Image.fromarray(arr).save(path)
        self._frame_idx += 1
        return str(path)

    def observe(self) -> Observation:
        frame = self._save_frame()
        self._state.frame_path = frame
        return Observation(
            state=self._state,
            screenshots=[frame],
            metadata={"game": self.game, "info": dict(self._state.info)},
        )

    def execute(self, action: Action) -> ExecutionResult:
        spec = self.action_space().get(action.action)
        if spec is None:
            return ExecutionResult(ok=False, error=f"unknown VGB action {action.action!r}")
        if spec["type"] == "noop":
            obs, reward, terminated, truncated, info = self.env.step(self._noop_action())
        else:
            idx = self._button_to_action.get(spec["button"])
            if idx is None:
                return ExecutionResult(ok=False, error=f"button {spec['button']!r} not in env")
            obs, reward, terminated, truncated, info = self.env.step(idx)
        self._state = VGBState(frame_path=self._state.frame_path, info=dict(info),
                                last_reward=float(reward), terminated=bool(terminated),
                                truncated=bool(truncated))
        return ExecutionResult(ok=True, result={"reward": float(reward), "terminated": bool(terminated)})

    def _noop_action(self) -> int:
        return self._button_to_action.get("NOOP", 0)

    def reward(self, prev: Observation, curr: Observation, actions: list[Action]) -> dict[str, float]:
        # VGB envs already shape per-step reward; we expose it raw + a progress proxy.
        s: VGBState = curr.state
        breakdown = {
            "env_reward": s.last_reward,
            "terminated_bonus": 1.0 if s.terminated else 0.0,
            "progress": float(s.info.get("progress", 0.0)),
        }
        breakdown["total"] = float(sum(breakdown.values()))
        return breakdown

    def reset(self, seed: int | None = None) -> Observation:
        obs, info = self.env.reset(seed=seed)
        self._frame_idx = 0
        self._state = VGBState(frame_path="", info=dict(info))
        return self.observe()

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


@register("videogamebench")
def _factory(**kwargs) -> VideoGameBenchBackend:
    return VideoGameBenchBackend(**kwargs)


# Convenience factories for the headline games (also discoverable via the registry).
@register("pokemon_red")
def _pokemon_red(**kwargs) -> VideoGameBenchBackend:
    kwargs.setdefault("game", GAME_POKEMON_RED)
    return VideoGameBenchBackend(**kwargs)


@register("zelda_minish_cap")
def _zelda_minish_cap(**kwargs) -> VideoGameBenchBackend:
    kwargs.setdefault("game", GAME_ZELDA_MINISH_CAP)
    return VideoGameBenchBackend(**kwargs)
