"""Record gameplay episodes as JSON: per-step screenshots, structured game state, the
model's reasoning, its chosen actions, and the reward breakdown (project spec §5a).

The dataclasses here are the canonical on-disk schema; :mod:`rimworld_agent.training`
reads them back to build replay training examples, and the notebooks replay them. Pure
serialisation logic — fully covered by the offline test suite.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rimworld_agent.game.action_space import Action
from rimworld_agent.utils import ensure_dir, read_json, write_json


@dataclass
class Colonist:
    name: str
    mood: float
    health: float
    job: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class GameState:
    colonists: list[Colonist] = field(default_factory=list)
    resources: dict[str, float] = field(default_factory=dict)
    research_progress: dict[str, float] = field(default_factory=dict)
    threats: list[dict[str, Any]] = field(default_factory=list)
    quests_active: list[dict[str, Any]] = field(default_factory=list)
    season: str = ""
    day: int = 0
    temperature: float = 0.0
    wealth: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["colonists"] = [c.to_dict() for c in self.colonists]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GameState":
        d = dict(d)
        d["colonists"] = [Colonist(**c) for c in d.get("colonists", [])]
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Step:
    step: int
    game_tick: int = 0
    screenshots: list[str] = field(default_factory=list)
    game_state: GameState = field(default_factory=GameState)
    visible_rsids: list[str] = field(default_factory=list)  # READ SIDs: what the model sees
    action_wsids: list[str] = field(default_factory=list)  # WRITE SIDs: the workflow it plans
    reasoning: str = ""
    actions: list[Action] = field(default_factory=list)
    reward: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "game_tick": self.game_tick,
            "screenshots": list(self.screenshots),
            "game_state": self.game_state.to_dict(),
            "visible_rsids": list(self.visible_rsids),
            "action_wsids": list(self.action_wsids),
            "reasoning": self.reasoning,
            "actions": [a.to_dict() for a in self.actions],
            "reward": dict(self.reward),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Step":
        return cls(
            step=d["step"],
            game_tick=d.get("game_tick", 0),
            screenshots=list(d.get("screenshots", [])),
            game_state=GameState.from_dict(d.get("game_state", {})),
            visible_rsids=list(d.get("visible_rsids", [])),
            action_wsids=list(d.get("action_wsids", [])),
            reasoning=d.get("reasoning", ""),
            actions=[Action.from_dict(a) for a in d.get("actions", [])],
            reward=dict(d.get("reward", {})),
        )


@dataclass
class Episode:
    episode_id: str
    game_seed: str = ""
    scenario: str = "crashlanded"
    difficulty: str = "strive_to_survive"
    steps: list[Step] = field(default_factory=list)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def total_reward(self) -> float:
        return float(sum(s.reward.get("total", 0.0) for s in self.steps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "game_seed": self.game_seed,
            "scenario": self.scenario,
            "difficulty": self.difficulty,
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Episode":
        ep = cls(
            episode_id=d["episode_id"],
            game_seed=d.get("game_seed", ""),
            scenario=d.get("scenario", "crashlanded"),
            difficulty=d.get("difficulty", "strive_to_survive"),
        )
        ep.steps = [Step.from_dict(s) for s in d.get("steps", [])]
        return ep


class EpisodeRecorder:
    """Accumulate steps for one episode and persist them under ``<root>/<episode_id>/``."""

    def __init__(self, episode_id: str | None = None, root: str | Path = "data/episodes", **meta: Any):
        self.episode_id = episode_id or f"ep_{time.strftime('%Y%m%d_%H%M%S')}"
        self.root = Path(root)
        self.episode = Episode(episode_id=self.episode_id, **meta)
        self.dir = ensure_dir(self.root / self.episode_id)

    def frame_paths(self, step_index: int, n: int = 4) -> list[str]:
        """Relative paths the screenshot capture should write its 4 frames to."""
        base = f"{self.episode_id}/frame_{step_index:03d}"
        return [f"{base}_{i}.png" for i in range(n)]

    def record(
        self,
        game_state: GameState,
        reasoning: str,
        actions: list[Action],
        reward: dict[str, float],
        screenshots: list[str] | None = None,
        game_tick: int = 0,
        visible_rsids: list[str] | None = None,
        action_wsids: list[str] | None = None,
    ) -> Step:
        step = Step(
            step=len(self.episode.steps),
            game_tick=game_tick,
            screenshots=screenshots or [],
            game_state=game_state,
            visible_rsids=visible_rsids or [],
            action_wsids=action_wsids or [],
            reasoning=reasoning,
            actions=actions,
            reward=reward,
        )
        self.episode.steps.append(step)
        return step

    def save(self) -> Path:
        path = self.dir / "episode.json"
        write_json(self.episode.to_dict(), path)
        return path


def load_episode(path: str | Path) -> Episode:
    return Episode.from_dict(read_json(path))
