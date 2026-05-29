"""RimWorld :class:`GameBackend` — thin adapter over the existing RIMAPI client + action
space + reward function. Keeps the original modules as the canonical implementation; this
file just plugs them into the multi-game protocol.
"""

from __future__ import annotations

from pathlib import Path

from rimworld_agent.game.action_space import ACTION_SPACE, Action
from rimworld_agent.game.episode_recorder import GameState
from rimworld_agent.game.reward import DEFAULT_WEIGHTS, RewardWeights, compute_reward
from rimworld_agent.game.rimapi_client import RimAPIClient
from rimworld_agent.games.base import ExecutionResult, Observation, register
from rimworld_agent.utils import get_logger

log = get_logger("backend.rimworld")


class RimWorldBackend:
    name = "rimworld"

    def __init__(
        self,
        rimapi_url: str = "http://127.0.0.1:7860",
        data_root: str | Path = "data",
        reward_weights: RewardWeights = DEFAULT_WEIGHTS,
    ):
        self.client = RimAPIClient(rimapi_url)
        self.data_root = Path(data_root)
        self.reward_weights = reward_weights

    def action_space(self) -> dict[str, dict]:
        return dict(ACTION_SPACE)

    def knowledge_dirs(self) -> dict[str, Path]:
        return {
            "xml_defs": self.data_root / "rimworld_xml_defs",
            "csharp_source": self.data_root / "rimworld_source",
            "wiki": self.data_root / "rimworld_wiki",
        }

    def observe(self) -> Observation:
        state = self.client.get_state()
        visible = self.client.visible_entities()
        return Observation(state=state, visible_entities=visible)

    def execute(self, action: Action) -> ExecutionResult:
        r = self.client.execute(action)
        return ExecutionResult(ok=r["ok"], error=r["error"], result=r["result"])

    def reward(self, prev: Observation, curr: Observation, actions: list[Action]) -> dict[str, float]:
        return compute_reward(prev.state, curr.state, actions, self.reward_weights)

    def reset(self, seed: int | None = None) -> Observation:
        # RimWorld does not support API-level reset; the user must load a save manually.
        log.warning("RimWorldBackend.reset() is a no-op; load a save in-game.")
        return self.observe()

    def close(self) -> None:
        self.client.session.close()

    # Helpful default state for tests / cold start.
    @staticmethod
    def empty_state() -> GameState:
        return GameState()


@register("rimworld")
def _factory(**kwargs) -> RimWorldBackend:
    return RimWorldBackend(**kwargs)
