"""The main game-interaction loop: observe (4 frames + state) -> reason -> act -> reward.

A ``Policy`` maps an :class:`Observation` to ``(reasoning, actions)``. The loop executes
the actions via RIMAPI, captures the next state, scores the transition with
:func:`rimworld_agent.game.reward.compute_reward`, and records the step. Invalid/rejected
actions incur the −0.3 penalty (gotcha #5). During self-play the caller runs the game at
3x (gotcha #3); during mmproj data collection at 1x for clean frames.

Live-only: needs a running game + RIMAPI. The function signatures are import-safe so the
training code can type against them without a game present.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from rimworld_agent.game.action_space import MAX_ACTIONS_PER_TURN, Action
from rimworld_agent.game.episode_recorder import EpisodeRecorder, GameState
from rimworld_agent.game.reward import DEFAULT_WEIGHTS, RewardWeights, compute_reward
from rimworld_agent.game.rimapi_client import RimAPIClient
from rimworld_agent.utils import get_logger

log = get_logger("game_loop")


@dataclass
class Observation:
    game_state: GameState
    screenshots: list[str] = field(default_factory=list)  # paths to the last 4 frames
    visible_sids: list[str] = field(default_factory=list)


class Policy(Protocol):
    def __call__(self, obs: Observation) -> tuple[str, list[Action]]:
        """Return (reasoning, up-to-5 actions) for an observation."""
        ...


def _capture_frames(client: RimAPIClient, recorder: EpisodeRecorder, step_index: int, n: int = 4) -> list[str]:
    rels = recorder.frame_paths(step_index, n)
    paths = []
    for rel in rels:
        abs_path = recorder.root / rel
        client.screenshot(str(abs_path))
        paths.append(rel)
    return paths


def play_episode(
    client: RimAPIClient,
    policy: Policy,
    recorder: EpisodeRecorder,
    max_steps: int = 150,
    weights: RewardWeights = DEFAULT_WEIGHTS,
    frame_interval: float = 5.0,
    sids_for: Callable[[list[str]], list[str]] | None = None,
) -> EpisodeRecorder:
    """Run one episode to completion (or ``max_steps``) and return the recorder."""
    prev_state = client.get_state()
    for step_index in range(max_steps):
        screenshots = _capture_frames(client, recorder, step_index)
        visible = client.visible_entities()
        visible_sids = sids_for(visible) if sids_for else []
        obs = Observation(game_state=prev_state, screenshots=screenshots, visible_sids=visible_sids)

        reasoning, actions = policy(obs)
        actions = actions[:MAX_ACTIONS_PER_TURN]

        invalid = 0
        for action in actions:
            result = client.execute(action)
            if not result["ok"]:
                invalid += 1
                log.info("invalid action %s: %s", action.action, result["error"])

        time.sleep(frame_interval)  # let the game advance before re-observing
        curr_state = client.get_state()
        reward = compute_reward(prev_state, curr_state, actions, weights, invalid_action_count=invalid)
        recorder.record(
            game_state=prev_state,
            reasoning=reasoning,
            actions=actions,
            reward=reward,
            screenshots=screenshots,
            visible_sids=visible_sids,
        )
        prev_state = curr_state
        if _is_terminal(curr_state):
            log.info("episode terminal at step %d", step_index)
            break

    recorder.save()
    log.info("episode %s done: %d steps, reward=%.2f", recorder.episode_id,
             recorder.episode.total_steps, recorder.episode.total_reward)
    return recorder


def _is_terminal(state: GameState) -> bool:
    """All colonists dead = colony lost."""
    return len(state.colonists) == 0
