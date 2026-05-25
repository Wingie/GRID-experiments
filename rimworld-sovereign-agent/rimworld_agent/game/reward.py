"""Compute the reward signal from a state transition + the actions taken (spec §5b).

Quest/todo progress is the primary objective; wealth, research, construction, and mood
are shaped secondaries; colonist death is a severe penalty; idleness and pure no-ops are
mildly penalised. Returns a breakdown dict (matching the episode schema) whose ``total``
is the scalar reward. Pure logic — fully covered by the offline test suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rimworld_agent.game.action_space import Action
from rimworld_agent.game.episode_recorder import GameState


@dataclass
class RewardWeights:
    quest_progress: float = 2.0
    wealth: float = 0.001
    research: float = 1.5
    construction: float = 0.5
    mood: float = 1.0
    death: float = -10.0
    idle: float = -0.1
    noop: float = -0.5
    invalid_action: float = -0.3


DEFAULT_WEIGHTS = RewardWeights()


def parse_progress(value) -> float:
    """Parse a quest progress value (``"1/3"``, ``"40%"``, ``0.4``) to a [0,1] fraction."""
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    s = str(value).strip()
    if s.endswith("%"):
        try:
            return max(0.0, min(1.0, float(s[:-1]) / 100.0))
        except ValueError:
            return 0.0
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$", s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        return num / den if den else 0.0
    try:
        return max(0.0, min(1.0, float(s)))
    except ValueError:
        return 0.0


def _quest_map(state: GameState) -> dict[str, float]:
    out: dict[str, float] = {}
    for q in state.quests_active:
        name = q.get("name", "")
        if name:
            out[name] = parse_progress(q.get("progress", 0.0))
    return out


def _avg_mood(state: GameState) -> float:
    if not state.colonists:
        return 0.0
    return sum(c.mood for c in state.colonists) / len(state.colonists)


def _count_idle(state: GameState) -> int:
    return sum(1 for c in state.colonists if str(c.job).strip().lower() in ("idle", "", "none"))


def _count_deaths(prev: GameState, curr: GameState) -> int:
    prev_names = {c.name for c in prev.colonists}
    curr_names = {c.name for c in curr.colonists}
    return len(prev_names - curr_names)


def _sum_research(state: GameState) -> float:
    return float(sum(state.research_progress.values()))


def _new_buildings(prev: GameState, curr: GameState) -> int:
    # Convention: RIMAPI reports a running building count under resources["buildings"].
    return max(0, int(curr.resources.get("buildings", 0)) - int(prev.resources.get("buildings", 0)))


def compute_reward(
    prev_state: GameState,
    curr_state: GameState,
    actions: list[Action],
    weights: RewardWeights = DEFAULT_WEIGHTS,
    invalid_action_count: int = 0,
) -> dict[str, float]:
    """Return a reward breakdown dict with a ``total`` key (spec §5b)."""
    prev_q = _quest_map(prev_state)
    curr_q = _quest_map(curr_state)
    quest_delta = sum(frac - prev_q.get(name, 0.0) for name, frac in curr_q.items())

    wealth_delta = curr_state.wealth - prev_state.wealth
    research_delta = _sum_research(curr_state) - _sum_research(prev_state)
    new_buildings = _new_buildings(prev_state, curr_state)
    mood_delta = _avg_mood(curr_state) - _avg_mood(prev_state)
    deaths = _count_deaths(prev_state, curr_state)
    idle = _count_idle(curr_state)

    breakdown = {
        "quest_progress": quest_delta * weights.quest_progress,
        "wealth_delta": wealth_delta * weights.wealth,
        "research_progress": research_delta * weights.research,
        "construction_progress": new_buildings * weights.construction,
        "colonist_mood_delta": mood_delta * weights.mood,
        "death_penalty": deaths * weights.death,
        "idle_penalty": idle * weights.idle,
        "invalid_action_penalty": invalid_action_count * weights.invalid_action,
        "resource_delta": float(
            sum(v for v in curr_state.resources.values() if isinstance(v, (int, float)))
            - sum(v for v in prev_state.resources.values() if isinstance(v, (int, float)))
        ),
    }
    if actions and all(a.action == "wait" for a in actions):
        breakdown["noop_penalty"] = weights.noop
    else:
        breakdown["noop_penalty"] = 0.0

    # resource_delta is reported for analysis but not summed into total (wealth captures it).
    breakdown["total"] = float(
        sum(v for k, v in breakdown.items() if k not in ("resource_delta",))
    )
    return breakdown
