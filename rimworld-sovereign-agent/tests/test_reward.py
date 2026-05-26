"""Offline unit tests for the reward function (project spec §5b)."""

from rimworld_agent.game.action_space import Action
from rimworld_agent.game.episode_recorder import Colonist, GameState
from rimworld_agent.game.reward import RewardWeights, compute_reward, parse_progress


def test_parse_progress_forms():
    assert parse_progress("1/3") == 1 / 3
    assert parse_progress("40%") == 0.4
    assert parse_progress(0.7) == 0.7
    assert parse_progress("3/3") == 1.0
    assert parse_progress("garbage") == 0.0


def _state(**kw):
    base = dict(colonists=[Colonist("A", 0.6, 1.0, "Mining")], quests_active=[], research_progress={}, resources={}, wealth=0.0)
    base.update(kw)
    return GameState(**base)


def test_quest_progress_rewarded():
    prev = _state(quests_active=[{"name": "Build 3 beds", "progress": "1/3"}])
    curr = _state(quests_active=[{"name": "Build 3 beds", "progress": "2/3"}])
    r = compute_reward(prev, curr, [Action("wait")])
    # +1/3 quest delta * 2.0 weight
    assert abs(r["quest_progress"] - (1 / 3) * 2.0) < 1e-6


def test_death_penalty():
    prev = _state(colonists=[Colonist("A", 0.6, 1.0, "Mining"), Colonist("B", 0.6, 1.0, "Cooking")])
    curr = _state(colonists=[Colonist("A", 0.6, 1.0, "Mining")])
    r = compute_reward(prev, curr, [Action("speed_3x")])
    assert r["death_penalty"] == -10.0
    assert r["total"] < 0


def test_idle_and_noop_penalty():
    prev = _state(colonists=[Colonist("A", 0.6, 1.0, "Idle")])
    curr = _state(colonists=[Colonist("A", 0.6, 1.0, "Idle")])
    r = compute_reward(prev, curr, [Action("wait")])
    assert r["idle_penalty"] == -0.1
    assert r["noop_penalty"] == -0.5


def test_research_and_mood_and_invalid():
    prev = _state(research_progress={"Electricity": 0.2}, colonists=[Colonist("A", 0.5, 1.0, "Mining")])
    curr = _state(research_progress={"Electricity": 0.4}, colonists=[Colonist("A", 0.7, 1.0, "Mining")])
    r = compute_reward(prev, curr, [Action("order_research", {"project_def": "Electricity"})], invalid_action_count=1)
    assert abs(r["research_progress"] - 0.2 * 1.5) < 1e-6
    assert abs(r["colonist_mood_delta"] - 0.2 * 1.0) < 1e-6
    assert abs(r["invalid_action_penalty"] - (-0.3)) < 1e-6
    assert r["noop_penalty"] == 0.0  # not all actions are wait


def test_construction_from_building_count():
    prev = _state(resources={"buildings": 2})
    curr = _state(resources={"buildings": 4})
    r = compute_reward(prev, curr, [Action("order_build", {"def_name": "Bed", "x": 1, "y": 1})])
    assert r["construction_progress"] == 2 * 0.5


def test_custom_weights():
    w = RewardWeights(quest_progress=5.0)
    prev = _state(quests_active=[{"name": "q", "progress": "0/2"}])
    curr = _state(quests_active=[{"name": "q", "progress": "1/2"}])
    r = compute_reward(prev, curr, [Action("wait")], weights=w)
    assert abs(r["quest_progress"] - 0.5 * 5.0) < 1e-6
