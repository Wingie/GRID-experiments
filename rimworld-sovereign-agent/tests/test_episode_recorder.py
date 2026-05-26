"""Offline unit tests for the episode schema + recorder round-trip."""

from rimworld_agent.game.action_space import Action
from rimworld_agent.game.episode_recorder import (
    Colonist,
    Episode,
    EpisodeRecorder,
    GameState,
    load_episode,
)


def _state():
    return GameState(
        colonists=[Colonist("Engie", 0.65, 1.0, "Mining"), Colonist("Doc", 0.45, 1.0, "Idle")],
        resources={"steel": 120, "wood": 340},
        research_progress={"Electricity": 0.4},
        quests_active=[{"name": "Build 3 beds", "progress": "1/3"}],
        season="Spring",
        day=5,
        temperature=18.0,
        wealth=1000.0,
    )


def test_gamestate_roundtrip():
    s = _state()
    s2 = GameState.from_dict(s.to_dict())
    assert s2.day == 5 and s2.colonists[0].name == "Engie"
    assert s2.resources["steel"] == 120


def test_recorder_and_reload(tmp_path):
    rec = EpisodeRecorder(episode_id="ep_test_001", root=tmp_path, scenario="crashlanded")
    actions = [
        Action("order_build", {"def_name": "Bed", "x": 15, "y": 22}, "quest bed"),
        Action("speed_3x", reason="advance"),
    ]
    rec.record(
        _state(),
        "Doc idle, build beds.",
        actions,
        {"total": 0.55, "quest_progress": 0.3},
        visible_rsids=["<RSID_L3_01>", "<RSID_L2_05>"],
        action_wsids=["<WSID_L1_02>"],
    )
    path = rec.save()
    assert path.exists()

    ep = load_episode(path)
    assert isinstance(ep, Episode)
    assert ep.episode_id == "ep_test_001"
    assert ep.total_steps == 1
    assert abs(ep.total_reward - 0.55) < 1e-9
    step = ep.steps[0]
    assert step.actions[0].action == "order_build"
    assert step.actions[0].params["x"] == 15
    assert step.reasoning.startswith("Doc idle")
    # dual SID fields round-trip: RSIDs for perception, WSIDs for planning
    assert step.visible_rsids == ["<RSID_L3_01>", "<RSID_L2_05>"]
    assert step.action_wsids == ["<WSID_L1_02>"]


def test_frame_paths():
    rec = EpisodeRecorder(episode_id="ep_x", root="/tmp")
    paths = rec.frame_paths(3, n=4)
    assert paths == ["ep_x/frame_003_0.png", "ep_x/frame_003_1.png", "ep_x/frame_003_2.png", "ep_x/frame_003_3.png"]


def test_total_reward_accumulates(tmp_path):
    rec = EpisodeRecorder(episode_id="ep_acc", root=tmp_path)
    rec.record(_state(), "", [Action("wait")], {"total": 1.0})
    rec.record(_state(), "", [Action("wait")], {"total": 2.5})
    assert abs(rec.episode.total_reward - 3.5) < 1e-9
    assert rec.episode.total_steps == 2
