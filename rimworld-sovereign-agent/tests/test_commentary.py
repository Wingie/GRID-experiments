"""Offline tests for the commentary wrapper: user Q&A on top of any game backend."""

import pytest

from rimworld_agent.games.base import Action, get_backend
from rimworld_agent.games.commentary import (
    CommentaryWrapper,
    FileQuestionSource,
    QueueQuestionSource,
    commentary_reward,
    wrap,
)

# Register the toy backend.
from tests import test_games_registry  # noqa: F401


def _wrapped():
    return wrap("toy", source=QueueQuestionSource())


def test_wrap_extends_action_space_with_say():
    w = _wrapped()
    space = w.action_space()
    assert "say" in space and "tick" in space  # gained `say`, kept the inner actions


def test_user_question_is_surfaced_in_observation():
    w = _wrapped()
    w.source.push("Why did you build a bed there?")
    obs = w.observe()
    assert obs.metadata["user_question"].startswith("Why")


def test_say_records_transcript_and_clears_question(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    w = CommentaryWrapper(get_backend("toy"), QueueQuestionSource(), transcript)
    w.source.push("What is that <RSID_L3_02>?")
    w.observe()
    r = w.execute(Action("say", {"text": "I see <RSID_L3_02> on the floor.", "cited_items": []}))
    assert r.ok and r.result["text"] == "I see <RSID_L3_02> on the floor."
    # Question was consumed.
    assert w.observe().metadata["user_question"] == ""
    # RSID found inline counts as a citation even though the agent didn't list it.
    assert "<RSID_L3_02>" in r.result["cited_items"]
    # Transcript is persisted.
    assert transcript.exists() and "<RSID_L3_02>" in transcript.read_text()


def test_question_marks_stripped_from_say():
    w = _wrapped()
    w.source.push("Got beds?")
    w.observe()
    r = w.execute(Action("say", {"text": "Yes I built one. Want more?"}))
    assert "?" not in r.result["text"]


def test_non_say_actions_pass_through_to_inner():
    w = _wrapped()
    prev = w.reset()
    w.execute(Action("tick"))
    curr = w.observe()
    # Inner reward (delta in ticks) still flows when no question is pending.
    r = w.reward(prev, curr, [Action("tick")])
    assert r["total"] == 1.0


def test_reward_adds_commentary_term_when_question_answered():
    w = _wrapped()
    w.source.push("What is that?")
    prev = w.observe()  # captures user_question
    w.execute(Action("tick"))
    w.execute(Action("say", {"text": "It is <RSID_L1_00>."}))
    curr = w.observe()
    breakdown = w.reward(prev, curr, [Action("tick"), Action("say", {"text": "It is <RSID_L1_00>."})])
    assert breakdown["commentary"] > 0
    # Gameplay reward (1 tick delta) is preserved.
    assert breakdown["total"] >= 1.0


def test_commentary_reward_penalises_asking_back_and_empty():
    good = {"text": "Yes, it's <RSID_L2_03>.", "cited_items": []}
    asked = {"text": "Which item do you mean?"}
    empty = {"text": ""}
    sg = commentary_reward("what?", good)
    sa = commentary_reward("what?", asked)
    se = commentary_reward("what?", empty)
    assert sg["total"] > sa["total"]    # grounded answer beats asking back
    assert sg["total"] > se["total"]    # grounded answer beats silence
    assert sa["asked_back_penalty"] == -2.0
    assert se["answered"] == -1.0


def test_file_question_source_tails_appended_lines(tmp_path):
    path = tmp_path / "q.txt"
    src = FileQuestionSource(path=path)
    assert src.next() is None      # empty file
    src.push("first?")
    src.push("second.")
    assert src.next() == "first?"
    assert src.next() == "second."
    assert src.next() is None      # caught up
    src.push("third!")
    assert src.next() == "third!"  # continues from offset


def test_wrapper_name_reports_composition():
    w = _wrapped()
    assert w.name == "toy+commentary"


def test_reset_clears_transcript_and_question_state():
    w = _wrapped()
    w.source.push("q?")
    w.observe()
    w.execute(Action("say", {"text": "answer."}))
    assert w.transcript
    w.reset()
    assert not w.transcript and w.current_question is None
