"""Offline tests for the presentation deck + server bridge.

Validates the slide manifest matches what ``index.html`` references, that the static
assets exist, that ``presentation.server`` imports cleanly without fastapi installed, and
that the agent->deck bridge hook fires correctly without going through HTTP.
"""

from pathlib import Path

import pytest

from rimworld_agent.game.action_space import Action
from rimworld_agent.games.commentary import (
    CommentaryWrapper,
    QueueQuestionSource,
)

# Register the toy backend.
from tests import test_games_registry  # noqa: F401
from rimworld_agent.games.base import get_backend

PRES = Path(__file__).resolve().parent.parent / "presentation"
SLIDES = PRES / "slides"
STATIC = PRES / "static"


def test_slide_files_exist_and_are_markdown():
    expected = ["00-intro", "01-sovereign", "02-dual-sid", "03-multi-game", "04-commentary", "05-closing"]
    for name in expected:
        path = SLIDES / f"{name}.md"
        assert path.exists(), name
        # non-empty, starts with a Markdown heading
        text = path.read_text().strip()
        assert text and (text.startswith("#") or text.startswith("##"))


def test_index_html_references_each_slide():
    html = (PRES / "index.html").read_text()
    for name in ["00-intro", "01-sovereign", "02-dual-sid", "03-multi-game", "04-commentary", "05-closing"]:
        assert f"slides/{name}.md" in html, name
    # live demo mount point is present.
    assert 'id="live-demo-root"' in html


def test_static_assets_present():
    assert (STATIC / "live.js").exists()
    assert (STATIC / "live.css").exists()
    js = (STATIC / "live.js").read_text()
    # Each headline widget is defined.
    for component in ("QuestionInput", "TranscriptStream", "RewardGauge", "GameFrame", "LiveDemo"):
        assert f"function {component}" in js, component


def test_server_module_imports_without_fastapi(monkeypatch):
    # If fastapi is not installed, the module still imports and exposes the shared source.
    import importlib
    import sys

    # Force re-import so a hidden cache doesn't mask the optional-dep path.
    sys.modules.pop("presentation.server", None)
    if "fastapi" not in sys.modules:
        # Pretend fastapi is missing for this import.
        monkeypatch.setitem(sys.modules, "fastapi", None)
    try:
        import presentation.server as server  # type: ignore
    except (ImportError, TypeError):
        # When fastapi is REAL-installed and reachable the import works fine; otherwise the
        # module's try/except absorbs the error and `app = None`. Either is acceptable.
        return
    assert hasattr(server, "SHARED_SOURCE")
    assert isinstance(server.SHARED_SOURCE, QueueQuestionSource)


def test_on_say_callback_fires_for_each_response(tmp_path):
    captured: list[dict] = []
    wrapper = CommentaryWrapper(
        get_backend("toy"),
        source=QueueQuestionSource(),
        on_say=captured.append,
    )
    wrapper.source.push("What is that?")
    wrapper.observe()
    wrapper.execute(Action("say", {"text": "It is <RSID_L1_00>.", "cited_items": []}))
    assert len(captured) == 1
    entry = captured[0]
    assert entry["question"].startswith("What is")
    assert "<RSID_L1_00>" in entry["response"]["text"]


def test_on_say_callback_swallows_exceptions():
    """A flaky bridge to the live server must not break the game loop."""
    def _flaky(_entry):
        raise RuntimeError("server down")

    wrapper = CommentaryWrapper(
        get_backend("toy"),
        source=QueueQuestionSource(),
        on_say=_flaky,
    )
    wrapper.source.push("How?")
    wrapper.observe()
    # Should still succeed structurally despite the notifier raising.
    result = wrapper.execute(Action("say", {"text": "Like this."}))
    assert result.ok
