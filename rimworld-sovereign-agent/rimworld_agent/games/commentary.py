"""Commentary wrapper: let the user ask questions while the agent plays.

Wrap any :class:`GameBackend` with :class:`CommentaryWrapper` to add a parallel "user Q&A"
channel without changing the underlying game. Each `observe()` checks the question source
(an in-memory queue, a tailed file, a websocket, ...) and surfaces a pending user question
in `Observation.metadata["user_question"]`. The wrapped action space gains a single new
action, `say`, which records a response to the transcript without touching the game state.

The response-only contract from the kiosk applies here too — there is no `ask_clarification`
action, and a question-mark stripper sanitises the `say` text before it is recorded. The
reward function delegates to the inner backend for gameplay and adds a `commentary` term
that rewards grounded RSID citations and penalises any '?' that survives.

Verbal I/O (ASR in, TTS out) plugs in at the question source / transcript boundary; the
wrapper itself is text-only.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rimworld_agent.games.base import Action, ExecutionResult, GameBackend, Observation
from rimworld_agent.games.kiosk import _strip_questions
from rimworld_agent.utils import get_logger

log = get_logger("backend.commentary")

_RSID_RE = re.compile(r"<RSID_L\d+_\d+>")


# --------------------------------------------------------------------------- #
# Where user questions come from
# --------------------------------------------------------------------------- #
class QuestionSource(Protocol):
    def next(self) -> str | None: ...
    def push(self, question: str) -> None: ...


@dataclass
class QueueQuestionSource:
    """FIFO in-memory queue; ``push`` is how a UI / mic / websocket feeds in live questions."""

    questions: deque = field(default_factory=deque)

    def next(self) -> str | None:
        return self.questions.popleft() if self.questions else None

    def push(self, question: str) -> None:
        if question and question.strip():
            self.questions.append(question.strip())


@dataclass
class FileQuestionSource:
    """Tail a text file (one question per line). Each line is consumed once."""

    path: Path
    _offset: int = 0

    def next(self) -> str | None:
        p = Path(self.path)
        if not p.exists():
            return None
        with open(p) as fh:
            fh.seek(self._offset)
            line = fh.readline()
            if not line:
                return None
            self._offset = fh.tell()
        line = line.strip()
        return line or None

    def push(self, question: str) -> None:
        with open(self.path, "a") as fh:
            fh.write(question.rstrip("\n") + "\n")


# --------------------------------------------------------------------------- #
# Reward shaping for the commentary side
# --------------------------------------------------------------------------- #
def commentary_reward(question: str, response: dict) -> dict[str, float]:
    text = str(response.get("text", "") or "")
    cited = list(response.get("cited_items") or [])
    breakdown = {
        "answered": 1.0 if text.strip() else -1.0,
        "rsid_grounded": 1.0 if (_RSID_RE.search(text) or cited) else 0.0,
        "asked_back_penalty": -2.0 if "?" in text else 0.0,
    }
    breakdown["total"] = float(sum(breakdown.values()))
    return breakdown


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #
SAY_ACTION = {"type": "respond", "params": ["text", "cited_items"]}


class CommentaryWrapper:
    """Wrap a backend to expose a user-question channel and a ``say`` action."""

    def __init__(
        self,
        inner: GameBackend,
        source: QuestionSource | None = None,
        transcript_path: str | Path | None = None,
        on_say=None,
    ):
        """``on_say(entry)`` fires after every recorded response — used to bridge the
        running agent into a live presentation server (or any other live consumer).
        """
        self.inner = inner
        self.source: QuestionSource = source or QueueQuestionSource()
        self.transcript_path = Path(transcript_path) if transcript_path else None
        self.transcript: list[dict] = []
        self.current_question: str | None = None
        self.last_response: dict | None = None
        self.last_question: str = ""
        self.on_say = on_say

    @property
    def name(self) -> str:
        return f"{self.inner.name}+commentary"

    def action_space(self) -> dict[str, dict]:
        return {**self.inner.action_space(), "say": SAY_ACTION}

    def knowledge_dirs(self) -> dict[str, Path]:
        return self.inner.knowledge_dirs()

    def observe(self) -> Observation:
        obs = self.inner.observe()
        if self.current_question is None:
            self.current_question = self.source.next()
        obs.metadata = {**(obs.metadata or {}), "user_question": self.current_question or ""}
        return obs

    def execute(self, action: Action) -> ExecutionResult:
        if action.action != "say":
            return self.inner.execute(action)
        text = _strip_questions(str(action.params.get("text", "") or ""))
        cited = list(action.params.get("cited_items") or [])
        # any RSIDs mentioned in the text are implicit citations too
        cited = sorted(set(cited) | set(_RSID_RE.findall(text)))
        response = {"text": text, "cited_items": cited, "kind": "say"}
        entry = {
            "ts": time.time(),
            "question": self.current_question or "",
            "response": response,
        }
        self.transcript.append(entry)
        if self.transcript_path is not None:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.transcript_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        self.last_question = self.current_question or ""
        self.last_response = response
        self.current_question = None  # done; observe() pulls the next one
        if self.on_say is not None:
            try:
                self.on_say(entry)
            except Exception as exc:  # never fail the game loop on a notify error
                log.warning("on_say notifier raised %s", exc)
        return ExecutionResult(ok=True, result=response)

    def reward(self, prev: Observation, curr: Observation, actions: list[Action]) -> dict[str, float]:
        say_actions = [a for a in actions if a.action == "say"]
        game_actions = [a for a in actions if a.action != "say"]
        game_reward = self.inner.reward(prev, curr, game_actions)
        if say_actions and (self.last_question or prev.metadata.get("user_question")):
            cr = commentary_reward(self.last_question or prev.metadata.get("user_question", ""),
                                   self.last_response or {})
            game_reward["commentary"] = cr["total"]
            game_reward["commentary_breakdown"] = cr  # type: ignore[assignment]
            game_reward["total"] = float(game_reward.get("total", 0.0)) + cr["total"]
        return game_reward

    def reset(self, seed: int | None = None) -> Observation:
        self.transcript.clear()
        self.current_question = None
        self.last_response = None
        return self.inner.reset(seed=seed)

    def close(self) -> None:
        self.inner.close()


def wrap(inner_name: str, source: QuestionSource | None = None,
         transcript_path: str | Path | None = None, **inner_kwargs) -> CommentaryWrapper:
    """Convenience constructor: ``wrap("rimworld", source=..., transcript_path=...)``."""
    from rimworld_agent.games.base import get_backend

    return CommentaryWrapper(get_backend(inner_name, **inner_kwargs), source, transcript_path)
