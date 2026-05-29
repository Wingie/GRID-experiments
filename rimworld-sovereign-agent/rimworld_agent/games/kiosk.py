"""Kiosk :class:`GameBackend` — a response-only verbal Q&A mode for "shop terminal" style
deployments. The agent answers customer questions about catalog items grounded in a JSON
catalog, **never asks** questions back, and does not mutate state.

This is the agent-as-a-service end of the multi-game protocol: there is no game-world to
play, only a catalog of items and a queue of incoming questions. The structural enforcement
of "responds only, never asks" is in the action space — there is no ``ask_clarification`` or
``request_more_info`` action; every action emits a final response. The same sovereign SLM
trained for RimWorld/EVE can be deployed here without retraining: RSIDs serve as canonical
addresses for items the response cites.

Verbal I/O (ASR in, TTS out) plugs in at the boundary; the backend itself is text-only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rimworld_agent.games.base import Action, ExecutionResult, Observation, register
from rimworld_agent.utils import get_logger

log = get_logger("backend.kiosk")


# --------------------------------------------------------------------------- #
# Action space — strictly response-only.
# --------------------------------------------------------------------------- #
KIOSK_ACTION_SPACE: dict[str, dict] = {
    # The four response actions. Every action emits a final answer; none solicit info.
    "answer":           {"type": "respond", "params": ["text", "cited_items"]},
    "lookup_price":     {"type": "respond", "params": ["item_id"]},
    "list_items":       {"type": "respond", "params": ["category"]},
    "recommend":        {"type": "respond", "params": ["item_id", "rationale"]},
    "inventory_query":  {"type": "respond", "params": ["item_id"]},
    "wait":             {"type": "noop"},
}


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
@dataclass
class CatalogItem:
    item_id: str
    name: str
    category: str = ""
    price: float = 0.0
    stock: int = 0
    rsid: str | None = None  # canonical READ-SID address (set when the dual pipeline ran)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Catalog:
    items: dict[str, CatalogItem] = field(default_factory=dict)  # item_id -> item

    def by_category(self, category: str) -> list[CatalogItem]:
        return [it for it in self.items.values() if it.category == category]

    def search(self, query: str) -> list[CatalogItem]:
        q = query.lower()
        return [it for it in self.items.values() if q in it.name.lower() or q in it.description.lower()]

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        p = Path(path)
        if not p.exists():
            log.warning("no catalog file at %s; kiosk starts empty", p)
            return cls()
        raw = json.loads(p.read_text())
        items = {it["item_id"]: CatalogItem(**it) for it in raw.get("items", [])}
        return cls(items=items)

    def lookup(self, item_id: str) -> CatalogItem | None:
        return self.items.get(item_id)


# --------------------------------------------------------------------------- #
# State + reward
# --------------------------------------------------------------------------- #
@dataclass
class KioskState:
    question: str = ""               # the current customer question (empty == idle)
    questions_queue: list[str] = field(default_factory=list)
    served: int = 0
    last_response: dict[str, Any] | None = None


def score_response(question: str, response: dict, catalog: Catalog) -> dict[str, float]:
    """Quality score for one response (offline reward shaping)."""
    text = str(response.get("text", "") or "")
    cited = list(response.get("cited_items") or [])
    item_id = response.get("item_id")
    if item_id and item_id not in cited:
        cited.append(item_id)
    valid_cites = [c for c in cited if catalog.lookup(c) is not None]
    asked = "?" in text  # responses must NOT contain questions back at the customer
    breakdown = {
        "served": 1.0,
        "cite_accuracy": (len(valid_cites) / len(cited)) if cited else 0.0,
        "grounded": 1.0 if valid_cites else 0.0,
        "asked_back_penalty": -2.0 if asked else 0.0,
        "empty_penalty": -1.0 if not text.strip() else 0.0,
    }
    breakdown["total"] = float(sum(breakdown.values()))
    return breakdown


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class KioskBackend:
    """Shop-terminal Q&A mode: respond-only over a static catalog."""

    name = "kiosk"

    def __init__(
        self,
        catalog_path: str | Path = "data/kiosk_catalog/catalog.json",
        questions_path: str | Path | None = None,
        data_root: str | Path = "data",
    ):
        self.catalog = Catalog.load(catalog_path)
        self.data_root = Path(data_root)
        self.state = KioskState()
        if questions_path and Path(questions_path).exists():
            self.state.questions_queue = [
                ln.strip() for ln in Path(questions_path).read_text().splitlines() if ln.strip()
            ]

    def action_space(self) -> dict[str, dict]:
        return dict(KIOSK_ACTION_SPACE)

    def knowledge_dirs(self) -> dict[str, Path]:
        return {"catalog": self.data_root / "kiosk_catalog"}

    def observe(self) -> Observation:
        # Pull the next question from the queue when idle.
        if not self.state.question and self.state.questions_queue:
            self.state.question = self.state.questions_queue.pop(0)
        return Observation(
            state=self.state,
            visible_entities=[it.item_id for it in self.catalog.items.values()][:32],
            metadata={
                "question": self.state.question,
                "catalog_size": len(self.catalog.items),
                "queue_depth": len(self.state.questions_queue),
            },
        )

    def execute(self, action: Action) -> ExecutionResult:
        spec = KIOSK_ACTION_SPACE.get(action.action)
        if spec is None:
            return ExecutionResult(ok=False, error=f"unknown kiosk action {action.action!r}")
        if spec["type"] == "noop":
            return ExecutionResult(ok=True, result={})

        # Build the response payload from the action's params. Structurally, the agent has no
        # way to ask the customer something back: there is no clarification action.
        response = self._build_response(action)
        # Belt-and-braces: strip any question marks the model might have emitted in the text.
        response["text"] = _strip_questions(response.get("text", ""))
        self.state.last_response = response
        self.state.served += 1
        self.state.question = ""  # done; pull the next question on the next observe()
        return ExecutionResult(ok=True, result=response)

    def _build_response(self, action: Action) -> dict[str, Any]:
        kind = action.action
        params = dict(action.params)
        if kind == "answer":
            return {"kind": "answer", "text": params.get("text", ""),
                    "cited_items": list(params.get("cited_items") or [])}
        if kind == "lookup_price":
            it = self.catalog.lookup(str(params.get("item_id", "")))
            if it is None:
                return {"kind": "lookup_price", "text": "Item not in catalog.", "cited_items": []}
            return {"kind": "lookup_price", "text": f"{it.name}: {it.price:.2f}.",
                    "cited_items": [it.item_id], "price": it.price}
        if kind == "list_items":
            cat = params.get("category", "")
            items = self.catalog.by_category(cat) or list(self.catalog.items.values())[:8]
            names = ", ".join(it.name for it in items[:8])
            return {"kind": "list_items", "text": f"{cat or 'Available'}: {names}.",
                    "cited_items": [it.item_id for it in items]}
        if kind == "inventory_query":
            it = self.catalog.lookup(str(params.get("item_id", "")))
            if it is None:
                return {"kind": "inventory_query", "text": "Not stocked.", "cited_items": []}
            return {"kind": "inventory_query",
                    "text": f"{it.name}: {it.stock} in stock.",
                    "cited_items": [it.item_id], "stock": it.stock}
        if kind == "recommend":
            it = self.catalog.lookup(str(params.get("item_id", "")))
            rationale = params.get("rationale", "")
            if it is None:
                return {"kind": "recommend", "text": "No recommendation available.", "cited_items": []}
            return {"kind": "recommend", "text": f"I recommend {it.name}. {rationale}".strip(),
                    "cited_items": [it.item_id]}
        return {"kind": kind, "text": "", "cited_items": []}

    def reward(self, prev: Observation, curr: Observation, actions: list[Action]) -> dict[str, float]:
        question = prev.metadata.get("question", "") if prev else ""
        response = curr.state.last_response or {}
        return score_response(question, response, self.catalog)

    def reset(self, seed: int | None = None) -> Observation:
        self.state = KioskState(questions_queue=list(self.state.questions_queue))
        return self.observe()

    def close(self) -> None:
        pass


_QUESTION_RE = re.compile(r"[?]+")


def _strip_questions(text: str) -> str:
    """Belt-and-braces: enforce the no-asking-back rule on the response text."""
    if not text:
        return text
    return _QUESTION_RE.sub(".", text)


@register("kiosk")
def _factory(**kwargs) -> KioskBackend:
    return KioskBackend(**kwargs)
