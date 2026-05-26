"""REST client for the RIMAPI RimWorld mod: read game state, capture screenshots, send
key events, and execute parameterised orders.

RIMAPI exposes a small HTTP server inside the running game (project spec §12, gotcha #1).
This client is exercised only against a live game (or the mock in the tests that target
the request-building logic); it is not part of the offline-runnable subset.
"""

from __future__ import annotations

from typing import Any, Iterator

import requests

from rimworld_agent.game.action_space import ACTION_SPACE, Action
from rimworld_agent.game.episode_recorder import Colonist, GameState
from rimworld_agent.utils import get_logger

log = get_logger("rimapi_client")


class RimAPIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7860", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    # --- low level ---------------------------------------------------------- #
    def _get(self, path: str, **params: Any) -> Any:
        r = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        r = self.session.post(self.base_url + path, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.content else {}

    # --- observation -------------------------------------------------------- #
    def get_state(self) -> GameState:
        """Fetch and parse the structured colony state."""
        raw = self._get("/state")
        return self._parse_state(raw)

    @staticmethod
    def _parse_state(raw: dict[str, Any]) -> GameState:
        colonists = [
            Colonist(
                name=c.get("name", "?"),
                mood=float(c.get("mood", 0.0)),
                health=float(c.get("health", 1.0)),
                job=c.get("job", "Idle"),
            )
            for c in raw.get("colonists", [])
        ]
        return GameState(
            colonists=colonists,
            resources=dict(raw.get("resources", {})),
            research_progress=dict(raw.get("research_progress", {})),
            threats=list(raw.get("threats", [])),
            quests_active=list(raw.get("quests_active", [])),
            season=raw.get("season", ""),
            day=int(raw.get("day", 0)),
            temperature=float(raw.get("temperature", 0.0)),
            wealth=float(raw.get("wealth", 0.0)),
        )

    def screenshot(self, path: str | None = None) -> bytes:
        """Return PNG bytes from RIMAPI's screenshot endpoint (optionally save to ``path``)."""
        r = self.session.get(self.base_url + "/screenshot", timeout=self.timeout)
        r.raise_for_status()
        if path:
            with open(path, "wb") as fh:
                fh.write(r.content)
        return r.content

    def visible_entities(self) -> list[str]:
        """defNames currently visible on screen (used to attach SIDs to the prompt)."""
        return list(self._get("/visible").get("def_names", []))

    # --- actions ------------------------------------------------------------ #
    def send_key(self, key: str) -> Any:
        return self._post("/key", {"key": key})

    def click(self, x: int, y: int, button: str = "left") -> Any:
        return self._post("/click", {"x": x, "y": y, "button": button})

    def execute(self, action: Action) -> dict[str, Any]:
        """Route an :class:`Action` to the correct RIMAPI call.

        Returns ``{"ok": bool, "error": str | None, "result": Any}``. Invalid/rejected
        actions surface ``ok=False`` so the game loop can apply the invalid-action penalty.
        """
        valid, msg = action.is_valid()
        if not valid:
            return {"ok": False, "error": msg, "result": None}
        spec = ACTION_SPACE[action.action]
        kind = spec.get("type")
        try:
            if kind == "key":
                return {"ok": True, "error": None, "result": self.send_key(spec["key"])}
            if kind in ("click", "rclick"):
                btn = "right" if kind == "rclick" else "left"
                return {"ok": True, "error": None, "result": self.click(action.params["x"], action.params["y"], btn)}
            if kind == "api":
                payload = {"action": action.action, **action.params}
                return {"ok": True, "error": None, "result": self._post(spec["endpoint"], payload)}
            if kind == "noop":
                return {"ok": True, "error": None, "result": {}}
            if kind in ("submenu", "drag"):
                payload = {"action": action.action, **spec, **action.params}
                return {"ok": True, "error": None, "result": self._post("/ui", payload)}
        except requests.RequestException as exc:
            return {"ok": False, "error": str(exc), "result": None}
        return {"ok": False, "error": f"unhandled action type {kind!r}", "result": None}

    # --- events (SSE / websocket) ------------------------------------------- #
    def events(self) -> Iterator[dict[str, Any]]:
        """Yield server-sent game events (raids, quest completion, deaths) as dicts."""
        with self.session.get(self.base_url + "/events", stream=True, timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    import json

                    yield json.loads(line[len("data:") :].strip())
