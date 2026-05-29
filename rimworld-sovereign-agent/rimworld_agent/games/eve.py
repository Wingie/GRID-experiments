"""EVE Online :class:`GameBackend` — ESI REST client + SDE knowledge parser + action space.

EVE is unusual: most "playing" happens in-client and isn't exposed to the REST API. The
ESI-actionable subset — skill queue, market orders, contracts, navigation waypoints, mail —
makes EVE a great testbed for the *industry/economy* axis of the sovereign-agent recipe
(reading code = SDE/types, writing code = market+skill workflows). The same dual-SID idea
applies: READ SIDs cluster types by taxonomy (frigates, modules, ore), WRITE SIDs cluster
them by workflow (mining loop, T2 manufacturing chain).

Knowledge sources:
  * **SDE** (Static Data Export) YAMLs under ``data/eve_sde/`` — types, groups,
    marketGroups, blueprints, skills, dogma attributes;
  * **ESI** public endpoints — universe constants, market prices.

Auth: character-scoped endpoints need an OAuth2 access token (set ``$EVE_ACCESS_TOKEN``).
The backend degrades to read-only / public operations when no token is configured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from rimworld_agent.games.base import Action, ExecutionResult, Observation, register
from rimworld_agent.utils import get_logger

log = get_logger("backend.eve")

ESI_BASE = "https://esi.evetech.net/latest"

# --------------------------------------------------------------------------- #
# Action space — REST-actionable subset (project spec analogue to RimWorld §4a).
# --------------------------------------------------------------------------- #
EVE_ACTION_SPACE: dict[str, dict] = {
    # Skill queue
    "skill_train": {"type": "api", "endpoint": "/characters/{cid}/skillqueue", "method": "POST",
                    "params": ["skill_id", "level"], "auth": True},
    "skill_cancel": {"type": "api", "endpoint": "/characters/{cid}/skillqueue", "method": "DELETE",
                     "params": ["queue_position"], "auth": True},
    # Market
    "order_buy": {"type": "api", "endpoint": "/characters/{cid}/orders", "method": "POST",
                  "params": ["type_id", "price", "quantity", "location_id", "duration_days"], "auth": True},
    "order_sell": {"type": "api", "endpoint": "/characters/{cid}/orders", "method": "POST",
                   "params": ["type_id", "price", "quantity", "location_id", "duration_days"], "auth": True},
    "order_cancel": {"type": "api", "endpoint": "/characters/{cid}/orders/{order_id}", "method": "DELETE",
                     "params": ["order_id"], "auth": True},
    # Navigation
    "set_destination": {"type": "api", "endpoint": "/ui/autopilot/waypoint", "method": "POST",
                        "params": ["destination_id", "clear_other_waypoints"], "auth": True},
    "add_waypoint": {"type": "api", "endpoint": "/ui/autopilot/waypoint", "method": "POST",
                     "params": ["destination_id"], "auth": True},
    # Contracts
    "accept_contract": {"type": "api", "endpoint": "/characters/{cid}/contracts/{contract_id}/bids",
                        "method": "POST", "params": ["contract_id", "bid"], "auth": True},
    # Mail
    "send_mail": {"type": "api", "endpoint": "/characters/{cid}/mail", "method": "POST",
                  "params": ["subject", "body", "recipients"], "auth": True},
    # Misc
    "wait": {"type": "noop"},
}


# --------------------------------------------------------------------------- #
# ESI client (thin)
# --------------------------------------------------------------------------- #
class ESIClient:
    def __init__(self, character_id: int | None = None, access_token: str | None = None,
                 base_url: str = ESI_BASE, timeout: float = 20.0):
        self.character_id = character_id or int(os.environ.get("EVE_CHARACTER_ID", "0"))
        self.access_token = access_token or os.environ.get("EVE_ACCESS_TOKEN", "")
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "rimworld-sovereign-agent/0.1 (eve backend)"

    @property
    def authed(self) -> bool:
        return bool(self.access_token and self.character_id)

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}

    def request(self, method: str, path: str, **kwargs) -> Any:
        url = self.base + path.replace("{cid}", str(self.character_id))
        kwargs.setdefault("headers", {}).update(self._auth_headers())
        r = self.session.request(method, url, timeout=self.timeout, **kwargs)
        r.raise_for_status()
        return r.json() if r.content else {}

    # Common reads
    def wallet_balance(self) -> float:
        if not self.authed:
            return 0.0
        return float(self.request("GET", f"/characters/{self.character_id}/wallet/"))

    def skill_queue(self) -> list[dict]:
        if not self.authed:
            return []
        return self.request("GET", f"/characters/{self.character_id}/skillqueue/")

    def market_orders(self) -> list[dict]:
        if not self.authed:
            return []
        return self.request("GET", f"/characters/{self.character_id}/orders/")

    def location(self) -> dict:
        if not self.authed:
            return {}
        return self.request("GET", f"/characters/{self.character_id}/location/")

    def universe_type(self, type_id: int) -> dict:
        return self.request("GET", f"/universe/types/{type_id}/")


# --------------------------------------------------------------------------- #
# Structured state + reward
# --------------------------------------------------------------------------- #
@dataclass
class EVEState:
    wallet_isk: float = 0.0
    skill_queue: list[dict] = field(default_factory=list)
    market_orders: list[dict] = field(default_factory=list)
    location: dict = field(default_factory=dict)
    timestamp: float = 0.0


def compute_reward_eve(prev: EVEState, curr: EVEState, actions: list[Action]) -> dict[str, float]:
    """Reward = ISK gained + skills queued − wallet losses − no-op penalty."""
    isk_delta = curr.wallet_isk - prev.wallet_isk
    skills_started = max(0, len(curr.skill_queue) - len(prev.skill_queue))
    orders_filled = max(0, len(prev.market_orders) - len(curr.market_orders))  # orders disappear when filled/cancelled
    breakdown = {
        "isk_delta": float(isk_delta) * 1e-6,                # weight ISK at micro-millions
        "skills_queued": skills_started * 1.0,
        "orders_filled": orders_filled * 2.0,
        "noop_penalty": -0.5 if actions and all(a.action == "wait" for a in actions) else 0.0,
    }
    breakdown["total"] = float(sum(breakdown.values()))
    return breakdown


# --------------------------------------------------------------------------- #
# SDE knowledge extraction
# --------------------------------------------------------------------------- #
@dataclass
class EVEType:
    type_id: int
    name: str
    description: str = ""
    group_id: int | None = None
    market_group_id: int | None = None
    published: bool = True

    def text_blob(self) -> str:
        return f"EVE type {self.type_id} {self.name}\n{self.description}".strip()


def extract_sde_types(sde_dir: str | Path) -> list[EVEType]:
    """Parse ``typeIDs.yaml`` (or ``fsd/types.yaml``) into :class:`EVEType` records.

    The SDE ships in two layouts; we try both. Requires PyYAML and a populated SDE checkout
    (see https://developers.eveonline.com/resource/resources).
    """
    import yaml

    sde_dir = Path(sde_dir)
    candidates = [sde_dir / "fsd" / "types.yaml", sde_dir / "bsd" / "typeIDs.yaml", sde_dir / "typeIDs.yaml"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        log.warning("no SDE types file under %s; tried %s", sde_dir, [str(p) for p in candidates])
        return []

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    out: list[EVEType] = []
    for type_id, data in raw.items():
        name = data.get("name", {}).get("en") if isinstance(data.get("name"), dict) else data.get("name", "")
        desc = data.get("description", {}).get("en") if isinstance(data.get("description"), dict) else data.get("description", "")
        out.append(EVEType(
            type_id=int(type_id),
            name=name or f"type_{type_id}",
            description=desc or "",
            group_id=data.get("groupID"),
            market_group_id=data.get("marketGroupID"),
            published=bool(data.get("published", True)),
        ))
    log.info("parsed %d EVE types from %s", len(out), path)
    return out


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class EVEBackend:
    name = "eve"

    def __init__(self, character_id: int | None = None, access_token: str | None = None,
                 data_root: str | Path = "data"):
        self.client = ESIClient(character_id, access_token)
        self.data_root = Path(data_root)
        self._prev: EVEState | None = None

    def action_space(self) -> dict[str, dict]:
        return dict(EVE_ACTION_SPACE)

    def knowledge_dirs(self) -> dict[str, Path]:
        return {"sde": self.data_root / "eve_sde", "wiki": self.data_root / "eve_wiki"}

    def observe(self) -> Observation:
        import time

        state = EVEState(
            wallet_isk=self.client.wallet_balance(),
            skill_queue=self.client.skill_queue(),
            market_orders=self.client.market_orders(),
            location=self.client.location(),
            timestamp=time.time(),
        )
        return Observation(
            state=state,
            visible_entities=[str(o.get("type_id")) for o in state.market_orders],
            metadata={"authed": self.client.authed},
        )

    def execute(self, action: Action) -> ExecutionResult:
        spec = EVE_ACTION_SPACE.get(action.action)
        if spec is None:
            return ExecutionResult(ok=False, error=f"unknown EVE action {action.action!r}")
        if spec["type"] == "noop":
            return ExecutionResult(ok=True, result={})
        if spec.get("auth") and not self.client.authed:
            return ExecutionResult(ok=False, error="character auth required ($EVE_ACCESS_TOKEN)")
        try:
            path = spec["endpoint"]
            # naive {param} substitution from action params
            for k, v in action.params.items():
                path = path.replace("{" + k + "}", str(v))
            method = spec.get("method", "POST")
            payload = {k: v for k, v in action.params.items() if "{" + k + "}" not in spec["endpoint"]}
            result = self.client.request(method, path, json=payload)
            return ExecutionResult(ok=True, result=result)
        except requests.RequestException as exc:
            return ExecutionResult(ok=False, error=str(exc))

    def reward(self, prev: Observation, curr: Observation, actions: list[Action]) -> dict[str, float]:
        return compute_reward_eve(prev.state, curr.state, actions)

    def reset(self, seed: int | None = None) -> Observation:
        log.info("EVEBackend.reset is a no-op; EVE is a persistent universe.")
        return self.observe()

    def close(self) -> None:
        self.client.session.close()


@register("eve")
def _factory(**kwargs) -> EVEBackend:
    return EVEBackend(**kwargs)
