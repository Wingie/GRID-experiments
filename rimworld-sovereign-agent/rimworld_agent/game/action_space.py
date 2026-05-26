"""The finite action vocabulary the agent can emit, plus the structured token format.

Actions map to RIMAPI REST endpoints and/or keyboard shortcuts (project spec §4). Each
action the model outputs is rendered as a token sequence::

    <ACTION_START>
      <ACT:order_build>
      <PARAM:def_name=SolarGenerator>
      <PARAM:x=42>
      <PARAM:y=18>
      <REASON>Need power for research bench</REASON>
    <ACTION_END>

This module is pure data + string logic and is fully covered by the offline test suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Action vocabulary (project spec §4a)
# --------------------------------------------------------------------------- #
ACTION_SPACE: dict[str, dict[str, Any]] = {
    # Camera / navigation
    "camera_pan_left": {"type": "key", "key": "A"},
    "camera_pan_right": {"type": "key", "key": "D"},
    "camera_pan_up": {"type": "key", "key": "W"},
    "camera_pan_down": {"type": "key", "key": "S"},
    "camera_zoom_in": {"type": "key", "key": "ScrollUp"},
    "camera_zoom_out": {"type": "key", "key": "ScrollDown"},
    # Game speed
    "speed_pause": {"type": "key", "key": "Space"},
    "speed_1x": {"type": "key", "key": "1"},
    "speed_2x": {"type": "key", "key": "2"},
    "speed_3x": {"type": "key", "key": "3"},
    # Selection
    "select_at": {"type": "click", "params": ["x", "y"]},
    "right_click_at": {"type": "rclick", "params": ["x", "y"]},
    "drag_select": {"type": "drag", "params": ["x1", "y1", "x2", "y2"]},
    # Architect menu
    "menu_architect": {"type": "key", "key": "F1"},
    "menu_orders": {"type": "submenu", "path": ["Architect", "Orders"]},
    "menu_zone": {"type": "submenu", "path": ["Architect", "Zone"]},
    "menu_structure": {"type": "submenu", "path": ["Architect", "Structure"]},
    "menu_production": {"type": "submenu", "path": ["Architect", "Production"]},
    "menu_furniture": {"type": "submenu", "path": ["Architect", "Furniture"]},
    "menu_power": {"type": "submenu", "path": ["Architect", "Power"]},
    "menu_security": {"type": "submenu", "path": ["Architect", "Security"]},
    "menu_research": {"type": "key", "key": "F2"},
    # Work priorities
    "menu_work": {"type": "key", "key": "F3"},
    "increase_priority": {"type": "api", "endpoint": "/work/priority", "params": ["pawn", "work_type", "delta"]},
    "decrease_priority": {"type": "api", "endpoint": "/work/priority", "params": ["pawn", "work_type", "delta"]},
    # Direct orders (RIMAPI)
    "order_build": {"type": "api", "endpoint": "/build", "params": ["def_name", "x", "y"]},
    "order_mine": {"type": "api", "endpoint": "/designate", "params": ["x", "y", "type"]},
    "order_cut": {"type": "api", "endpoint": "/designate", "params": ["x", "y", "type"]},
    "order_hunt": {"type": "api", "endpoint": "/designate", "params": ["x", "y", "type"]},
    "order_research": {"type": "api", "endpoint": "/research/set", "params": ["project_def"]},
    "draft_pawn": {"type": "api", "endpoint": "/pawn/draft", "params": ["pawn_id"]},
    "undraft_pawn": {"type": "api", "endpoint": "/pawn/undraft", "params": ["pawn_id"]},
    # Zone management (RIMAPI)
    "create_stockpile": {"type": "api", "endpoint": "/zone/create", "params": ["type", "x1", "y1", "x2", "y2"]},
    "create_growing_zone": {
        "type": "api",
        "endpoint": "/zone/create",
        "params": ["type", "x1", "y1", "x2", "y2", "plant_def"],
    },
    # Misc
    "wait": {"type": "noop"},
    "open_tab": {"type": "key", "key": "Tab"},
}

MAX_ACTIONS_PER_TURN = 5


def action_names() -> list[str]:
    return list(ACTION_SPACE)


def is_valid_action(name: str) -> bool:
    return name in ACTION_SPACE


def required_params(name: str) -> list[str]:
    return list(ACTION_SPACE.get(name, {}).get("params", []))


@dataclass
class Action:
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def is_valid(self) -> tuple[bool, str]:
        """Structural validity: known action + all required params present.

        (Spatial/state validity — "is there room here?" — is checked by the live game in
        :mod:`rimworld_agent.game.game_loop`, not here.)
        """
        if not is_valid_action(self.action):
            return False, f"unknown action {self.action!r}"
        missing = [p for p in required_params(self.action) if p not in self.params]
        if missing:
            return False, f"missing params {missing} for {self.action}"
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "params": dict(self.params), "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Action":
        return cls(action=d["action"], params=dict(d.get("params", {})), reason=d.get("reason", ""))


# --------------------------------------------------------------------------- #
# Token (de)serialisation
# --------------------------------------------------------------------------- #
_PARAM_RE = re.compile(r"<PARAM:([^=>]+)=([^>]*)>")
_ACT_RE = re.compile(r"<ACT:([^>]+)>")
_REASON_RE = re.compile(r"<REASON>(.*?)</REASON>", re.S)


def _coerce(value: str) -> Any:
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def format_action(action: Action) -> str:
    """Render an :class:`Action` as its ``<ACTION_START>...<ACTION_END>`` token block."""
    lines = ["<ACTION_START>", f"<ACT:{action.action}>"]
    for key in required_params(action.action) or action.params:
        if key in action.params:
            lines.append(f"<PARAM:{key}={action.params[key]}>")
    if action.reason:
        lines.append(f"<REASON>{action.reason}</REASON>")
    lines.append("<ACTION_END>")
    return "\n".join(lines)


def parse_action(block: str) -> Action:
    """Parse one ``<ACTION_START>...<ACTION_END>`` block back into an :class:`Action`."""
    act_match = _ACT_RE.search(block)
    if not act_match:
        raise ValueError("no <ACT:...> tag found in action block")
    params = {k.strip(): _coerce(v.strip()) for k, v in _PARAM_RE.findall(block)}
    reason_match = _REASON_RE.search(block)
    reason = reason_match.group(1).strip() if reason_match else ""
    return Action(action=act_match.group(1).strip(), params=params, reason=reason)


def format_actions(actions: list[Action]) -> str:
    return "\n".join(format_action(a) for a in actions[:MAX_ACTIONS_PER_TURN])


def parse_actions(text: str) -> list[Action]:
    blocks = re.findall(r"<ACTION_START>.*?<ACTION_END>", text, re.S)
    return [parse_action(b) for b in blocks]


# The special tokens this action grammar introduces (registered alongside SID tokens).
def action_special_tokens() -> list[str]:
    structural = ["<ACTION_START>", "<ACTION_END>", "<REASON>", "</REASON>"]
    act_tokens = [f"<ACT:{name}>" for name in ACTION_SPACE]
    return structural + act_tokens
