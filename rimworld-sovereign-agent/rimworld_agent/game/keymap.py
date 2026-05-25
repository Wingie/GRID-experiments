"""Keyboard shortcuts <-> action mapping.

RimWorld is largely keyboard-driven. This derives a key->action and action->key map from
the ``key``-typed entries in :data:`rimworld_agent.game.action_space.ACTION_SPACE`, and
exposes helpers the RIMAPI client / game loop use to translate a chosen action into a key
event. Pure data logic; covered by the offline tests.
"""

from __future__ import annotations

from rimworld_agent.game.action_space import ACTION_SPACE

# Additional human-facing shortcuts that are not 1:1 with an action key but are useful
# context for the model (documented for the prompt, not all directly emitted).
EXTRA_SHORTCUTS = {
    "Escape": "cancel",
    "Home": "camera_home",
    "Delete": "cancel_designation",
}


def action_to_key() -> dict[str, str]:
    """Map action name -> key for all ``key``-typed actions."""
    return {name: spec["key"] for name, spec in ACTION_SPACE.items() if spec.get("type") == "key"}


def key_to_action() -> dict[str, str]:
    """Reverse map key -> action name (last writer wins on duplicate keys)."""
    return {key: name for name, key in action_to_key().items()}


def key_for(action_name: str) -> str | None:
    return action_to_key().get(action_name)


def is_key_action(action_name: str) -> bool:
    return ACTION_SPACE.get(action_name, {}).get("type") == "key"
