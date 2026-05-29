"""Game backends — uniform :class:`GameBackend` over RimWorld, EVE Online, VideoGameBench.

See ``base.py`` for the protocol and registry. Each backend self-registers on import.
"""

from rimworld_agent.games.base import (  # noqa: F401
    Action,
    ExecutionResult,
    GameBackend,
    Observation,
    get_backend,
    list_backends,
    register,
)
