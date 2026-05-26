"""Detect on-screen UI elements (alerts, the architect menu, selection panels, the speed
control) so the vision encoder and prompt can be grounded in what the player would see.

Two detection backends:
  * ``rimapi`` — ask RIMAPI for the current UI element rects/labels (exact, preferred);
  * ``heuristic`` — fixed-region + colour heuristics on the screenshot (offline-ish, for
    when RIMAPI lacks a UI endpoint).

Live/GPU-adjacent; not part of the offline test subset.
"""

from __future__ import annotations

from dataclasses import dataclass

from rimworld_agent.utils import get_logger

log = get_logger("ui_elements")


@dataclass
class UIElement:
    kind: str  # alert | menu | selection_panel | speed_control | button
    label: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float = 1.0


def detect_via_rimapi(client) -> list[UIElement]:
    raw = client._get("/ui/elements")
    return [
        UIElement(
            kind=e.get("kind", "button"),
            label=e.get("label", ""),
            bbox=tuple(e.get("bbox", (0, 0, 0, 0))),
            confidence=float(e.get("confidence", 1.0)),
        )
        for e in raw.get("elements", [])
    ]


# RimWorld's default 1920x1080 layout: alert column top-right, speed control bottom-right.
_DEFAULT_REGIONS = {
    "alert": (1660, 10, 1910, 400),
    "speed_control": (1640, 1000, 1910, 1075),
    "selection_panel": (0, 760, 460, 1075),
    "architect_menu": (0, 600, 200, 1000),
}


def detect_heuristic(image, regions: dict | None = None) -> list[UIElement]:
    """Crude region-based detector: report which fixed UI regions are non-empty.

    ``image`` is a PIL image. This is a placeholder for a trained detector; it exists so the
    pipeline is runnable end-to-end before a detector is trained.
    """
    import numpy as np

    regions = regions or _DEFAULT_REGIONS
    arr = np.asarray(image.convert("L"))
    out: list[UIElement] = []
    for kind, (x1, y1, x2, y2) in regions.items():
        crop = arr[y1:y2, x1:x2]
        if crop.size and float(crop.std()) > 5.0:  # some content present
            out.append(UIElement(kind=kind, label=kind, bbox=(x1, y1, x2, y2), confidence=0.5))
    return out
