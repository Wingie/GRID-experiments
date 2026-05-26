"""Capture game screenshots — via RIMAPI's screenshot endpoint (preferred) or ``mss``
screen capture as a fallback (gotcha #2: screenshots must come from the game window, not
the desktop).

For mmproj training, capture at 1x for clean frames; during self-play, the 4-frame history
is captured by the game loop at the step interval. Live-only.
"""

from __future__ import annotations

from pathlib import Path

from rimworld_agent.utils import ensure_dir, get_logger

log = get_logger("screenshot")


def capture_via_rimapi(client, path: str | Path) -> Path:
    """Save a screenshot through RIMAPI (the only source that sees the game window cleanly)."""
    path = Path(path)
    ensure_dir(path.parent)
    client.screenshot(str(path))
    return path


def capture_via_mss(region: dict | None, path: str | Path) -> Path:
    """Fallback desktop capture of the game window region via ``mss``."""
    import mss
    from PIL import Image

    path = Path(path)
    ensure_dir(path.parent)
    with mss.mss() as sct:
        monitor = region or sct.monitors[1]
        shot = sct.grab(monitor)
        Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX").save(path)
    return path


def collect_training_screenshots(client, out_dir: str | Path, count: int, interval_s: float = 30.0) -> list[Path]:
    """Manual-play capture loop: a screenshot + paired game state every ``interval_s`` (spec §7b)."""
    import json
    import time

    out_dir = ensure_dir(out_dir)
    paths: list[Path] = []
    for i in range(count):
        img = capture_via_rimapi(client, out_dir / f"shot_{i:05d}.png")
        state = client.get_state()
        (out_dir / f"shot_{i:05d}.json").write_text(json.dumps(state.to_dict(), indent=2))
        paths.append(img)
        log.info("captured %d/%d", i + 1, count)
        time.sleep(interval_s)
    return paths
