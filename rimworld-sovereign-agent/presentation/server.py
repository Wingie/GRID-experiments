"""Live presentation server — bridges the reveal.js deck to the running agent.

Exposes a small HTTP + WebSocket surface so the audience can ask questions during the talk
and the agent's responses stream back into the slides:

  GET  /                        -> reveal.js presentation
  GET  /transcript              -> JSON of the agent's full commentary transcript so far
  POST /question  {text}        -> push a question into the shared question source
  POST /transcript {entry}      -> agent posts a new commentary entry (relayed on /events)
  POST /reward    {reward}      -> agent posts the latest reward breakdown (relayed)
  POST /frame     {url}         -> agent posts the latest screenshot URL (relayed)
  WS   /events                  -> broadcast: {type: question_pushed | transcript | reward | frame, ...}

The :data:`SHARED_SOURCE` is a :class:`rimworld_agent.games.commentary.QueueQuestionSource`
that the agent's :class:`CommentaryWrapper` should receive at construction time, so audience
questions land in the same FIFO the game loop pulls from. Run with:

    uvicorn presentation.server:app --reload --port 8000

(Optional ``presentation`` extra — ``pip install fastapi uvicorn``.)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rimworld_agent.games.commentary import QueueQuestionSource

HERE = Path(__file__).parent

# Shared with the running agent: any caller that creates a CommentaryWrapper should pass
# ``presentation.server.SHARED_SOURCE`` as the ``source`` so audience questions are picked up.
SHARED_SOURCE = QueueQuestionSource()
TRANSCRIPT: list[dict] = []
SUBSCRIBERS: list[asyncio.Queue] = []


def _build_app():
    """Construct the FastAPI app lazily so importing this module doesn't require fastapi."""
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Sovereign agent — live presentation")

    @app.get("/")
    def index():
        return FileResponse(HERE / "index.html")

    @app.get("/transcript")
    def get_transcript():
        return {"items": TRANSCRIPT}

    @app.post("/question")
    async def push_question(req: Request):
        body = await req.json()
        text = (body.get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "empty question"}
        SHARED_SOURCE.push(text)
        await _broadcast({"type": "question_pushed", "text": text})
        return {"ok": True, "queue_depth": len(SHARED_SOURCE.questions)}

    @app.post("/transcript")
    async def post_transcript(req: Request):
        """The agent posts a new commentary entry here after each ``say`` action."""
        entry = await req.json()
        TRANSCRIPT.append(entry)
        await _broadcast({"type": "transcript", "entry": entry})
        return {"ok": True}

    @app.post("/reward")
    async def post_reward(req: Request):
        body = await req.json()
        await _broadcast({"type": "reward", "reward": body})
        return {"ok": True}

    @app.post("/frame")
    async def post_frame(req: Request):
        body = await req.json()
        await _broadcast({"type": "frame", "url": body.get("url", "")})
        return {"ok": True}

    @app.websocket("/events")
    async def events(ws: WebSocket):
        await ws.accept()
        queue: asyncio.Queue = asyncio.Queue()
        SUBSCRIBERS.append(queue)
        try:
            await ws.send_text(json.dumps({"type": "hello", "transcript_size": len(TRANSCRIPT)}))
            while True:
                msg = await queue.get()
                await ws.send_text(json.dumps(msg))
        except WebSocketDisconnect:
            pass
        finally:
            SUBSCRIBERS.remove(queue)

    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.mount("/slides", StaticFiles(directory=HERE / "slides"), name="slides")
    return app


async def _broadcast(message: dict) -> None:
    for q in list(SUBSCRIBERS):
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            pass


# Module-level ``app`` is what `uvicorn presentation.server:app` looks for. It is built
# lazily; importing this module without fastapi installed must NOT fail (the test suite
# imports the module to verify the slide/static layout without needing the extra).
try:
    app = _build_app()
except ImportError:
    app = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Agent <-> presentation bridge
# --------------------------------------------------------------------------- #
def bridge_to_server(base_url: str = "http://127.0.0.1:8000"):
    """Return an ``on_say(entry)`` callback that POSTs each commentary entry to the live
    presentation server. Hook into :class:`CommentaryWrapper` like so::

        from presentation.server import bridge_to_server, SHARED_SOURCE
        wrapper = CommentaryWrapper(backend, source=SHARED_SOURCE,
                                    on_say=bridge_to_server("http://127.0.0.1:8000"))

    Failures are logged and swallowed so the game loop is never blocked by the deck server.
    """
    import requests

    base = base_url.rstrip("/")
    session = requests.Session()

    def _on_say(entry: dict) -> None:
        try:
            session.post(f"{base}/transcript", json=entry, timeout=2.0)
        except requests.RequestException:
            pass

    return _on_say


def post_reward(base_url: str, reward: dict) -> None:
    """Convenience: the game loop can call this to broadcast the latest reward breakdown."""
    import requests

    try:
        requests.post(f"{base_url.rstrip('/')}/reward", json=reward, timeout=2.0)
    except requests.RequestException:
        pass


def post_frame(base_url: str, url: str) -> None:
    """Convenience: broadcast the latest screenshot URL (e.g. a /static/ path)."""
    import requests

    try:
        requests.post(f"{base_url.rstrip('/')}/frame", json={"url": url}, timeout=2.0)
    except requests.RequestException:
        pass
