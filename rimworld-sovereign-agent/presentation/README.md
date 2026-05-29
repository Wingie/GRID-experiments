# Live presentation deck

A reveal.js + Markdown talk on the sovereign agent, with React widgets that bridge to a running agent so the audience can ask questions mid-demo and see the answers stream in.

## Run it

```bash
pip install fastapi uvicorn          # the `presentation` extra
uvicorn presentation.server:app --reload --port 8000
# open http://127.0.0.1:8000
```

Navigate to the "Live demo" slide. The grid mounts on slide change and tears down when you navigate away (so the WebSocket only runs when needed).

## Wiring the agent to the deck

The same `QueueQuestionSource` that the deck pushes into is exported as `presentation.server.SHARED_SOURCE`. Hook it into your `CommentaryWrapper`:

```python
from presentation.server import SHARED_SOURCE, bridge_to_server
from rimworld_agent.games.commentary import CommentaryWrapper
from rimworld_agent.games.base import get_backend

backend = CommentaryWrapper(
    get_backend("rimworld"),
    source=SHARED_SOURCE,
    on_say=bridge_to_server("http://127.0.0.1:8000"),
)
# ...your game loop runs, audience questions arrive via /question, every
# say-action is POSTed back to /transcript, and the deck re-renders.
```

For reward and game-frame broadcasts:

```python
from presentation.server import post_reward, post_frame

post_reward("http://127.0.0.1:8000", {"gameplay": 12.4, "commentary": 4.0, "total": 16.4})
post_frame("http://127.0.0.1:8000", "/static/frames/frame_000123.png")
```

## Endpoints

| Method | Path | Body | Purpose |
|-------|------|------|---------|
| GET | `/` | — | reveal.js index |
| GET | `/transcript` | — | full transcript so far |
| POST | `/question` | `{text}` | audience pushes a question |
| POST | `/transcript` | `{ts, question, response}` | agent posts a new entry |
| POST | `/reward` | `{gameplay, commentary, total}` | agent posts the latest reward |
| POST | `/frame` | `{url}` | agent posts the latest frame URL |
| WS | `/events` | — | broadcast of `question_pushed / transcript / reward / frame` |

## Slide source

Markdown files live under `slides/`; reveal.js loads them via the bundled Markdown plugin. Vertical splits use `\n--\n`, horizontal `\n---\n`. Edit and reload — no build step.

## What's React vs. what's reveal.js

- **reveal.js** owns navigation, theme, and slide-from-Markdown rendering.
- **React** (via UMD + `htm` for JSX-free templates — no build) owns the live widgets on the "Live demo" slide: question input, transcript stream, reward gauge, game frame.
- The two compose because reveal.js renders raw HTML inside `<section>`; the React root mounts into `#live-demo-root` on `slidechanged` and unmounts off-slide.
