## Commentary while it plays

```python
from rimworld_agent.games.commentary import wrap, QueueQuestionSource

source = QueueQuestionSource()
backend = wrap("rimworld", source=source,
               transcript_path="results/commentary.jsonl")
source.push("Why did you build the bed there?")
# policy answers with Action("say", {text, cited_items})
# while still emitting its in-game actions for the same turn.
```

every `observe()` checks the source and surfaces a pending question in `Observation.metadata["user_question"]`.

---

## Strictly response-only

structural enforcement:

- no `ask_clarification` action exists in the wrapped vocabulary;
- a `?` stripper sanitises the `say` text before publication;
- the reward shaper adds:
  - `+1` *answered*
  - `+1` *rsid_grounded* (citation references a real RSID)
  - `−2` *asked_back_penalty* on any surviving `?`

so the model is rewarded for **grounded, ungrudging answers**.

---

## Question sources are pluggable

- `QueueQuestionSource` — FIFO, `push` from any UI / mic / websocket
- `FileQuestionSource` — tails a text file (used by the benchmark runner)
- *(this presentation)* a WebSocket bridge → audience input goes straight in

verbal I/O plugs in at the source (ASR in) / transcript (TTS out) boundary.
