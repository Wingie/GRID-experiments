## One protocol, four flavours of game

```python
class GameBackend(Protocol):
    name: str
    def action_space(self):       ...
    def knowledge_dirs(self):     ...
    def observe(self):            ...
    def execute(self, action):    ...
    def reward(self, prev, curr, actions): ...
    def reset(self, seed=None):   ...
    def close(self):              ...
```

self-registering factories: `get_backend("rimworld" | "eve" | "videogamebench" | "kiosk")`.

---

## Bundled backends

- **`rimworld`** — RIMAPI client + game state + reward (the original).
- **`eve`** — ESI REST + SDE knowledge parser. industry / market / skills / contracts.
- **`videogamebench`** — emulator gym envs. **headline cap: Pokémon Red + Zelda: The Minish Cap**.
- **`kiosk`** — standalone shop-terminal Q&A. response-only contract.
- **`commentary`** — *wraps* any backend with a user-question channel (next slide).

---

## Cross-game benchmark

```bash
python -m rimworld_agent.benchmarks.videogamebench \
  experiment=benchmark \
  benchmark.commentary.enabled=true
```

policy-agnostic; backends that fail to construct (missing dep, missing ROM, missing auth) get recorded with `error` and the rest of the benchmark keeps running.

```json
"pokemon_red": {
  "mean_reward": 47.3,
  "commentary": { "questions_received": 18, "answered": 17, "answer_rate": 0.94 }
}
```
