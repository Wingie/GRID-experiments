# Multi-game framework

The sovereign agent is not RimWorld-specific. It plugs into any game that exposes
**knowledge** (docs/source/data), **state** (a readable snapshot), and **control** (an API or
emulator action set). This file documents the unified backend protocol and the three
bundled backends.

## The protocol

`rimworld_agent.games.base.GameBackend` is a small `Protocol`:

```python
class GameBackend(Protocol):
    name: str
    def action_space(self)       -> dict[str, dict]
    def knowledge_dirs(self)     -> dict[str, Path]
    def observe(self)            -> Observation
    def execute(self, action)    -> ExecutionResult
    def reward(self, prev, curr, actions) -> dict[str, float]
    def reset(self, seed=None)   -> Observation
    def close(self)              -> None
```

Backends self-register via `@register("name")`. Build one with
`get_backend("name", **kwargs)`; list them with `list_backends()`.

The `Action` payload (`name`, `params`, `reason`) is **shared across all backends**; what
changes per game is the `action_space()` vocabulary and the underlying `execute` mechanism.

## Bundled backends

### 1. `rimworld` — `RimWorldBackend`

Thin adapter over the existing `RimAPIClient`, `ACTION_SPACE`, and `compute_reward`. State
type: `GameState`. Knowledge: `data/rimworld_xml_defs/`, `data/rimworld_source/`,
`data/rimworld_wiki/`. Needs the RIMAPI mod (see `RIMWORLD_SETUP.md`).

### 2. `eve` — `EVEBackend`  (EVE Online)

REST client over the **ESI** API (`https://esi.evetech.net/latest/`) + a parser for the
**SDE** YAML Static Data Export. Most "playing" in EVE happens in-client, so the actionable
subset is industry / market / skill / contracts / navigation — a great testbed for the
*economy* axis of the sovereign-agent recipe:

| Action | ESI endpoint |
|--------|--------------|
| `skill_train` | `POST /characters/{cid}/skillqueue` |
| `order_buy` / `order_sell` | `POST /characters/{cid}/orders` |
| `set_destination` | `POST /ui/autopilot/waypoint` |
| `accept_contract` | `POST /characters/{cid}/contracts/{id}/bids` |
| `send_mail` | `POST /characters/{cid}/mail` |

Auth: character-scoped endpoints need an OAuth2 token in `$EVE_ACCESS_TOKEN` (with
`$EVE_CHARACTER_ID`). Without one the backend is read-only but still callable. Reward sums
ISK delta, skills queued, orders filled, with a no-op penalty.

Knowledge extraction: `extract_sde_types` reads `data/eve_sde/fsd/types.yaml` (or the legacy
`bsd/typeIDs.yaml`) into `EVEType` records — directly compatible with the same dual
RQ-VAE pipeline (RSIDs cluster by hull/module/ore taxonomy; WSIDs cluster by industry chain).

### 3. `kiosk` — `KioskBackend` (response-only Q&A)

A "shop terminal" mode where the agent **only responds** to customer questions about a
catalog of items and **never asks** questions back. There is no game world — just a catalog
(JSON) and a queue of incoming questions. Structural enforcement of the no-asking rule is in
the action space: there is no `ask_clarification` action, and every response goes through a
question-mark stripper before publication.

Action space (all response-shaped): `answer`, `lookup_price`, `list_items`, `recommend`,
`inventory_query`, `wait`. Reward shaping rewards grounded citations of real catalog items
and penalises empty responses + any question-mark in the text (`asked_back_penalty = −2`).
Catalog items can carry their canonical `<RSID_L*_*>` address so a response that cites
`rifle_01` also cites the same RSID the dual SID pipeline assigned to it — the same trained
sovereign SLM can be deployed here without retraining.

Verbal I/O (ASR in, TTS out) plugs in at the boundary; the backend itself is text-only.

```bash
python -c "from rimworld_agent.games.base import get_backend; from rimworld_agent.game.action_space import Action; \
b = get_backend('kiosk', catalog_path='data/kiosk_catalog/catalog.json'); b.observe(); \
print(b.execute(Action('lookup_price', {'item_id': 'rifle_01'})).result)"
```

### 4. `videogamebench` — `VideoGameBenchBackend`

Adapter over a [VideoGameBench](https://www.videogamebench.com) gym env (Game Boy / GBA / DOS
emulators). The benchmark default is **capped to the two headline games**:

| ID | Game | Platform |
|----|------|----------|
| `pokemon_red` | Pokémon Red | GB |
| `zelda_minish_cap` | The Legend of Zelda: The Minish Cap | GBA |

Convenience factories (`get_backend("pokemon_red")`) preset the game; extra titles work via
`get_backend("videogamebench", game="<vgb_game_id>")`. The action space is the GB/GBA button
set (`press_a`, `press_b`, `press_up`, ..., `wait`), mapped to the env's discrete action
indices at construction. Reward uses the env's per-step reward plus a terminated bonus and an
optional `info.progress` proxy. Needs `pip install videogamebench`.

## The cross-game benchmark

`rimworld_agent.benchmarks.videogamebench.run_benchmark(policy, games, n_episodes, ...)`
drives each backend through `n_episodes` and aggregates `mean_reward / best_reward /
termination_rate / mean_progress` per game. Backends that fail to construct (missing dep,
missing ROM, missing auth) are recorded with `error` and skipped — the rest of the benchmark
keeps running.

```bash
python -m rimworld_agent.benchmarks.videogamebench experiment=benchmark
#   -> results/videogamebench.json  (per-game table; default games=pokemon_red+zelda_minish_cap)
```

The runner is **policy-agnostic**: it takes any `policy(obs) -> (reasoning, actions)` so the
same trained sovereign SLM can be benchmarked across RimWorld, EVE, and the VGB titles
without bespoke glue.

## Adding a new game

1. Drop a module at `rimworld_agent/games/<game>.py`.
2. Implement the `GameBackend` protocol (`action_space`, `observe`, `execute`, `reward`,
   `reset`, `close`, `knowledge_dirs`).
3. `@register("<game>")` a factory.
4. Add `configs/games/<game>.yaml` for game-specific config.
5. Add a unit test under `tests/test_<game>.py`.

The dual RQ-VAE, vision (mmproj), and self-play layers are game-agnostic and will work
against the new backend as soon as it conforms.
