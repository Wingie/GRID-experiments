## What was novel here

1. **dual semantic IDs** (READ + WRITE) — Spotify's split applied to a codebase.
2. **mmproj for a domain UI** — vision aligned to RSIDs.
3. **discrete diffusion** for action planning.
4. **the game is the reward model** — no separate critic.
5. **commentary wrapper** — response-only mid-play audience channel.

---

## What it generalises to

- testing **your** web app
- navigating **your** IDE
- automating **your** internal tools
- a kiosk in your store
- live commentary on anything the agent does

```bash
# everything you saw, on one 3090:
pip install -e ../vocab-extend-qlora && pip install -e .
python -m rimworld_agent.benchmarks.videogamebench experiment=benchmark
```

---

## Thanks

repo: `wingie/grid-experiments` / `rimworld-sovereign-agent`

ask anything — the agent already is.
