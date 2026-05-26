# RimWorld setup

This agent needs three things from a RimWorld install: the **XML Defs** and **decompiled
C#** for knowledge extraction, the **RIMAPI mod** for state/actions, and **screenshots**.

> None of this runs in CI / the cloud sandbox — it requires a local RimWorld install,
> a GPU, and a display. The offline subset (parsing, semantic IDs, tests) needs only the
> copied XML Defs.

## 1. Copy the XML Defs

RimWorld's content lives in `Mods/Core/Defs/` (and DLC `Mods/Royalty|Ideology|Biotech`).
Copy them into this repo so `extract_defs` can read them:

```bash
# macOS
cp -r "~/Library/Application Support/Steam/steamapps/common/RimWorld/RimWorldMac.app/Data/Core/Defs/" data/rimworld_xml_defs/
# Windows (PowerShell)
Copy-Item -Recurse "C:\Program Files (x86)\Steam\steamapps\common\RimWorld\Data\Core\Defs\*" data\rimworld_xml_defs\
# Linux
cp -r ~/.steam/steam/steamapps/common/RimWorld/Data/Core/Defs/. data/rimworld_xml_defs/
```

## 2. Decompile `Assembly-CSharp.dll`

Use ILSpy (`ilspycmd`) to decompile the game assembly into `data/rimworld_source/`:

```bash
scripts/decompile_source.sh "/path/to/RimWorld/RimWorldWin64_Data/Managed/Assembly-CSharp.dll"
```

The script invokes `ilspycmd -p -o data/rimworld_source <dll>`. Decompiled field names are
preserved; some local variable names are obfuscated (that's fine — `extract_csharp` reads
types, methods, and fields).

## 3. Install RIMAPI

RIMAPI is a RimWorld mod that exposes a small HTTP server inside the running game (read
state, capture a screenshot, execute orders). Without it you are limited to screenshot-only
observation and keyboard-only control.

```bash
scripts/setup_rimworld.sh        # downloads/links RIMAPI into your Mods folder
```

Then enable **RIMAPI** in the in-game mod list and restart. Verify:

```bash
curl http://127.0.0.1:7860/state | head
```

Set `game.rimapi_url` if you changed the port.

## 4. Screenshots

RimWorld runs on Unity — screenshots must come from the game window, not the desktop
(gotcha #2). Prefer RIMAPI's `/screenshot` endpoint. If unavailable, the `mss` fallback in
`rimworld_agent/vision/screenshot.py` captures the screen region; configure the monitor /
window rectangle there.

Game screenshots are ~1920×1080 while SigLIP wants 384/768 — resize carefully or UI text
becomes unreadable (gotcha #10). The default SigLIP-SO400M at 384 is a reasonable start.

## 5. Scenario

Start with **Crashlanded** on **Peaceful** (gotcha #6): no threats means pure
build/research optimisation, which isolates the planning/vision problem from combat. Add
harder difficulties once the agent is stable.

## RIMAPI endpoints used

| Method | Path | Purpose |
|-------|------|---------|
| GET | `/state` | structured colony state |
| GET | `/screenshot` | PNG of the game window |
| GET | `/visible` | defNames currently on screen |
| GET | `/events` | SSE stream of game events |
| POST | `/key` | send a key event |
| POST | `/click` | click at (x, y) |
| POST | `/build` `/designate` `/research/set` `/work/priority` `/pawn/*` `/zone/create` | orders |

See `rimworld_agent/game/action_space.py` for the full action → endpoint mapping.
