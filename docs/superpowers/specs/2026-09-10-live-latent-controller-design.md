# Live latent controller — design

Date: 2026-09-10
Status: approved, pending implementation plan

## Purpose

Replace the ad-hoc TouchDesigner test rig (see `w_osc_debug_viewer.py` /
`w_osc_debug_receiver.py` troubleshooting session earlier the same day) with a
real performance controller, built in Open Stage Control, that sits between
`live_pipeline.py` and Autolume. It lets an operator:

- pick one of the actress' recorded emotions from a thumbnail grid and smoothly
  transition the output latent vector toward it, live, during a show
- blend that actress-driven vector against the incoming visitor's own latent
  vector (`w_u`, computed by `live_pipeline.py` from their biodata), and add
  controlled noise

It replaces the existing plain session-status control panel
(`live_pipeline/crocodile-control-panel.json`) by extending it — the session
panel becomes one tab among several in the same instance, rather than a
separate process, since there is no reason to keep them on separate ports.

## Current state (before this change)

- `live_pipeline.py` computes `w_u` from live biodata and sends it directly to
  Autolume: `--out-host/--out-port` (default `127.0.0.1:1338`), address
  `/crocodile/w`, 512 floats.
- `live_pipeline/crocodile-control-panel.json` is a single-tab Open Stage
  Control session (session id, calibration, live start/stop, status display)
  launched by `live_pipeline/run_control_panel.sh`: HTTP UI on `8090`, sends
  session-control OSC to `live_pipeline.py`'s listen port `9000`, itself
  listens on `9001` for status broadcasts. No custom module is currently
  loaded for this session.
- A separate, unrelated Open Stage Control project
  (`open-stage-control/EmotionVectorsControl.json` +
  `EmotionVectorsCustomModule.js`) already demonstrates the two building
  blocks this design reuses: a custom module bridging external OSC
  (`EmotionVectorsCustomModule.js`, for Wekinator) and per-widget vector math
  (`EmotionVectorsControl.json`'s `matrix_emotions` → `matrix_eigens`
  multiply). It is not otherwise touched by this work.
- `emotion_grid/data/manifest.csv` (private, gitignored, built by
  `emotion_grid/build_grid.py`) already has exactly the data panel 1 needs:
  22 emotion codes × 5 images each, one row per image with
  `id, emotion, feeling_it, thumbnail_path, pool_name, frame_number,
  w_000..w_511`. `emotion_grid/data/thumbnails/<emotion>/<id>.png` holds the
  256×256 images. `latent_pipeline/data/emotion_labels.csv` (tracked, not
  private) maps each 3-letter code to a full display name.

## Architecture & data flow

```
live_pipeline.py --out-host/--out-port(9001)--> [control panel: Open Stage Control server]
   /crocodile/latent/user, 512 floats                - session JSON: tabs (Session / Emotion Grid / Mixing)
                                                       - custom module: state + math + OSC bridge
                                                    --send()--> Autolume :1338
                                                       /crocodile/latent/final, 512 floats
```

- One Open Stage Control process, unchanged ports: HTTP `8090`, OSC-in `9001`
  (already used for status broadcasts; now also carries the `w_u` stream and
  all new panel-2/panel-3 widget addresses — same "one port, disambiguate by
  address" pattern already used on `9000` for `/crocodile/biodata` +
  `/crocodile/session/*`), OSC-out to `live_pipeline.py`'s listen port `9000`
  (unchanged, session-control tab keeps working exactly as today).
- `run_control_panel.sh` gains a `--custom-module` flag pointing at the new
  module file.
- `live_pipeline.py` changes: `--out-port` default `1338` → `9001`,
  `--out-address` default `/crocodile/w` → `/crocodile/latent/user`. No change
  to how it computes `w_u` — only where it sends it.
- The custom module intercepts `/crocodile/latent/user` via `oscInFilter`,
  updates its internal `w_u` state, and drops the message (it's not a normal
  widget value). It intercepts the new lightweight control addresses
  (thumbnail pick, sliders, start/stop, mode switch) the same way, updating
  internal state and pushing visual feedback back to widgets via
  `receive("/SET", widget_id, value, ...)` — the same mechanism
  `EmotionVectorsCustomModule.js` already uses.
- The module's own `init()` starts a fixed-rate output loop (default 30 Hz,
  matching the GAN pipeline's video FPS) that advances `current_w` toward
  `target_w`, computes the actress/user blend + noise, and `send()`s the
  512-float result to Autolume on `127.0.0.1:1338`,
  `/crocodile/latent/final` — Autolume's own OSC-input address setting must
  be updated to match (it's user-configurable there, per `README.md`; the
  port `1338` itself is Autolume's fixed default and doesn't change).
- Thumbnail W-vectors: loaded once at module `init()` from
  `emotion_grid/data/manifest.csv` into an in-memory array (id → 512-float
  vector + emotion code). Never leaves this machine, never touched by git
  (already gitignored).

### Downstream naming follow-through

`w_osc_debug_receiver.py` and `w_osc_debug_viewer.py` are kept — they remain
useful as a lightweight stand-in for Autolume when testing the pipeline
without it running. They're renamed (`git mv`) to
`latent_osc_debug_receiver.py` / `latent_osc_debug_viewer.py` for consistency
with the rest of the `w` → `latent` renaming, and their `--in-address`
defaults change from `/crocodile/w` to `/crocodile/latent/final` — their
purpose (sanity-check the exact stream Autolume receives) is unchanged, only
the address naming and, now, the sender (module instead of `live_pipeline.py`
directly).

## Session layout: tabs

The root of `crocodile-control-panel.json` switches from a single top-level
`panel` to a `tab` container with three tabs:

1. **Session** — today's existing content (session id, calibration controls,
   live start/stop, status display), moved in unchanged.
2. **Emotion Grid** — panel 1, described below.
3. **Mixing** — panel 2, described below.

## Panel 1 — Emotion Grid (direct actress emotion selection)

- **Grid**: 22 columns (one per emotion, headed with its full name from
  `emotion_labels.csv`) × 5 rows. Each cell is a `button` widget (not the
  `image` widget — its click/tap semantics aren't reliably documented,
  whereas `button` guarantees `onTouch`), styled via `css` with
  `background-image: url(file://<absolute thumbnail path>)`, no text label,
  laid out as square tiles.
- **Selecting a tile** sends its row `id` (e.g. `ang_00`) to the module on a
  new address (e.g. `/grid/select`). The module looks up that id's 512-float
  vector in the manifest lookup, sets `target_w`, and updates a shared
  `selected_id` state that every tile's `colorStroke`/`colorWidget`
  references via `@{}` — so the previously- and newly-selected tiles
  re-style themselves reactively, no per-tile script needed.
- **Mode switch** (`switch`, 2 values: *Auto-start on select* / *Manual
  start*): read by the module when a tile is picked, to decide whether to
  set `running = true` immediately or wait for the Start button.
- **Transition speed** (`fader`, 0–1, a "responsiveness" coefficient, not a
  fixed duration): each output tick, while `running`,
  `current_w += (target_w - current_w) * speed`. Chosen over a fixed-duration
  linear walk because it handles re-targeting mid-transition gracefully —
  picking a new tile while already moving smoothly redirects from wherever
  `current_w` currently is, which matters for live use. A fixed-duration
  linear walk is the noted alternative if a "arrives in exactly N seconds"
  feel turns out to matter more in practice.
- **Start/Stop button**: gates the stepping above. Stop freezes `current_w`
  exactly where it is; Start resumes moving toward whatever `target_w`
  currently holds.

## Panel 2 — Actress/User Mixing

- **Mix fader** (0–100%): `final_w = mix*current_w + (1-mix)*w_u` — 100% =
  pure actress, 0% = pure incoming visitor vector.
- **Noise fader** (0–100%): scales a persistent per-dimension noise state the
  module low-pass-filters each tick (exponentially-smoothed random walk, not
  fresh white noise per tick — avoids visible per-frame flicker on the
  rendered face), added to `final_w` after mixing.
- **Before any `w_u` has arrived** (pipeline not running yet / no visitor
  connected): the module defaults `w_u` to `current_w`, so the mix slider has
  no visible effect until real visitor data shows up, instead of mixing
  toward zero or stale garbage.

## Custom module: state summary

Held entirely in the module (not in session widgets, since 512-float vectors
aren't practical widget values):

| State | Set by |
|---|---|
| `w_u` | `/crocodile/latent/user` (from `live_pipeline.py`) |
| `target_w`, `selected_id` | `/grid/select` (thumbnail tap) |
| `mode` (auto/manual start) | `/transition/mode` (switch) |
| `speed` | `/transition/speed` (fader) |
| `running` | `/transition/running` (start/stop button), or auto-set on select if `mode` is auto-start |
| `current_w` | advanced every output tick while `running` |
| `mix` | `/mix/amount` (fader) |
| `noise_amount` | `/noise/amount` (fader) |
| `noise_state` (internal, 512 floats) | advanced every output tick regardless of `running`/`mix` |

Output tick (default 30 Hz, `init()`-started `setInterval`):
1. if `running`: `current_w += (target_w - current_w) * speed`
2. advance `noise_state` (smoothed random walk)
3. `w_u_effective = w_u ?? current_w`
4. `final_w = mix*current_w + (1-mix)*w_u_effective + noise_state*noise_amount`
5. `send('127.0.0.1', 1338, '/crocodile/latent/final', ...final_w)`

## Files touched / created

- `live_pipeline/crocodile-control-panel.json` — restructured to `tab`
  container; existing session content becomes tab 1; new tabs 2 and 3 added.
- `live_pipeline/crocodile-control-module.js` — new custom module (state,
  manifest loading, output loop, OSC bridging described above).
- `live_pipeline/run_control_panel.sh` — add `--custom-module` flag.
- `live_pipeline/live_pipeline.py` — `--out-port` default `1338` → `9001`,
  `--out-address` default `/crocodile/w` → `/crocodile/latent/user`.
- `live_pipeline/w_osc_debug_receiver.py` → renamed
  `live_pipeline/latent_osc_debug_receiver.py`;
  `live_pipeline/w_osc_debug_viewer.py` → renamed
  `live_pipeline/latent_osc_debug_viewer.py`. Both keep their behavior,
  `--in-address` default changes `/crocodile/w` → `/crocodile/latent/final`.
- `live_pipeline/run_debug_receiver.sh`, `live_pipeline/run_debug_viewer.sh`
  — updated to invoke the renamed scripts.
- `PIPELINE.md`, `INSTALL.md` — references to the old script names and
  `/crocodile/w` updated to match.
- Autolume (separate repo, `~/Documents/workspace/autolume`) — its OSC
  input address setting needs updating to `/crocodile/latent/final` to match
  (manual/UI change on that side, not part of this repo's changes; also
  still carries the unrelated `in_ip = "127.0.0.1"` bug flagged earlier,
  fixed separately if/when asked).
- Not touched: `open-stage-control/EmotionVectorsControl.json`,
  `EmotionVectorsCustomModule.js` (unrelated Wekinator-facing project).

## Out of scope / not decided here

- Exact numeric defaults (initial `speed`, `noise` smoothing constant, tick
  rate) — reasonable defaults proposed above, tunable during implementation
  and live testing, not load-bearing design decisions.
- Fixed-duration (vs. responsiveness-coefficient) transition mode — noted as
  a possible future alternative, not building both.
- Any change to how `live_pipeline.py` computes `w_u` itself (regressor,
  alignment, etc.) — untouched by this work.
