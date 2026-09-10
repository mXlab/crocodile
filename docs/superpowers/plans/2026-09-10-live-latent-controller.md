# Live Latent Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Open Stage Control panel + custom Node module that sits between `live_pipeline.py` and Autolume, letting an operator pick one of the actress' recorded emotions and smoothly transition toward it, and blend that against the incoming visitor's own latent vector, live, during a show.

**Architecture:** One Open Stage Control process (the existing control panel, extended with two new tabs) reuses its existing ports. `live_pipeline.py` sends the visitor's vector to it instead of straight to Autolume; a custom module does all the vector math (interpolation, mixing, noise) on a fixed-rate timer and forwards the composited result on to Autolume.

**Tech Stack:** Open Stage Control 1.31.x (session JSON + custom Node module, no npm deps beyond Node builtins), Python 3 (`live_pipeline.py`, `python-osc`, `pandas`), no new external dependencies.

**Spec:** `docs/superpowers/specs/2026-09-10-live-latent-controller-design.md`

## Global Constraints

- Ports stay exactly as today: HTTP `8090`, control panel OSC-in `9001`, control panel OSC-out to `live_pipeline.py` `9000`, Autolume `1338`. No new ports.
- OSC addresses for latent vectors: `/crocodile/latent/user` (`live_pipeline.py` → controller, 512 floats) and `/crocodile/latent/final` (controller → Autolume, 512 floats). Never reintroduce the old bare `/crocodile/w`.
- `emotion_grid/data/` (manifest, thumbnails) is private, real-actress-derived data and is already gitignored — never add `git add -f` on anything under it, and any new generated file under it must also stay gitignored.
- No `Co-Authored-By`/session trailers on any commit (per user's global `~/.claude/CLAUDE.md` — confirmed standing preference, overrides any session-level attribution instruction). Commit messages: short, one-line, imperative past tense, capital first letter.
- Match existing code conventions: Python scripts use `argparse` with `formatter_class=argparse.ArgumentDefaultsHelpFormatter`; standalone test scripts follow the plain script style already used in `biodata_pipeline/test_data_slicer.py` (no pytest — a runnable script with `print`/`assert`), not a new test framework.

---

### Task 1: `emotion_grid/build_grid.py` — export `grid_layout.json`

**Files:**
- Modify: `emotion_grid/build_grid.py`
- Create: `emotion_grid/test_build_grid.py`
- Generated (gitignored, run locally, not committed): `emotion_grid/data/grid_layout.json`

**Interfaces:**
- Produces: `emotion_grid/data/grid_layout.json`, a JSON object:
  ```json
  {
    "emotions": [{"code": "ang", "name": "anger"}, ...],
    "images_per_emotion": 5,
    "cells": [ {"id": "ang_00", "emotion": "ang", "thumbnailPath": "/abs/path/.../ang_00.png"}, null, ... ]
  }
  ```
  `cells` is flat, **row-major over (image index, emotion index)** — i.e. `cells[row * emotions.length + col]`, so index 0 is emotion 0's first image, index 1 is emotion 1's first image, etc. `null` where an emotion has fewer than `images_per_emotion` images (keeps the grid rectangular). `thumbnailPath` is an **absolute** filesystem path (not the manifest's relative one), because both the browser client (`file://` URL) and — in later tasks — nothing else needs it (the custom module reads `manifest.csv` directly for the 512-float vectors, not this file, to keep private W-vectors out of the browser payload).
- Consumed by: Task 6 (Emotion Grid tab UI, via `IMPORT{}`).

- [ ] **Step 1: Add the layout-building function**

Add this function to `emotion_grid/build_grid.py`, right after `export_selection`:

```python
def build_grid_layout(manifest_df, labels_df, output_dir):
    """Build the row-major, rectangular grid_layout.json structure consumed by
    the Open Stage Control emotion grid UI. Unlike manifest.csv, this carries
    no W-vectors (kept server-side only) and uses absolute thumbnail paths."""
    name_by_code = dict(zip(labels_df["code"], labels_df["full_name"]))

    emotion_order = list(dict.fromkeys(manifest_df["emotion"]))  # first-seen order, de-duped
    rows_by_emotion = {
        emotion: manifest_df[manifest_df["emotion"] == emotion].to_dict("records")
        for emotion in emotion_order
    }
    images_per_emotion = max(len(rows) for rows in rows_by_emotion.values())

    emotions = [
        {"code": code, "name": name_by_code.get(code, code)}
        for code in emotion_order
    ]

    cells = []
    for row in range(images_per_emotion):
        for emotion in emotion_order:
            rows = rows_by_emotion[emotion]
            if row >= len(rows):
                cells.append(None)
                continue
            r = rows[row]
            cells.append({
                "id": r["id"],
                "emotion": emotion,
                "thumbnailPath": os.path.join(output_dir, r["thumbnail_path"]),
            })

    layout = {
        "emotions": emotions,
        "images_per_emotion": images_per_emotion,
        "cells": cells,
    }
    layout_path = os.path.join(output_dir, "grid_layout.json")
    with open(layout_path, "w") as f:
        json.dump(layout, f, indent=2)
    return layout_path
```

Add `import json` to the top-of-file imports (alongside the existing `import os`, `import shutil`, `import sys`).

- [ ] **Step 2: Call it from `main()`**

In `main()`, right after the existing block that copies `emotion_labels.csv`:

```python
    shutil.copyfile(EMOTION_LABELS_CSV, os.path.join(args.output_dir, "emotion_labels.csv"))
```

add:

```python
    layout_path = build_grid_layout(manifest_df, labels_df, args.output_dir)
    print(f"Grid layout: {layout_path}")
```

(`labels_df` is already loaded a few lines above this point — reuse it, don't re-read the CSV.)

- [ ] **Step 3: Write the test script**

Create `emotion_grid/test_build_grid.py`:

```python
"""
Test script for build_grid.py's grid_layout.json export
"""
import json
import os
import tempfile

import pandas as pd

from build_grid import build_grid_layout

print("="*80)
print("BUILD_GRID_LAYOUT TEST")
print("="*80)

# Synthetic manifest: 2 emotions, uneven counts (2 vs 1) to exercise the
# rectangular-padding path.
manifest_df = pd.DataFrame([
    {"id": "ang_00", "emotion": "ang", "thumbnail_path": "thumbnails/ang/ang_00.png"},
    {"id": "ang_01", "emotion": "ang", "thumbnail_path": "thumbnails/ang/ang_01.png"},
    {"id": "joy_00", "emotion": "joy", "thumbnail_path": "thumbnails/joy/joy_00.png"},
])
labels_df = pd.DataFrame([
    {"code": "ang", "full_name": "anger"},
    {"code": "joy", "full_name": "joy"},
])

with tempfile.TemporaryDirectory() as output_dir:
    layout_path = build_grid_layout(manifest_df, labels_df, output_dir)
    assert os.path.isfile(layout_path)

    with open(layout_path) as f:
        layout = json.load(f)

    assert layout["emotions"] == [
        {"code": "ang", "name": "anger"},
        {"code": "joy", "name": "joy"},
    ], layout["emotions"]
    assert layout["images_per_emotion"] == 2, layout["images_per_emotion"]
    assert len(layout["cells"]) == 4, len(layout["cells"])  # 2 emotions x 2 rows

    # row 0: ang_00, joy_00
    assert layout["cells"][0]["id"] == "ang_00"
    assert layout["cells"][0]["thumbnailPath"] == os.path.join(output_dir, "thumbnails/ang/ang_00.png")
    assert layout["cells"][1]["id"] == "joy_00"
    # row 1: ang_01, then padding (joy only has 1 image)
    assert layout["cells"][2]["id"] == "ang_01"
    assert layout["cells"][3] is None

    print("\n✓ emotions, images_per_emotion, row-major ordering, and padding are all correct")

print("\n" + "="*80)
print("ALL TESTS PASSED")
print("="*80)
```

- [ ] **Step 4: Run the test**

```bash
cd /home/tats/Documents/workspace/crocodile/emotion_grid
source ../biodata_pipeline/venv/bin/activate  # has pandas
python3 test_build_grid.py
```

Expected: `ALL TESTS PASSED`, no assertion errors.

- [ ] **Step 5: Regenerate the real `grid_layout.json` against actual data**

```bash
cd /home/tats/Documents/workspace/crocodile
source biodata_pipeline/venv/bin/activate
python3 emotion_grid/build_grid.py --skip-grimace
```

(`--skip-grimace` avoids needing `latent_pipeline/.venv`/torch for this check — the existing `manifest.csv` in the repo already has grimace categories from a prior run; re-running with `--skip-grimace` will drop `pri`/`lau` from a fresh manifest. If you want them included, activate `latent_pipeline/.venv` instead and omit the flag.)

Expected: prints `Grid layout: .../emotion_grid/data/grid_layout.json` at the end; `emotion_grid/data/grid_layout.json` exists and `python3 -c "import json; d=json.load(open('emotion_grid/data/grid_layout.json')); print(len(d['emotions']), d['images_per_emotion'], len(d['cells']))"` prints three numbers consistent with the manifest (22 emotions if run with grimace included, 20 if `--skip-grimace`, ×5 images).

- [ ] **Step 6: Confirm `grid_layout.json` is gitignored**

```bash
git status --short emotion_grid/
```

Expected: `emotion_grid/data/grid_layout.json` does **not** appear (it's inside the already-gitignored `emotion_grid/data/` path). If it does appear, stop — do not commit it; check `.gitignore` before proceeding.

- [ ] **Step 7: Commit**

```bash
git add emotion_grid/build_grid.py emotion_grid/test_build_grid.py
git commit -m "Added grid_layout.json export to build_grid.py"
```

---

### Task 2: Rename debug scripts to `latent_osc_debug_*`

**Files:**
- Rename: `live_pipeline/w_osc_debug_receiver.py` → `live_pipeline/latent_osc_debug_receiver.py`
- Rename: `live_pipeline/w_osc_debug_viewer.py` → `live_pipeline/latent_osc_debug_viewer.py`
- Modify: `live_pipeline/run_debug_receiver.sh`, `live_pipeline/run_debug_viewer.sh`
- Modify: `PIPELINE.md`, `INSTALL.md`

**Interfaces:**
- Produces: `latent_osc_debug_receiver.py --in-address` default `/crocodile/latent/final` — this is the address Task 9's integration test listens on to observe the module's final output.

- [ ] **Step 1: Rename the files with git mv**

```bash
cd /home/tats/Documents/workspace/crocodile
git mv live_pipeline/w_osc_debug_receiver.py live_pipeline/latent_osc_debug_receiver.py
git mv live_pipeline/w_osc_debug_viewer.py live_pipeline/latent_osc_debug_viewer.py
```

- [ ] **Step 2: Update `--in-address` defaults and usage docstrings**

In `live_pipeline/latent_osc_debug_receiver.py`, change:

```python
    parser.add_argument('--in-address', default='/crocodile/w')
```
to:
```python
    parser.add_argument('--in-address', default='/crocodile/latent/final')
```

And in its module docstring, change the usage lines:
```
    python live_pipeline/w_osc_debug_receiver.py
    python live_pipeline/w_osc_debug_receiver.py --in-port 1338 --expected-dim 512
```
to:
```
    python live_pipeline/latent_osc_debug_receiver.py
    python live_pipeline/latent_osc_debug_receiver.py --in-port 1338 --expected-dim 512
```

In `live_pipeline/latent_osc_debug_viewer.py`, change:
```
    python live_pipeline/w_osc_debug_viewer.py --config latent_pipeline/configs/default.yaml
    python live_pipeline/w_osc_debug_viewer.py --in-port 1338 --in-address /crocodile/w
```
to:
```
    python live_pipeline/latent_osc_debug_viewer.py --config latent_pipeline/configs/default.yaml
    python live_pipeline/latent_osc_debug_viewer.py --in-port 1338 --in-address /crocodile/latent/final
```
and:
```python
    parser.add_argument('--in-address', default='/crocodile/w')
```
to:
```python
    parser.add_argument('--in-address', default='/crocodile/latent/final')
```

- [ ] **Step 3: Update the wrapper shell scripts**

Find and update the script paths in each:

```bash
grep -n "w_osc_debug" live_pipeline/run_debug_receiver.sh live_pipeline/run_debug_viewer.sh
```

Replace `w_osc_debug_receiver.py` → `latent_osc_debug_receiver.py` in `run_debug_receiver.sh`, and `w_osc_debug_viewer.py` → `latent_osc_debug_viewer.py` in `run_debug_viewer.sh` (edit whatever line `grep` found — these are simple path references, not logic changes).

- [ ] **Step 4: Update doc references**

```bash
grep -n "w_osc_debug\|/crocodile/w\b" PIPELINE.md INSTALL.md
```

For each match: rename `w_osc_debug_receiver.py`/`w_osc_debug_viewer.py` to their new names, and change any bare `/crocodile/w` mention that refers to this stream to `/crocodile/latent/final` (leave any `/crocodile/w` mention alone if — after reading it in context — it turns out to refer to something Task 3/5 hasn't renamed yet; re-check after Task 3 and Task 5 land, since those introduce `/crocodile/latent/user` too and PIPELINE.md's port table should end up describing both streams accurately).

- [ ] **Step 5: Verify no stale references remain**

```bash
grep -rn "w_osc_debug" /home/tats/Documents/workspace/crocodile --include="*.py" --include="*.md" --include="*.sh" | grep -v venv
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add -A live_pipeline/latent_osc_debug_receiver.py live_pipeline/latent_osc_debug_viewer.py \
    live_pipeline/w_osc_debug_receiver.py live_pipeline/w_osc_debug_viewer.py \
    live_pipeline/run_debug_receiver.sh live_pipeline/run_debug_viewer.sh PIPELINE.md INSTALL.md
git commit -m "Renamed w_osc_debug_* scripts to latent_osc_debug_*"
```

(The `git add` of both old and new paths lets git record it as a rename rather than a delete+add; `git mv` in Step 1 already staged this correctly, so this is just adding the remaining modified files alongside it.)

---

### Task 3: `live_pipeline.py` — redirect W output to the controller

**Files:**
- Modify: `live_pipeline/live_pipeline.py:144`

**Interfaces:**
- Produces: `live_pipeline.py --out-port` default `9001`, `--out-address` default `/crocodile/latent/user` — this is the address/port Task 5's module listens on for the visitor's vector.

- [ ] **Step 1: Change the two defaults**

Change:
```python
    parser.add_argument('--out-host', default='127.0.0.1', help="Autolume's host")
    parser.add_argument('--out-port', type=int, default=1338, help="Autolume's default OSC input port")
    parser.add_argument('--out-address', default='/crocodile/w',
                        help='OSC address to send the 512-float W vector to -- must match '
                             "the address configured in Autolume's latent-vector OSC menu")
```
to:
```python
    parser.add_argument('--out-host', default='127.0.0.1', help="The live latent controller's host")
    parser.add_argument('--out-port', type=int, default=9001,
                        help="The live latent controller's OSC-in port (same port its status "
                             "broadcasts already use)")
    parser.add_argument('--out-address', default='/crocodile/latent/user',
                        help='OSC address to send the 512-float user W vector to -- the live '
                             'latent controller listens here, composites it with the actress '
                             'vector, and forwards the result on to Autolume')
```

And update the module-level docstring block near the top of the file. Change:
```
W (OSC out to Autolume, unchanged from before):
  One OSC message per finalized feature row during LIVE, 512 floats, on
  --out-address (default /crocodile/w). Autolume renders and displays the
  face itself; this script never touches StyleGAN2. Autolume's
  latent-vector OSC handler expects exactly this shape -- set its "vec"
  OSC address to match --out-address, and leave its "project" checkbox
  UNCHECKED (our W is already W-space, not Z-space).
```
to:
```
W (OSC out to the live latent controller, changed):
  One OSC message per finalized feature row during LIVE, 512 floats, on
  --out-address (default /crocodile/latent/user). This no longer goes
  straight to Autolume -- it goes to the live latent controller (Open
  Stage Control + crocodile-control-module.js, see
  live_pipeline/run_control_panel.sh), which composites it with the
  actress vector and forwards the result on to Autolume itself. This
  script never touches StyleGAN2.
```

- [ ] **Step 2: Sanity-check the defaults**

```bash
cd /home/tats/Documents/workspace/crocodile
grep -n "out-port\|out-address" live_pipeline/live_pipeline.py
```

Expected: shows `default='/crocodile/latent/user'` for `--out-address` and `default=9001` (or equivalent) for `--out-port`.

- [ ] **Step 3: Commit**

```bash
git add live_pipeline/live_pipeline.py
git commit -m "Redirected live_pipeline.py's W output to the controller"
```

---

### Task 4: Restructure `crocodile-control-panel.json` into tabs

**Files:**
- Modify: `live_pipeline/crocodile-control-panel.json`

**Interfaces:**
- Produces: root-level `tabs` array with tab ids `tab_session`, `tab_grid`, `tab_mixing`. Tasks 6 and 7 populate `tab_grid`'s and `tab_mixing`'s (currently empty) `widgets` arrays.

- [ ] **Step 1: Replace the root's `widgets`/`tabs` with a 3-tab structure**

In `live_pipeline/crocodile-control-panel.json`, the root `content` object currently has:

```json
    "widgets": [
      {
        "type": "panel",
        "id": "panel_session",
        ... (11 children: caption_id, session_id, start_calibration, stop_calibration,
             caption_emotion, calibration_emotion, start_live, recalibrate,
             refit_transformer, end_session, caption_status, status_display)
      }
    ],
    "tabs": []
```

Replace this whole `"widgets": [...], "tabs": []` block with:

```json
    "widgets": [],
    "onPreload": "globals.gridData = IMPORT{\"../emotion_grid/data/grid_layout.json\"}",
    "tabs": [
      {
        "type": "tab",
        "id": "tab_session",
        "label": "Session",
        "widgets": [
          {
            "type": "panel",
            "id": "panel_session",
            "top": 20,
            "left": 20,
            "width": 340,
            "height": 560,
            "lock": false,
            "visible": true,
            "interaction": true,
            "comments": "",
            "html": "Crocodile — Live Pipeline",
            "colorBg": "auto",
            "layout": "default",
            "value": "",
            "default": "",
            "address": "auto",
            "widgets": [
              { "type": "text", "id": "caption_id", "top": 20, "left": 20, "width": 300, "height": 20, "interaction": false, "align": "left", "value": "", "default": "Session ID (optional — press Enter to start):", "address": "auto" },
              { "type": "input", "id": "session_id", "top": 44, "left": 20, "width": 300, "height": 34, "align": "left", "asYouType": false, "numeric": false, "value": "", "default": "", "address": "/crocodile/session/start", "typeTags": "", "comments": "Type a session/visitor id and press Enter to send /crocodile/session/start [id]. Leave empty and press Enter to auto-generate one server-side." },
              { "type": "button", "id": "start_calibration", "top": 96, "left": 20, "width": 145, "height": 44, "label": "Start Calibration", "mode": "momentary", "value": "", "default": "", "address": "/crocodile/calibration/start", "typeTags": "" },
              { "type": "button", "id": "stop_calibration", "top": 96, "left": 175, "width": 145, "height": 44, "label": "Stop Calibration", "mode": "momentary", "value": "", "default": "", "address": "/crocodile/calibration/stop", "typeTags": "" },
              { "type": "text", "id": "caption_emotion", "top": 148, "left": 20, "width": 300, "height": 18, "interaction": false, "align": "left", "value": "", "default": "Calibration emotion (optional — press Enter to set):", "address": "auto" },
              { "type": "input", "id": "calibration_emotion", "top": 168, "left": 20, "width": 300, "height": 34, "align": "left", "asYouType": false, "numeric": false, "value": "", "default": "", "address": "/crocodile/calibration/set_emotion", "typeTags": "", "comments": "Type an emotion label (e.g. anx, neu, sad) and press Enter to tag subsequent calibration rows with it -- CALIBRATING only. Un-cued calibration defaults to 'neu'." },
              { "type": "button", "id": "start_live", "top": 210, "left": 20, "width": 300, "height": 56, "label": "Start Live", "colorWidget": "#2e7d32", "mode": "momentary", "value": "", "default": "", "address": "/crocodile/live/start", "typeTags": "" },
              { "type": "button", "id": "recalibrate", "top": 278, "left": 20, "width": 300, "height": 40, "label": "Recalibrate (while Live)", "mode": "momentary", "value": "", "default": "", "address": "/crocodile/calibration/recalibrate", "typeTags": "" },
              { "type": "button", "id": "refit_transformer", "top": 322, "left": 20, "width": 300, "height": 36, "label": "Refit Alignment (from calibration)", "mode": "momentary", "value": "", "default": "", "address": "/crocodile/calibration/refit", "typeTags": "", "comments": "Re-fit the live alignment transformer from the stored calibration buffer, without re-running calibration -- CALIBRATED or LIVE." },
              { "type": "button", "id": "end_session", "top": 366, "left": 20, "width": 300, "height": 44, "label": "End Session", "colorWidget": "#b71c1c", "mode": "momentary", "value": "", "default": "", "address": "/crocodile/session/end", "typeTags": "" },
              { "type": "text", "id": "caption_status", "top": 430, "left": 20, "width": 300, "height": 20, "interaction": false, "align": "left", "value": "", "default": "Status (from /crocodile/session/status):", "address": "auto" },
              { "type": "text", "id": "status_display", "top": 454, "left": 20, "width": 300, "height": 60, "interaction": false, "align": "center", "wrap": true, "value": "", "default": "IDLE", "address": "/crocodile/session/status" }
            ],
            "tabs": []
          }
        ],
        "tabs": []
      },
      {
        "type": "tab",
        "id": "tab_grid",
        "label": "Emotion Grid",
        "widgets": [],
        "tabs": []
      },
      {
        "type": "tab",
        "id": "tab_mixing",
        "label": "Mixing",
        "widgets": [],
        "tabs": []
      }
    ]
```

(`panel_session`'s 11 children are copied verbatim from the file's current content — no behavior change, just moved one level deeper under `tab_session`.)

- [ ] **Step 2: Validate JSON**

```bash
cd /home/tats/Documents/workspace/crocodile
python3 -c "import json; json.load(open('live_pipeline/crocodile-control-panel.json')); print('valid JSON')"
```

Expected: `valid JSON`.

- [ ] **Step 3: Smoke-test the session loads headless**

```bash
open-stage-control --no-gui --load live_pipeline/crocodile-control-panel.json --port 8090 --osc-port 9001 &
sleep 2
curl -sf http://127.0.0.1:8090 > /dev/null && echo "HTTP UI reachable"
kill %1
```

Expected: `HTTP UI reachable`, and no JSON/parse errors printed by the server before you kill it.

- [ ] **Step 4: Commit**

```bash
git add live_pipeline/crocodile-control-panel.json
git commit -m "Restructured control panel into tabs (Session / Emotion Grid / Mixing)"
```

---

### Task 5: Custom module — `crocodile-control-module.js`

**Files:**
- Create: `live_pipeline/crocodile-control-module.js`

**Interfaces:**
- Consumes: `emotion_grid/data/manifest.csv` (columns `id,emotion,feeling_it,thumbnail_path,pool_name,frame_number,w_000..w_511`, from Task 1's data — already present in the repo).
- Produces: the OSC address contract Tasks 6 and 7's widgets send to, all on the existing `9001` port:
  - `/crocodile/latent/user` (512 floats) — consumed, not forwarded
  - `/grid/select` (1 string arg: an `id` like `ang_00`)
  - `/transition/mode` (1 int arg: `0` = manual, `1` = auto-start)
  - `/transition/speed` (1 float arg, 0–1)
  - `/transition/running` (1 int arg: `0`/`1`)
  - `/mix/amount` (1 float arg, 0–1)
  - `/noise/amount` (1 float arg, 0–1)
  - Everything else passes through unchanged (session-control addresses, status broadcasts).
  - Pushes feedback to widgets `selected_id` (a `variable` widget Task 6 creates) and `transition_running` (the start/stop button Task 6 creates) via `receive("/SET", widgetId, value, {clientId})` — matching the exact call pattern already used by `open-stage-control/EmotionVectorsCustomModule.js` in this repo.
  - Sends the composited vector to `127.0.0.1:1338` `/crocodile/latent/final` (512 floats) every tick.

- [ ] **Step 1: Write the module**

Create `live_pipeline/crocodile-control-module.js`:

```js
// Custom Open Stage Control module for the live latent controller.
// Loaded via `open-stage-control --custom-module live_pipeline/crocodile-control-module.js`
// (see run_control_panel.sh). Holds all latent-vector state and math -- the
// session JSON's widgets only send lightweight control messages here and
// receive small feedback pushes back; the 512-float vectors themselves never
// round-trip through widget values.
//
// See docs/superpowers/specs/2026-09-10-live-latent-controller-design.md.

var fs = nativeRequire('fs')
var path = nativeRequire('path')

var MANIFEST_PATH = path.join(__dirname, '..', 'emotion_grid', 'data', 'manifest.csv')
var W_DIM = 512
var TICK_MS = 1000 / 30 // 30 Hz, matches Autolume's rendering cadence
var NOISE_SMOOTHING = 0.05 // lower = slower-drifting noise
var AUTOLUME_HOST = '127.0.0.1'
var AUTOLUME_PORT = 1338
var AUTOLUME_ADDRESS = '/crocodile/latent/final'

var manifestById = {} // id -> plain array of 512 floats

function loadManifest() {
    manifestById = {}
    var text
    try {
        text = fs.readFileSync(MANIFEST_PATH, 'utf8')
    } catch (e) {
        console.error('[crocodile-control-module] could not read manifest.csv at ' + MANIFEST_PATH + ': ' + e.message)
        return
    }
    var lines = text.split('\n').filter(function (l) { return l.trim().length > 0 })
    if (lines.length < 2) {
        console.error('[crocodile-control-module] manifest.csv has no data rows: ' + MANIFEST_PATH)
        return
    }
    var header = lines[0].split(',')
    var idCol = header.indexOf('id')
    var wStart = header.indexOf('w_000')
    if (idCol === -1 || wStart === -1) {
        console.error('[crocodile-control-module] manifest.csv missing id/w_000 columns')
        return
    }
    for (var i = 1; i < lines.length; i++) {
        var cols = lines[i].split(',')
        var id = cols[idCol]
        var w = new Array(W_DIM)
        for (var d = 0; d < W_DIM; d++) {
            w[d] = parseFloat(cols[wStart + d])
        }
        manifestById[id] = w
    }
    console.log('[crocodile-control-module] loaded ' + Object.keys(manifestById).length + ' emotion vectors from manifest.csv')
}

function zeros() {
    var a = new Array(W_DIM)
    for (var i = 0; i < W_DIM; i++) a[i] = 0
    return a
}

var state = {
    w_u: null,              // 512-array from live_pipeline.py, or null until first message
    current_w: zeros(),     // the "actress" vector, advances toward target_w each tick
    target_w: zeros(),
    target_id: null,
    mode: 'manual',         // 'auto' | 'manual'
    speed: 0.08,
    running: false,
    mix: 0.5,                // 1 = pure actress, 0 = pure incoming user vector
    noise_amount: 0,
    noise_state: zeros(),
}

var clients = []

app.on('open', function (data, client) {
    if (clients.indexOf(client.id) === -1) clients.push(client.id)
})
app.on('close', function (data, client) {
    var idx = clients.indexOf(client.id)
    if (idx !== -1) clients.splice(idx, 1)
})

function pushFeedback() {
    if (clients.length === 0) return
    receive('/SET', 'selected_id', state.target_id || '', { clientId: clients[0] })
    receive('/SET', 'transition_running', state.running ? 1 : 0, { clientId: clients[0] })
}

function tick() {
    if (state.target_id === null) return // nothing selected yet -- send nothing

    if (state.running) {
        for (var i = 0; i < W_DIM; i++) {
            state.current_w[i] += (state.target_w[i] - state.current_w[i]) * state.speed
        }
    }

    for (var j = 0; j < W_DIM; j++) {
        var white = Math.random() * 2 - 1
        state.noise_state[j] += (white - state.noise_state[j]) * NOISE_SMOOTHING
    }

    var w_u = state.w_u || state.current_w
    var final_w = new Array(W_DIM)
    for (var k = 0; k < W_DIM; k++) {
        final_w[k] = state.mix * state.current_w[k]
            + (1 - state.mix) * w_u[k]
            + state.noise_state[k] * state.noise_amount
    }

    send.apply(null, [AUTOLUME_HOST, AUTOLUME_PORT, AUTOLUME_ADDRESS].concat(final_w))
}

module.exports = {

    init: function () {
        loadManifest()
        setInterval(tick, TICK_MS)
    },

    oscInFilter: function (data) {
        var address = data.address
        var args = data.args

        if (address === '/crocodile/latent/user') {
            if (args.length !== W_DIM) {
                console.error('[crocodile-control-module] /crocodile/latent/user: expected ' + W_DIM + ' floats, got ' + args.length)
                return
            }
            state.w_u = args.slice()
            return // consumed, not forwarded to widgets
        }

        if (address === '/grid/select') {
            var id = args[0]
            var w = manifestById[id]
            if (!w) {
                console.error('[crocodile-control-module] /grid/select: unknown id ' + id)
                return
            }
            state.target_id = id
            state.target_w = w
            if (state.mode === 'auto') {
                state.running = true
            }
            pushFeedback()
            return
        }

        if (address === '/transition/mode') {
            state.mode = args[0] === 1 ? 'auto' : 'manual'
            return
        }

        if (address === '/transition/speed') {
            state.speed = args[0]
            return
        }

        if (address === '/transition/running') {
            state.running = args[0] === 1
            pushFeedback()
            return
        }

        if (address === '/mix/amount') {
            state.mix = args[0]
            return
        }

        if (address === '/noise/amount') {
            state.noise_amount = args[0]
            return
        }

        return data // everything else (session control, status, etc.) passes through unchanged
    },

}
```

- [ ] **Step 2: Verify the raw OSC args shape against the real server**

The exact shape of `data.args` inside `oscInFilter` (plain numbers vs. `{value, type}` objects) isn't fully nailed down by the public docs, and this module currently assumes plain numbers (matching the existing `EmotionVectorsCustomModule.js` pattern already in this repo). Confirm it directly:

```bash
cd /home/tats/Documents/workspace/crocodile
open-stage-control --no-gui --debug --load live_pipeline/crocodile-control-panel.json \
    --port 8090 --osc-port 9001 --custom-module live_pipeline/crocodile-control-module.js &
sleep 2
python3 -c "
from pythonosc.udp_client import SimpleUDPClient
c = SimpleUDPClient('127.0.0.1', 9001)
c.send_message('/mix/amount', 0.75)
"
sleep 1
kill %1
```

Watch the `--debug` console output for the incoming `/mix/amount` message. If `state.mix` ends up as `0.75` (add a temporary `console.log(state.mix)` at the end of the `/mix/amount` branch to check, then remove it), the plain-number assumption holds — no code change needed. If instead you see `args[0]` is an object like `{value: 0.75, type: 'f'}`, change every `args[0]` / `args.slice()` usage in `oscInFilter` to extract `.value` first (e.g. `args.map(function(a){ return a.value })`).

- [ ] **Step 3: Confirm manifest loads correctly**

With the server still running (or restart it as in Step 2), check the console for the `loaded N emotion vectors from manifest.csv` line, and confirm `N` matches the row count:

```bash
tail -n +2 emotion_grid/data/manifest.csv | wc -l
```

Expected: the two numbers match.

- [ ] **Step 4: Commit**

```bash
git add live_pipeline/crocodile-control-module.js
git commit -m "Added crocodile-control-module.js for latent vector state and math"
```

---

### Task 6: Emotion Grid tab UI

**Files:**
- Modify: `live_pipeline/crocodile-control-panel.json` (`tab_grid`'s `widgets` array, currently `[]` from Task 4)

**Interfaces:**
- Consumes: `globals.gridData` (set in Task 4's root `onPreload` from Task 1's `grid_layout.json`); the module's OSC contract from Task 5 (`/grid/select`, `/transition/mode`, `/transition/speed`, `/transition/running`, and the `selected_id`/`transition_running` feedback targets).

**Important — target, not just address:** `run_control_panel.sh` sets `--send 127.0.0.1:9000` as the *default* OSC send target (correct for the existing Session tab, since that's `live_pipeline.py`'s listen port). Every widget below that needs to reach the module (which intercepts on the server's own `9001`) must set an explicit `"target": "127.0.0.1:9001"` — otherwise its messages silently go to `9000` instead, where nothing listens for them. This applies to every new widget/matrix-instance-prop in this task and Task 7, not just ones using `address`.

- [ ] **Step 1: Add the grid + controls**

Replace `tab_grid`'s `"widgets": []` (from Task 4) with:

```json
"widgets": [
  {
    "type": "variable",
    "id": "selected_id",
    "value": ""
  },
  {
    "type": "text",
    "id": "grid_caption",
    "top": 20,
    "left": 20,
    "width": 600,
    "height": 20,
    "interaction": false,
    "align": "left",
    "default": "Columns = emotions, rows = images. Tap a thumbnail to set it as the target."
  },
  {
    "type": "matrix",
    "id": "grid_matrix",
    "top": 50,
    "left": 20,
    "width": 900,
    "height": 240,
    "layout": "grid",
    "gridTemplate": "JS{ return 'repeat(' + globals.gridData.emotions.length + ', 1fr)' }",
    "widgetType": "button",
    "quantity": "JS{ return globals.gridData.cells.length }",
    "props": {
      "width": 36,
      "height": 36,
      "mode": "momentary",
      "label": "",
      "target": "127.0.0.1:9001",
      "visible": "JS{ return globals.gridData.cells[$] !== null }",
      "css": "JS{ var c = globals.gridData.cells[$]; if (!c) return ''; var sel = @{selected_id} === c.id; return '.widget{background-image:url(file://' + c.thumbnailPath + ');background-size:cover;background-position:center;border:2px solid ' + (sel ? '#2e7d32' : 'transparent') + '}' }",
      "onTouch": "if (event.type !== 'start') return; var c = globals.gridData.cells[getIndex('this')]; if (!c) return; send('/grid/select', c.id)"
    }
  },
  {
    "type": "text",
    "id": "controls_caption",
    "top": 310,
    "left": 20,
    "width": 300,
    "height": 20,
    "interaction": false,
    "align": "left",
    "default": "Transition controls"
  },
  {
    "type": "switch",
    "id": "transition_mode",
    "top": 336,
    "left": 20,
    "width": 300,
    "height": 34,
    "values": { "Manual start": 0, "Auto-start on select": 1 },
    "default": 0,
    "address": "/transition/mode",
    "target": "127.0.0.1:9001"
  },
  {
    "type": "text",
    "id": "speed_caption",
    "top": 380,
    "left": 20,
    "width": 300,
    "height": 18,
    "interaction": false,
    "align": "left",
    "default": "Transition speed"
  },
  {
    "type": "fader",
    "id": "transition_speed",
    "top": 400,
    "left": 20,
    "width": 300,
    "height": 30,
    "horizontal": true,
    "range": { "min": 0, "max": 1 },
    "default": 0.08,
    "address": "/transition/speed",
    "target": "127.0.0.1:9001"
  },
  {
    "type": "button",
    "id": "transition_running",
    "top": 440,
    "left": 20,
    "width": 300,
    "height": 44,
    "label": "#{@{this.value} === 1 ? 'Stop' : 'Start'}",
    "mode": "toggle",
    "colorWidget": "#{@{this.value} === 1 ? '#2e7d32' : 'auto'}",
    "value": 0,
    "default": 0,
    "address": "/transition/running",
    "target": "127.0.0.1:9001"
  },
  {
    "type": "text",
    "id": "target_display",
    "top": 494,
    "left": 20,
    "width": 300,
    "height": 20,
    "interaction": false,
    "align": "left",
    "value": "#{'Target: ' + (@{selected_id} || '(none)')}"
  }
]
```

- [ ] **Step 2: Validate JSON**

```bash
cd /home/tats/Documents/workspace/crocodile
python3 -c "import json; json.load(open('live_pipeline/crocodile-control-panel.json')); print('valid JSON')"
```

- [ ] **Step 3: Manual smoke test (GUI)**

```bash
live_pipeline/run_control_panel.sh --custom-module live_pipeline/crocodile-control-module.js
```

(This flag will become the permanent default once Task 8 lands; passing it explicitly here lets you test before that.) In the window that opens: switch to the "Emotion Grid" tab, confirm the 22×5 (or however many emotions your local `grid_layout.json` has) grid of thumbnails renders with real images, tap one and confirm it gets a green border, and that `target_display` updates to show its id. Toggle "Start"/"Stop" and confirm the button's label/color reflect the state.

- [ ] **Step 4: Commit**

```bash
git add live_pipeline/crocodile-control-panel.json
git commit -m "Added Emotion Grid tab: thumbnail grid and transition controls"
```

---

### Task 7: Mixing tab UI

**Files:**
- Modify: `live_pipeline/crocodile-control-panel.json` (`tab_mixing`'s `widgets` array, currently `[]` from Task 4)

**Interfaces:**
- Consumes: the module's `/mix/amount` and `/noise/amount` addresses from Task 5.

**Same target caveat as Task 6**: both faders need `"target": "127.0.0.1:9001"` explicitly, since the default send target (`127.0.0.1:9000`, set in `run_control_panel.sh`) goes to `live_pipeline.py`, not the module.

- [ ] **Step 1: Add the two faders**

Replace `tab_mixing`'s `"widgets": []` (from Task 4) with:

```json
"widgets": [
  {
    "type": "text",
    "id": "mix_caption",
    "top": 20,
    "left": 20,
    "width": 300,
    "height": 18,
    "interaction": false,
    "align": "left",
    "default": "Actress / User mix (100% = pure actress)"
  },
  {
    "type": "fader",
    "id": "mix_amount",
    "top": 42,
    "left": 20,
    "width": 300,
    "height": 30,
    "horizontal": true,
    "range": { "min": 0, "max": 1 },
    "default": 0.5,
    "address": "/mix/amount",
    "target": "127.0.0.1:9001"
  },
  {
    "type": "text",
    "id": "noise_caption",
    "top": 90,
    "left": 20,
    "width": 300,
    "height": 18,
    "interaction": false,
    "align": "left",
    "default": "Noise amount"
  },
  {
    "type": "fader",
    "id": "noise_amount",
    "top": 112,
    "left": 20,
    "width": 300,
    "height": 30,
    "horizontal": true,
    "range": { "min": 0, "max": 1 },
    "default": 0,
    "address": "/noise/amount",
    "target": "127.0.0.1:9001"
  }
]
```

- [ ] **Step 2: Validate JSON**

```bash
cd /home/tats/Documents/workspace/crocodile
python3 -c "import json; json.load(open('live_pipeline/crocodile-control-panel.json')); print('valid JSON')"
```

- [ ] **Step 3: Manual smoke test**

```bash
live_pipeline/run_control_panel.sh --custom-module live_pipeline/crocodile-control-module.js
```

Switch to "Mixing", drag both faders, confirm they move smoothly and don't error in the console.

- [ ] **Step 4: Commit**

```bash
git add live_pipeline/crocodile-control-panel.json
git commit -m "Added Mixing tab: actress/user mix and noise faders"
```

---

### Task 8: Wire `run_control_panel.sh` to load the module by default

**Files:**
- Modify: `live_pipeline/run_control_panel.sh`

- [ ] **Step 1: Add the flag**

```bash
exec open-stage-control \
    --load "$REPO_ROOT/live_pipeline/crocodile-control-panel.json" \
    --port 8090 \
    --send 127.0.0.1:9000 \
    --osc-port 9001 \
    --custom-module "$REPO_ROOT/live_pipeline/crocodile-control-module.js" \
    "$@"
```

Update the script's header comment to mention the module and the new `/crocodile/latent/*` addresses alongside the existing session-control ones it already documents.

- [ ] **Step 2: Confirm it launches clean**

```bash
cd /home/tats/Documents/workspace/crocodile
live_pipeline/run_control_panel.sh --no-gui &
sleep 2
curl -sf http://127.0.0.1:8090 > /dev/null && echo "OK"
kill %1
```

Expected: `OK`, and the console log shows the module's `loaded N emotion vectors` line (confirms the module actually got loaded, not just the session).

- [ ] **Step 3: Commit**

```bash
git add live_pipeline/run_control_panel.sh
git commit -m "Wired crocodile-control-module.js into run_control_panel.sh by default"
```

---

### Task 9: End-to-end integration test

**Files:**
- Create: `live_pipeline/test_latent_controller_integration.py`

**Interfaces:**
- Exercises the full chain: fake `live_pipeline.py` → controller (`9001`) → fake Autolume (`latent_osc_debug_receiver.py` on `1338`), plus the UI control addresses from Tasks 6/7.

- [ ] **Step 1: Write the test script**

Create `live_pipeline/test_latent_controller_integration.py`:

```python
"""
Test script for the live latent controller: launches the controller headless,
a debug receiver standing in for Autolume, feeds it fake OSC input, and
checks the composited output behaves as designed.

Run from the repo root with biodata_pipeline/venv active (has python-osc):
    source biodata_pipeline/venv/bin/activate
    python3 live_pipeline/test_latent_controller_integration.py
"""
import csv
import subprocess
import threading
import time

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

CONTROLLER_PORT = 9001
AUTOLUME_PORT = 1338
W_DIM = 512

print("="*80)
print("LIVE LATENT CONTROLLER INTEGRATION TEST")
print("="*80)

# --- fake Autolume: just record every /crocodile/latent/final we receive ---
received = []

def on_final(unused_address, *args):
    received.append(list(args))

dispatcher = Dispatcher()
dispatcher.map('/crocodile/latent/final', on_final)
fake_autolume = BlockingOSCUDPServer(('127.0.0.1', AUTOLUME_PORT), dispatcher)
fake_autolume_thread = threading.Thread(target=fake_autolume.serve_forever, daemon=True)
fake_autolume_thread.start()

# --- launch the controller headless ---
controller = subprocess.Popen([
    'open-stage-control', '--no-gui',
    '--load', 'live_pipeline/crocodile-control-panel.json',
    '--port', '8092',  # avoid colliding with a real instance on 8090
    '--osc-port', str(CONTROLLER_PORT),
    '--custom-module', 'live_pipeline/crocodile-control-module.js',
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

time.sleep(3)  # let the server + module init() (manifest load) finish

try:
    client = SimpleUDPClient('127.0.0.1', CONTROLLER_PORT)

    # 1. Before any /grid/select, the module has no target yet and tick()
    #    should send nothing at all.
    time.sleep(0.5)
    received.clear()
    time.sleep(0.5)
    assert len(received) == 0, f"expected no output before selecting a target, got {len(received)}"
    print("✓ no output before any target is selected and transition started")

    # 2. Select a real emotion id from the manifest and start the transition.
    with open('emotion_grid/data/manifest.csv') as f:
        first_row = next(csv.DictReader(f))
    target_id = first_row['id']

    client.send_message('/grid/select', target_id)
    client.send_message('/transition/speed', 1.0)  # jump immediately for a fast test
    client.send_message('/transition/running', 1)

    time.sleep(0.5)
    assert len(received) > 0, "expected output after selecting a target and starting the transition"
    last = received[-1]
    assert len(last) == W_DIM, f"expected {W_DIM} floats, got {len(last)}"
    print(f"✓ received {len(received)} /crocodile/latent/final messages, each with {W_DIM} floats")

    # 3. Feed a fake user vector and set mix to 0 (pure user) -- output should
    #    converge toward it.
    fake_w_u = [0.0] * W_DIM
    fake_w_u[0] = 42.0
    client.send_message('/crocodile/latent/user', fake_w_u)
    client.send_message('/mix/amount', 0.0)
    time.sleep(0.5)
    last = received[-1]
    assert abs(last[0] - 42.0) < 5.0, f"expected output to converge toward the user vector (42.0), got {last[0]}"
    print(f"✓ mix=0 output converges toward the incoming user vector (got {last[0]:.2f}, expected near 42.0)")

    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80)

finally:
    controller.terminate()
    controller.wait(timeout=5)
    fake_autolume.shutdown()
```

- [ ] **Step 2: Run it**

```bash
cd /home/tats/Documents/workspace/crocodile
source biodata_pipeline/venv/bin/activate
python3 live_pipeline/test_latent_controller_integration.py
```

Expected: `ALL TESTS PASSED`.

If the test hangs or the controller process never produces output, check `controller.stdout` for module load errors (a bad `require`/`nativeRequire` path is the most likely culprit). Temporarily wrap the body in `try`/`except AssertionError` and add `print(controller.stdout.read())` before `raise` to see the server's console output.

- [ ] **Step 3: Commit**

```bash
git add live_pipeline/test_latent_controller_integration.py
git commit -m "Added end-to-end integration test for the live latent controller"
```

---

## Manual steps outside this repo (not part of any task above)

- **Autolume**: update its OSC input address setting (in its own UI, per `README.md`'s existing instructions on this) from `/crocodile/w` to `/crocodile/latent/final`. Its port (`1338`) and the fact that it must run on the same machine for `file://`-less operation are unaffected.
- The separately-flagged `in_ip = "127.0.0.1"` hardcode in `~/Documents/workspace/autolume/modules/visualizer.py:88` is a pre-existing bug unrelated to this work (found while debugging the original TouchDesigner OSC issue) — still not fixed unless/until asked.
