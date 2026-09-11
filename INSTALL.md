# Installing and Testing the Live Pipeline

Step-by-step setup for `live_pipeline/` — the runtime layer that takes live
biodata over OSC and outputs a StyleGAN2 W latent vector over OSC for
Autolume. See [PIPELINE.md](PIPELINE.md#live-pipeline-biodata-osc--w-osc--autolume)
for the full design (session state machine, OSC protocol, recalibration);
this doc only covers getting a machine set up and confirming it works.

## 0. What's public vs. private

This repo does **not**, and will never, contain real biodata recordings or
the trained StyleGAN2 model — both identify a real person (the actress the
model was trained on, and any visitor whose physiological signals get
recorded) and are excluded from git entirely (see `.gitignore`: `*.csv`,
`*.pkl`, `*.joblib`, `models/`, `biodata_pipeline/models/`,
`latent_pipeline/outputs/`). You'll need to get these from your team's
private storage, or generate synthetic stand-ins where noted below.

| Needed for | What | Where it comes from |
|---|---|---|
| `live_pipeline.py` (**required**) | Trained regressor (`latent_pipeline/outputs/stage5_regressor_online/regressor.joblib`) | Private — ask a teammate, or train your own (`latent_pipeline/PLAN.md`, Stage 5) |
| `live_pipeline.py` (only for a **real calibration** — §5b) | Trained alignment transformer (`biodata_pipeline/models/transformer_ot_classconditional_online.pkl`) | Private — ask a teammate, or train your own (`biodata_pipeline/scripts/train_transformer.py`). Not needed for §5's quick check, which uses a public transformer pre-fit on synthetic data instead (`live_pipeline/data/synthetic_test_transformer.pkl`, committed) |
| `live_pipeline.py` (optional — enables live per-visitor alignment fitting) | The actress' (Laurence's) online-schema reference features (`biodata_pipeline/data/processed/continuous_features_online.csv` — NOT `erin_features_online.csv`, Erin is a separate test subject, not the actress), passed as `--reference-features` | Private — ask a teammate. Without it, every session just uses the static transformer above (unchanged behavior) — see PIPELINE.md's "Live per-visitor alignment fit" |
| Feeding the pipeline data | Real biodata recordings, or `--calibration-csv` priming | Private — optional; §5's quick check replays a committed synthetic recording instead, and §5b's full calibration walkthrough generates fresh synthetic data if you don't have a real one |
| Visual check only — `latent_osc_debug_viewer.py` (debug overlay) or `live_viewer.py` (no overlay, can replace Autolume outright) | StyleGAN2 checkpoint `models/finalModel_Crocodile.pkl` (~430MB) | Private — ask a teammate. Not needed for `live_pipeline.py` itself or for `latent_osc_debug_receiver.py` (§5) |
| Same as above | `stylegan_Autolume` code (`dnnlib`/`legacy.py`/`torch_utils`, imported by `latent_pipeline/models/stylegan.py` to load and run the checkpoint) | **Public** — a git submodule at `latent_pipeline/stylegan_Autolume`, not initialized by §1's default `--recursive` init, see §4b. Not the same thing as Autolume (the live performance app) below, despite the name |
| Real deployment only | Autolume, the separate live performance app | Private/separate project — not needed to install or test this repo |
| GUI session control (optional) | Open Stage Control | Public — §3 |
| Emotion Grid tab (required for the live latent controller) | `emotion_grid/data/` (`manifest.csv`, `grid_layout.json`, `thumbnails/`) | Private — see §4, or build it yourself with `emotion_grid/build_grid.py` from the private `latent_pipeline` dataset |

**In short**: you can install and fully test `live_pipeline.py`'s OSC
plumbing (§1–§5) with only the private regressor and no real biodata,
private alignment transformer, or StyleGAN2 model at all, using the
committed synthetic fixtures and the lightweight debug receiver.

## 1. Clone the repo

```bash
git clone <this repo's URL> crocodile
cd crocodile
git checkout pipeline_develop
git submodule update --init --recursive   # BioDataFeatureExtract/libraries/ — not needed for live_pipeline, but harmless
```

This deliberately skips `latent_pipeline/stylegan_Autolume` (it's marked
`update = none` in `.gitmodules` since it's a large codebase most setups
don't need) — see §4b if you want the visual viewer.

**Note the branch checkout above is required, not optional**: `master`
predates the entire `biodata_pipeline`/`latent_pipeline`/`live_pipeline`
restructuring this doc describes — none of the directories or files
referenced anywhere in this guide exist on `master`. All active work,
including everything the live pipeline needs, lives on `pipeline_develop`.

## 2. Set up the two Python environments

`live_pipeline/` has no venv of its own — each script runs under one of
`biodata_pipeline`'s or `latent_pipeline`'s existing venvs (both Python
3.12), via wrapper scripts that pin the right interpreter. Set up whichever
you need:

```bash
# Required for live_pipeline.py, session_control.py, replay_biodata_as_osc.py,
# generate_synthetic_biodata.py, latent_osc_debug_receiver.py -- i.e. everything
# except the visual debug viewer.
python3 -m venv biodata_pipeline/venv
biodata_pipeline/venv/bin/pip install -r biodata_pipeline/requirements.txt

# Only needed for latent_osc_debug_viewer.py (renders via StyleGAN2 -- needs torch).
python3 -m venv latent_pipeline/.venv
latent_pipeline/.venv/bin/pip install -r latent_pipeline/requirements.txt
```

If you don't plan to run the visual debug viewer yet, skip the
`latent_pipeline/.venv` setup — `latent_osc_debug_receiver.py` (§5) covers
smoke-testing the OSC output without it.

## 3. (Optional, but required for live output) Install Open Stage Control

Needed if you want the GUI control panel
(`live_pipeline/crocodile-control-panel.json`) instead of driving sessions
from the command line with `session_control.py`. Note that it's no longer
purely optional for a full live run: `crocodile-control-module.js` (loaded by
`run_control_panel.sh`, part of this same panel) is what composites the
visitor's vector with the operator's emotion selection and actually sends to
Autolume — `live_pipeline.py` no longer talks to Autolume directly — so some
form of Open Stage Control (with the control module loaded) must be running
for anything to reach Autolume, even if you drive session state itself from
`session_control.py` on the command line. Get a package for your OS
from the [Open Stage Control releases page](https://openstagecontrol.ammd.net/)
and install it so the `open-stage-control` binary is on your `PATH`
(e.g. `sudo dpkg -i open-stage-control_*.deb` on Debian/Ubuntu). Verify with:

```bash
open-stage-control --version
```

## 4. Get the private data

Every private/gitignored asset this pipeline needs lives in one place, a
`crocodile-private/` folder, kept entirely outside this repo (so there's no
risk of any of it ever being swept into git). Ask a teammate for a copy —
it should contain, mirroring this repo's own relative layout:

```
crocodile-private/
├── models/finalModel_Crocodile.pkl                          (needed for §4b, the visual viewer)
├── latent_pipeline/outputs/stage5_regressor_online/          (needed even for §5's quick check)
├── latent_pipeline/outputs/stage5_regressor(_continuous)/    (alternate alignment-method regressors)
├── biodata_pipeline/models/*.pkl                              (needed for §5b's full calibration)
├── biodata_pipeline/data/raw/                                (real recordings, optional)
├── biodata_pipeline/data/processed/continuous_features_online.csv  (the actress' reference features, optional)
├── emotion_grid/data/ (manifest.csv, grid_layout.json, thumbnails/) (needed for the Emotion Grid tab)
├── live_pipeline/data/erin_*.csv + sessions/                 (real recordings, optional)
└── luana-Crocodile-with-data/                                (real recordings, optional)
```

This is a curated subset, not a full mirror of every gitignored file on
someone's machine — it deliberately excludes bulky Stage 2 encoder-training
artifacts (`latent_pipeline/outputs/best.pt`, `recon_epoch_*.png`, etc.) and
`biodata_pipeline/data/processed/`'s many exploratory feature-extraction
experiment files, since nothing reads those by path outside of re-training
the encoder from scratch — ask a teammate directly if you need those too.

Place the folder as a sibling directory next to your checkout (i.e. if this
repo is at `workspace/crocodile`, put it at `workspace/crocodile-private`),
then symlink everything into place:

```bash
scripts/link_private_data.sh
```

(Pass a path as an argument if you put `crocodile-private/` somewhere else.)
This is safe to re-run any time — it skips anything already correctly
linked and never overwrites a real file/directory already sitting at one of
these paths. Unix-only (macOS or Linux, or Windows via WSL) — see §6 if
you're on plain Windows.

If you'd rather not use symlinks, just place the individual files at the
same relative paths yourself, or point `--regressor`/`--transformer` (§5/§5b)
at wherever you put them directly.

If you want the visual viewer (`latent_osc_debug_viewer.py`/`live_viewer.py`)
rather than just the OSC-plumbing check in §5, you also need a small piece
of **public** code alongside the checkpoint above — see §4b.

## 4b. Install stylegan_Autolume (only for the visual viewer)

`latent_pipeline/models/stylegan.py` loads and runs the checkpoint by
importing `dnnlib`, `legacy`, and `torch_utils` from `stylegan_Autolume` — a
StyleGAN3 fork maintained for the Autolume project. Despite the similar
name, this is a **different, unrelated thing** from Autolume itself (the
separate live performance app in §0's table) — it's just the bare synthesis
code, not the performance app.

It's a **git submodule** at `latent_pipeline/stylegan_Autolume`, pinned to a
known-good commit — pulled in
[our own fork](https://github.com/mXlab/stylegan_Autolume) rather than
upstream (`petercmh01/stylegan_Autolume`) so a deleted/renamed/rewritten
upstream repo can't break this. It's not part of §1's default
`--recursive` init (it's a large, optional codebase most setups don't need),
so init it explicitly:

```bash
git submodule update --init --checkout latent_pipeline/stylegan_Autolume
```

Only the trained checkpoint (obtained above) is private — the code itself,
and no separate environment or `pip install` for it, is needed:
`latent_pipeline/.venv` already has everything `dnnlib`/`legacy`/
`torch_utils` import at runtime, right down to the exact
`setuptools==70.2.0` pin its `torch_utils` needs (see the comment in
`latent_pipeline/requirements.txt`) — and `load_stylegan()` forces
pure-PyTorch reference ops specifically to avoid needing a matching CUDA
toolkit/compiler to JIT-compile its custom kernels. Its own
`environment.yml` (conda, CUDA 11.1, a GUI visualizer's extra dependencies)
is for its own training/visualization tools and can be ignored here.

`paths.stylegan_code` in `latent_pipeline/configs/default.yaml` is already
set to the right relative path (`latent_pipeline/stylegan_Autolume`) and
needs no editing. `paths.repo_root` still does — it's a hardcoded absolute
path from whoever last edited that file:

```yaml
# latent_pipeline/configs/default.yaml
paths:
  repo_root: /absolute/path/to/your/crocodile/checkout   # <- edit this
  stylegan_code: latent_pipeline/stylegan_Autolume        # already correct
  stylegan_model: models/finalModel_Crocodile.pkl         # relative to repo_root
```

Verify it worked with:

```bash
live_pipeline/run_debug_viewer.sh
```

If `dnnlib`/`legacy` fail to import, confirm the submodule actually has
content (`ls latent_pipeline/stylegan_Autolume/dnnlib` — an empty directory
means the `git submodule update --init --checkout` above was skipped or failed).

To update the pinned commit later (only if you have a specific reason to —
this is inference-only code loading a frozen checkpoint, so there's rarely
a need): `cd latent_pipeline/stylegan_Autolume && git checkout
<new-commit>`, then commit the updated submodule pointer from the repo
root after confirming the viewer still works.

## 5. Test the live pipeline end-to-end

Open several terminals, all from the repo root. This is the quickest
check — no calibration, no button-pressing, and no real recording, using two
files already committed under `live_pipeline/data/` (both 100% synthetic,
NeuroKit2-generated — see README.md's "Simplest possible test" for how
they were made and why they're safe to be in the repo).

**Terminal 1 — the core server**, with `--auto-start` so it goes straight to
`LIVE` on launch (no `session/start`/`live/start` OSC messages needed
either):

```bash
live_pipeline/run_live.sh \
    --regressor latent_pipeline/outputs/stage5_regressor_online/regressor.joblib \
    --transformer live_pipeline/data/synthetic_test_transformer.pkl \
    --auto-start
```

Wait for `Live output started`. Note this uses the committed
`synthetic_test_transformer.pkl`, not the private
`transformer_ot_classconditional_online.pkl` from §4 — the private one is
only needed for a real calibration (§5b below).

**Terminal 2 — the latent controller.** This composites the visitor's vector
with the actress' selection and is what actually forwards to Autolume —
`live_pipeline.py` no longer talks to Autolume directly, so nothing reaches
it without this running, even in this simplified test:

```bash
live_pipeline/run_control_panel.sh
```

Open `http://127.0.0.1:8090` and pick any thumbnail in the Emotion Grid tab
(§4) — with "Auto-start on select" checked (the default), that immediately
starts the actress-side transition.

**Terminal 3 — a lightweight W receiver (no StyleGAN2/torch needed):**

```bash
live_pipeline/run_debug_receiver.sh
```

This is the recommended first check — it just confirms 512-float W vectors
are actually arriving at the expected rate on Autolume's own port/address
(1338, `/crocodile/latent/final`), without needing the private StyleGAN2
model at all. Once you have that model installed (§4b), run one of these
instead/in addition for an actual visual check:

```bash
live_pipeline/run_debug_viewer.sh    # small preview window with a debug overlay
live_pipeline/run_live_viewer.sh     # larger, no overlay by default -- can replace Autolume outright
```

**Terminal 4 — replay the committed synthetic visitor recording:**

```bash
live_pipeline/run_replay.sh --input live_pipeline/data/synthetic_test_live.csv --speed 1.0 --loop
```

**What to expect**: Terminal 1 logs `Session started`/`Live output started`
right on launch. Once replay starts, Terminal 3's debug receiver (or viewer)
should show a steady `received=... rate=.../s norm=...` — that confirms the
whole chain (session → alignment → regressor → latent controller → OSC out)
is wired correctly. If no `WARNING` lines appeared anywhere, the pipeline is
correctly installed end-to-end.

### 5b. Testing a full calibration scenario (optional)

The above skips calibration entirely by using a transformer pre-fit offline
on synthetic data. To exercise the full state machine (`session/start` →
`calibration/start` → `calibration/stop` → `live/start`) and a live
per-visitor alignment fit instead, first generate your own synthetic
segments — or, if you have a real recording (raw `heart`/`gsr`/`respiration`
CSV columns, 100Hz), skip straight to the walkthrough below and use it
directly with `replay_biodata_as_osc.py`:

```bash
mkdir -p live_pipeline/data
# Two independent segments (different --seed = non-overlapping signals) --
# calibration should always be a different recording than what you replay
# as "live" (see PIPELINE.md for why).
live_pipeline/run_generate_synthetic.sh --duration 30 --seed 1 \
    --output live_pipeline/data/synthetic_calibration.csv
live_pipeline/run_generate_synthetic.sh --duration 60 --seed 2 \
    --output live_pipeline/data/synthetic_live.csv
```

Then, in place of §5's Terminal 1 and 4 commands (Terminals 2 and 3 are the
same as above):

**Terminal 1 — the core server**, using the private transformer this time:

```bash
live_pipeline/run_live.sh \
    --regressor latent_pipeline/outputs/stage5_regressor_online/regressor.joblib \
    --transformer biodata_pipeline/models/transformer_ot_classconditional_online.pkl
```

Wait for `Waiting for /crocodile/session/start ...`.

**Terminal 4 — drive the session** (CLI shown; `run_session_control.sh`
works the same way as the panel's buttons):

```bash
live_pipeline/run_session_control.sh --start-session test
live_pipeline/run_session_control.sh --start-calibration
live_pipeline/run_replay.sh --input live_pipeline/data/synthetic_calibration.csv --speed 1.0
live_pipeline/run_session_control.sh --stop-calibration
live_pipeline/run_session_control.sh --start-live
live_pipeline/run_replay.sh --input live_pipeline/data/synthetic_live.csv --speed 1.0
live_pipeline/run_session_control.sh --end-session
```

**What to expect**: Terminal 1 logs each state transition
(`Session started`, `Calibration started`, `Live output started`, ...) and
ends with `Session ended: test. Rows sent: N, skipped: 0`. Terminal 3's
debug receiver prints periodic `received=... rate=.../s norm=...` lines
while the second `run_replay.sh` call is running, and `N` there should
match Terminal 1's `Rows sent` count. If both match and no `WARNING` lines
appeared, the full calibration flow is correctly wired end-to-end.

## 6. Troubleshooting

- **`EADDRINUSE` / "address already in use"**: something is already bound to
  one of the ports involved (`9000` control+biodata, `9001` status, `1338`
  W-out, `8090` Open Stage Control's HTTP UI). Check for a stale process
  from a previous failed run (`ps aux | grep live_pipeline`) or pass
  different `--in-port`/`--out-port`/`--status-out-port` (and matching
  `--port`/`--send`/`--osc-port` to `session_control.py`/
  `run_control_panel.sh`) to run on alternate ports.
- **No W vectors received in the debug receiver/viewer**: confirm the core
  server actually reached the `LIVE` phase (`Live output started` in its
  log) — biodata arriving before `live/start` is intentionally discarded
  (see PIPELINE.md's state machine). Also confirm the latent controller
  (`run_control_panel.sh`) is running — nothing reaches port `1338` without
  it, even in §5's simplified test — and that `replay_biodata_as_osc.py` and
  `live_pipeline.py` agree on `--port`/`--address`.
- **`ModuleNotFoundError`**: you're running a script with the wrong venv's
  interpreter directly instead of through its `run_*.sh` wrapper, or you
  installed before the requirements file included `neurokit2`/`joblib`/
  `python-osc` — re-run `pip install -r biodata_pipeline/requirements.txt`.
- **On Windows**: every wrapper script in this project is a `.sh` file, so
  you need WSL or Git Bash regardless of §4. Symlinks specifically (what
  `scripts/link_private_data.sh` creates) need either WSL with Developer
  Mode enabled, or admin/elevated privileges in Git Bash — plain Windows
  accounts can't create them by default, and Git Bash may silently fall
  back to copying the file instead of linking it (harmless, just means
  updates to `crocodile-private/` won't be picked up automatically). If
  that's not workable, skip the script and just copy the individual files
  from `crocodile-private/` to the paths §4 lists directly.
- Anything else: see PIPELINE.md's "Two real bugs were caught building
  this" section for the two subtlest failure modes found so far (a
  calibration/live speed artifact and a peak-detector priming bug), both
  already fixed but documented in case something similar resurfaces.
