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
| `live_pipeline.py` (**required**) | Trained alignment transformer (`biodata_pipeline/models/transformer_ot_classconditional_online.pkl`) | Private — ask a teammate, or train your own (`biodata_pipeline/scripts/train_transformer.py`) |
| `live_pipeline.py` (optional — enables live per-visitor alignment fitting) | The actress' (Laurence's) online-schema reference features (`biodata_pipeline/data/processed/continuous_features_online.csv` — NOT `erin_features_online.csv`, Erin is a separate test subject, not the actress), passed as `--reference-features` | Private — ask a teammate. Without it, every session just uses the static transformer above (unchanged behavior) — see PIPELINE.md's "Live per-visitor alignment fit" |
| Feeding the pipeline data | Real biodata recordings, or `--calibration-csv` priming | Private — optional; §5 generates synthetic data as a substitute |
| `latent_osc_debug_viewer.py` (visual check only) | StyleGAN2 checkpoint `models/finalModel_Crocodile.pkl` (~430MB) + the `stylegan_Autolume` code repo | Private — ask a teammate. Not needed for `live_pipeline.py` itself or for `latent_osc_debug_receiver.py` (§6) |
| Real deployment only | Autolume, the separate live performance app | Private/separate project — not needed to install or test this repo |
| GUI session control (optional) | Open Stage Control | Public — §3 |

**In short**: you can install and fully test `live_pipeline.py`'s OSC
plumbing (§1–§6) with only the two trained artifacts above and no real
biodata or StyleGAN2 model at all, using synthetic data and the lightweight
debug receiver.

## 1. Clone the repo

```bash
git clone <this repo's URL> crocodile
cd crocodile
git checkout pipeline_develop
git submodule update --init --recursive   # BioDataFeatureExtract/libraries/ — not needed for live_pipeline, but harmless
```

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
`latent_pipeline/.venv` setup — `latent_osc_debug_receiver.py` (§6) covers
smoke-testing the OSC output without it.

## 3. (Optional) Install Open Stage Control

Only needed if you want the GUI control panel
(`live_pipeline/crocodile-control-panel.json`) instead of driving sessions
from the command line with `session_control.py`. Get a package for your OS
from the [Open Stage Control releases page](https://openstagecontrol.ammd.net/)
and install it so the `open-stage-control` binary is on your `PATH`
(e.g. `sudo dpkg -i open-stage-control_*.deb` on Debian/Ubuntu). Verify with:

```bash
open-stage-control --version
```

## 4. Get the private trained artifacts

Obtain from a teammate (or your project's private storage — ask whoever
last trained them):

- `latent_pipeline/outputs/stage5_regressor_online/regressor.joblib`
- `biodata_pipeline/models/transformer_ot_classconditional_online.pkl`

Place them at those exact relative paths (create the directories if they
don't exist) — the commands below reference them there. If your files live
elsewhere or under different names, just point `--regressor`/`--transformer`
(§6) at wherever you put them.

If you also obtained the StyleGAN2 checkpoint and `stylegan_Autolume` code
(for the visual debug viewer or `latent_pipeline` work generally), place the
checkpoint at `models/finalModel_Crocodile.pkl` (repo root, separate from
`latent_pipeline/models/`, which holds code not the weights) and edit
`latent_pipeline/configs/default.yaml`'s `paths.repo_root` and
`paths.stylegan_code` to match your machine — both are currently hardcoded
absolute paths from whoever last edited that file.

## 5. Generate synthetic biodata (if you don't have a real recording)

`generate_synthetic_biodata.py` uses NeuroKit2's own signal simulators to
produce a CSV with the same columns `replay_biodata_as_osc.py` expects
(`heart`, `gsr`, `respiration`) — enough to exercise the full pipeline
end-to-end. The output W vectors won't be physiologically meaningful (no
real subject/emotion signal underlies them), but this is enough to confirm
the plumbing works before you have real data.

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

If you do have a real recording (raw `heart`/`gsr`/`respiration` CSV
columns, 100Hz), skip this step and use it directly with
`replay_biodata_as_osc.py` in §6 instead.

## 6. Test the live pipeline end-to-end

Open several terminals, all from the repo root.

**Terminal 1 — the core server:**

```bash
live_pipeline/run_live.sh \
    --regressor latent_pipeline/outputs/stage5_regressor_online/regressor.joblib \
    --transformer biodata_pipeline/models/transformer_ot_classconditional_online.pkl
```

Wait for `Waiting for /crocodile/session/start ...`.

**Terminal 2 — a lightweight W receiver (no StyleGAN2/torch needed):**

```bash
live_pipeline/run_debug_receiver.sh
```

This is the recommended first check — it just confirms 512-float W vectors
are actually arriving at the expected rate, without needing the private
StyleGAN2 model at all. Once you have that model installed (§4), you can
additionally run `live_pipeline/run_debug_viewer.sh` in another terminal for
an actual visual preview — same OSC stream, rendered.

**Terminal 3 — drive the session** (CLI shown; `run_control_panel.sh` works
the same way through buttons):

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
ends with `Session ended: test. Rows sent: N, skipped: 0`. Terminal 2 prints
periodic `received=... rate=.../s norm=...` lines while Terminal 3's second
`run_replay.sh` call is running, and `N` there should match Terminal 1's
`Rows sent` count. If both match and no `WARNING` lines appeared, the
pipeline is correctly installed and wired end-to-end.

## 7. Troubleshooting

- **`EADDRINUSE` / "address already in use"**: something is already bound to
  one of the ports involved (`9000` control+biodata, `9001` status, `1338`
  W-out, `8090` Open Stage Control's HTTP UI). Check for a stale process
  from a previous failed run (`ps aux | grep live_pipeline`) or pass
  different `--in-port`/`--out-port`/`--status-out-port` (and matching
  `--port`/`--send`/`--osc-port` to `session_control.py`/
  `run_control_panel.sh`) to run on alternate ports.
- **No W vectors received in Terminal 2**: confirm Terminal 1 actually
  reached the `LIVE` phase (`Live output started` in its log) — biodata
  arriving before `live/start` is intentionally discarded (see
  PIPELINE.md's state machine). Also check `replay_biodata_as_osc.py` and
  `live_pipeline.py` agree on `--port`/`--address`.
- **`ModuleNotFoundError`**: you're running a script with the wrong venv's
  interpreter directly instead of through its `run_*.sh` wrapper, or you
  installed before the requirements file included `neurokit2`/`joblib`/
  `python-osc` — re-run `pip install -r biodata_pipeline/requirements.txt`.
- Anything else: see PIPELINE.md's "Two real bugs were caught building
  this" section for the two subtlest failure modes found so far (a
  calibration/live speed artifact and a peak-detector priming bug), both
  already fixed but documented in case something similar resurfaces.
