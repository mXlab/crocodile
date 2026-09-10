# Crocodile

Crocodile is an interactive installation: a user's physiological signals
(heart rate, EDA, respiration) drive real-time generation of a face via a
StyleGAN2 model, trained on a method actress, that reacts to the
user's emotional state. The project combines:

- **A W-space encoder pipeline** (`latent_pipeline/`) inverting the actress' own
  video into a pretrained StyleGAN2's latent space, so a future biodata→W
  regressor can drive that frozen generator — the active approach
- **A biodata feature/alignment pipeline** (`biodata_pipeline/`) extracting
  physiological features and mapping a new user's signal space onto
  the actress' reference space
- **Emotion classification** from physiological signals (heart rate, EDA/skin
  conductance, respiration)
- **Arduino-based biodata collection** using wearable sensors
- *(legacy)* a from-scratch biodata-conditioned GAN (`training_gan/`),
  superseded by the approach above

See [PIPELINE.md](PIPELINE.md) for how these fit together and current status.

## Repository Structure

```
crocodile/
├── biodata_pipeline/       # Modular emotion recognition pipeline (feature extraction, analysis)
├── BioDataFeatureExtract/  # Arduino/Teensy real-time biodata collection system
├── cnn_emotion_classifier/ # CNN-based emotion classifier from physiological signals
├── conda/                  # Legacy conda environment (PyTorch 1.5.0, Python 3.7)
├── data/                   # Emotion-labeled physiological recordings (raw + CSV)
├── latent_pipeline/        # W-space encoder: invert video frames into StyleGAN2 latent space, attach biodata
├── live_pipeline/          # Runtime layer: biodata (OSC) -> W (OSC) -> Autolume
├── lib/                    # Core Python library (datasets, models, signal processing, evaluation)
├── notebooks/              # Exploratory Jupyter notebooks for biodata feature engineering
├── requirements/           # Modern pip dependencies (PyTorch 2.1, Python 3.10+)
├── scripts/                # SLURM batch job scheduling
├── tools/                  # Utility scripts (signal testing, conversion, video generation)
└── training_gan/           # LEGACY — from-scratch conditional GAN training, superseded by latent_pipeline
```

Each major subdirectory has its own README with detailed documentation:

- [PIPELINE.md](PIPELINE.md) -- How biodata_pipeline and latent_pipeline fit together, current status
- [INSTALL.md](INSTALL.md) -- Step-by-step install and test instructions for the live pipeline
- [biodata_pipeline/README.md](biodata_pipeline/README.md) -- Emotion recognition pipeline status and documentation
- [latent_pipeline/PLAN.md](latent_pipeline/PLAN.md) -- W-space encoder pipeline architecture and implementation plan
- [cnn_emotion_classifier/README.md](cnn_emotion_classifier/README.md) -- Emotion classifier training
- [lib/README.md](lib/README.md) -- Core library API overview
- [tools/README.md](tools/README.md) -- Utility scripts reference
- [BioDataFeatureExtract/README.md](BioDataFeatureExtract/README.md) -- Arduino sensor collection setup
- [training_gan/README.md](training_gan/README.md) -- GAN training usage and configuration

## Environment Setup

```bash
python3.10 -m venv crocodile-venv
source crocodile-venv/bin/activate
pip install -r requirements/biodata_features.txt
```

Note: `requirements/biodata_features.txt` pins exact versions (pandas 2.1.4, numpy
1.26.2, torch 2.1.2, ...) that only have prebuilt wheels for Python 3.10 --
installing on a newer default `python3` (3.12+) will fail trying to build
old pandas/numpy from source.

## Quick Start

The active pipeline has two halves that meet at `latent_pipeline` Stage 4. See
[PIPELINE.md](PIPELINE.md) for the full picture, current build status, and
what's next.

### 1. Extract biodata features

```bash
python biodata_pipeline/scripts/extract_continuous_features.py \
    --input "emotion_biodata_*.csv" --output subject_features.csv
```

See [biodata_pipeline/README.md](biodata_pipeline/README.md) for feature
extraction, windowing/evaluation, and the cross-subject alignment transformer.

### 2. Invert video into StyleGAN2's W-space and attach biodata

```bash
python latent_pipeline/scripts/stage1_extract.py --config latent_pipeline/configs/default.yaml

# Stage 2A: synthetic pre-training (warm-start, optional but recommended)
python latent_pipeline/scripts/stage2a_train_synthetic.py --config latent_pipeline/configs/default.yaml

# Stage 2B: fine-tune on real frames — the actual encoder training.
# --pretrained loads 2A's weights; without it, 2B starts from random init.
python latent_pipeline/scripts/stage2b_train_frames.py --config latent_pipeline/configs/default.yaml \
    --pretrained latent_pipeline/outputs/train_synthetic/best.pt

python latent_pipeline/scripts/stage3_validate.py --config latent_pipeline/configs/default.yaml
python latent_pipeline/scripts/stage4_assemble.py --config latent_pipeline/configs/default.yaml
```

See [latent_pipeline/PLAN.md](latent_pipeline/PLAN.md) for stage detail,
including Stage 5 (biodata→W regressor, `stage5_train_regressor.py`).

### 3. Run the live pipeline (biodata → W → Autolume)

Once a regressor (Stage 5) and an alignment transformer exist, `live_pipeline/`
runs the runtime layer as a persistent OSC session server. Everything below
runs from the repo root; `live_pipeline/` has no venv of its own — the wrapper
scripts pin the right interpreter (`biodata_pipeline/venv` or
`latent_pipeline/.venv`) for you. **See [INSTALL.md](INSTALL.md) for full,
step-by-step setup** (including what's private and needs to come from a
teammate rather than this repo, and how to generate synthetic test data if
you don't have a real recording) — this is just the command summary.

```bash
# 1. Start the core server (biodata_pipeline/venv) -- loads the regressor +
#    transformer once, then handles sessions via an OSC state machine
#    (IDLE -> READY -> CALIBRATING -> CALIBRATED -> LIVE).
live_pipeline/run_live.sh \
    --regressor latent_pipeline/outputs/stage5_regressor_online/regressor.joblib \
    --transformer biodata_pipeline/models/transformer_ot_classconditional_online.pkl \
    --record-dir recordings/            # optional: save sessions to disk
    # --calibration-csv path/to/calib.csv   # optional: prime a session at start

# 2. Drive the session state machine -- either the GUI panel or the CLI.
live_pipeline/run_control_panel.sh          # Open Stage Control GUI, port 8090
# or, one message per call:
SC=live_pipeline/run_session_control.sh
$SC --start-session [ID]
$SC --start-calibration
$SC --stop-calibration
$SC --start-live
$SC --recalibrate   # LIVE only
$SC --end-session

# 3. Feed it biodata -- real sensor hardware (protocol not yet finalized), a
#    real recording replayed at real-time speed, or -- since real biodata
#    can't be shared on GitHub -- synthetic data generated with NeuroKit2:
live_pipeline/run_generate_synthetic.sh --duration 60 --seed 1 \
    --output live_pipeline/data/synthetic_live.csv
live_pipeline/run_replay.sh --input live_pipeline/data/synthetic_live.csv --speed 1.0

# 4. Start the latent controller (it composites the visitor's vector with the
#    operator's emotion selection and is what actually feeds Autolume now).
#    Already running if you used the GUI panel in step 2, since
#    run_control_panel.sh loads it by default; otherwise (CLI session
#    control) start it explicitly:
live_pipeline/run_control_panel.sh
#    Then point Autolume at the controller's OSC output (port 1338, address
#    /crocodile/latent/final; uncheck Autolume's "project" box, since the
#    output is already W-space); or sanity-check the OSC plumbing alone, no
#    StyleGAN2 model needed:
live_pipeline/run_debug_receiver.sh
#    or, for an actual visual preview (needs latent_pipeline/.venv AND the
#    private StyleGAN2 checkpoint -- see INSTALL.md):
live_pipeline/run_debug_viewer.sh
```

See [PIPELINE.md](PIPELINE.md#live-pipeline-biodata-osc--w-osc--autolume) for
the full state machine, OSC address table, recording format, and
recalibration design.

### 4. Train an emotion classifier

```bash
cd cnn_emotion_classifier
python train.py --path_to_dataset PATH --epochs 3 --batch_size_train 128
```

See [cnn_emotion_classifier/README.md](cnn_emotion_classifier/README.md) for details.

### 5. (Legacy) train a from-scratch biodata-conditioned GAN

Superseded by the pipeline above — kept for reference, not on the active path.

```bash
python lib/dataset.py VIDEO_PATH DATASET_PATH [-r RESOLUTION]
python training_gan/train_with_biodata.py OUTPUT_PATH -r 128 \
    --path-to-dataset DATASET_PATH \
    --path-to-biodata BIODATA_CSV
```

See [training_gan/README.md](training_gan/README.md) for all training options.

See [PIPELINE.md](PIPELINE.md) and [pipeline_diagram.md](pipeline_diagram.md)
for the full architecture, diagrams, and current per-stage status.

## Key Dependencies

- **Deep Learning**: PyTorch, TorchVision
- **Signal Processing**: BioSPPy, NeuroKit2, SciPy, PyWavelets
- **Data**: Pandas, NumPy, scikit-learn, imbalanced-learn
- **Visualization**: Matplotlib, TensorboardX
- **Video**: OpenCV, MoviePy

## License

See [LICENSE](LICENSE).
