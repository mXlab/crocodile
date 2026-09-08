# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Crocodile is an interactive art project that uses physiological markers of emotion and generative AI (GANs). Technically, it is a multimodal emotion classification and generation system. It generates synthetic facial videos conditioned on physiological signals (heart rate, EDA, respiration) and classifies emotions from those same signals. See the [README.md](README.md) for a general introduction.

See [PIPELINE.md](PIPELINE.md) for how `biodata_pipeline` and `latent_pipeline` fit together as the active preprocessing pipeline, current build status, and terminology notes.

## Repository Map

| Directory | Purpose | Key entry point |
|---|---|---|
| `training_gan/` | **Legacy** — from-scratch conditional GAN (unconditional, label-conditioned, biodata-conditioned), predates the StyleGAN2/`latent_pipeline` approach; superseded, not on the active critical path | `train_with_biodata.py` |
| `lib/` | Shared library: datasets, models, signal processing, FID, losses | `dataset.py`, `biodata.py`, `models/` |
| `cnn_emotion_classifier/` | 1D ResNet emotion classifier from physiological signals | `train.py` |
| `biodata_pipeline/` | Modular pipeline: data slicing, feature extraction, analysis | `modules/data_slicer.py`, `modules/feature_extractor.py` |
| `BioDataFeatureExtract/` | Arduino/Teensy real-time sensor collection (git submodules) | `src/feature_extract.ino` |
| `tools/` | Utilities: signal testing, WAV conversion, GAN transitions | `test_biosppy.py` |
| `scripts/` | SLURM batch job launcher | `launch_batch.py` |
| `notebooks/` | Exploratory Jupyter notebooks (v3 is latest, uses NeuroKit2) | `biodata_features-v3.ipynb` |
| `data/` | Emotion-labeled physiological recordings (ANG, ARO, FEA, HAP) | `csv/`, `raw/`, `timestamps.csv` |
| `latent_pipeline/` | W-space encoder: synthetic pre-training + real frame fine-tuning | `scripts/stage2a_train_synthetic.py`, `scripts/stage2b_train_frames.py` |
| `live_pipeline/` | Runtime layer: biodata (OSC) → W (OSC) → Autolume; imports `biodata_pipeline`/`latent_pipeline` modules as libraries, runs under their venvs (see wrapper scripts) | `live_pipeline.py`, `run_live.sh` |

Each subdirectory has its own README.md with detailed documentation.

## Privacy: Biodata and Trained Models Are NEVER Committed

Raw biodata (any CSV/recording with real heart/EDA/respiration signal from
a real person — the actress or a visitor), the actress' feature/reference
datasets, trained models derived from them (regressors, alignment
transformers), and the StyleGAN2 checkpoint are all **private** and must
never be committed to this repo, regardless of format or how small/
innocuous a file looks (a timestamps-only CSV with no raw signal is still
a real recording session and still private).

- `.gitignore` already blanket-excludes `*.csv`, `*.pkl`, `*.joblib`,
  `models/`, `biodata_pipeline/models/`, `biodata_pipeline/data/{raw,
  processed,features,metadata}/`, and `latent_pipeline/outputs/` — trust
  it, don't work around it (no `git add -f` on these paths).
- **Before any `git add`/commit touching a new data or model file**,
  double check it isn't a real recording or a model trained on one — this
  has slipped through before: `cnn_emotion_classifier/sensor_data.csv`
  and two `timestamps.csv` files were tracked from before the `*.csv`
  rule existed, and `.gitignore` only blocks *new* files, not already-
  tracked ones. If you ever find a tracked file like this, `git rm
  --cached` it (keep the local file, just untrack it) and flag to the
  user that it's already in git history / possibly already pushed — that
  needs a separate, explicit decision (history rewrite + force-push),
  never done unprompted.
- For testing/development without real data, use synthetic data
  (`live_pipeline/generate_synthetic_biodata.py`, NeuroKit2-based) instead
  of real recordings. See [INSTALL.md](INSTALL.md)'s "What's public vs.
  private" table for exactly which assets are private and where to get
  them (ask a teammate — never commit a copy into the repo to "fix" a
  missing-file error).

## Environment

Use the modern pip environment (Python 3.10 exactly -- `requirements/biodata_features.txt`
pins versions with no prebuilt wheels for 3.12+):

```bash
python3.10 -m venv crocodile-venv
source crocodile-venv/bin/activate
pip install -r requirements/biodata_features.txt
```

A legacy conda environment exists (`conda/crocodile.yml`, Python 3.7, PyTorch 1.5.0) but is not recommended for new work.

## Common Commands

```bash
# Train biodata-conditioned GAN (legacy, superseded by latent_pipeline — see PIPELINE.md)
python training_gan/train_with_biodata.py OUTPUT_PATH -r 128 --path-to-dataset DATASET_PATH --path-to-biodata BIODATA_CSV

# Train emotion classifier
cd cnn_emotion_classifier && python train.py --path_to_dataset PATH --epochs 3 --batch_size_train 128 --optim adam

# Test signal processing
python tools/test_biosppy.py CSV_FILE -s 1000
```

## Architecture Notes

### Model classes

- GAN models are in `lib/models/`: `SmallGenerator`, `SmallDiscriminator`, and their `Conditional` variants in `small_cnn.py`
- `ECGResNet` in `lib/models/deep_cnn.py` (also duplicated in `cnn_emotion_classifier/model.py`) is a 1D ResNet for physiological signal classification
- Base classes `Generator` and `Discriminator` in `lib/models/` provide shared methods (sampling, gradient penalty)
- Loss functions (NSGAN, WGAN) are in `lib/utils.py`

### Dataset classes

- `CrocodileDataset` (`lib/dataset.py`): loads image frames + biodata features aligned by timestamp — used by GAN training
- `EmotionDataset` (`lib/dataset.py`): loads physiological windows + emotion labels — used by classifier
- `EmotionDataset_v2` in `cnn_emotion_classifier/dataset.py` is a separate variant used by that module

### Signal processing

- `lib/biodata.py`: envelope filtering, heart rate detection classes (`MinMax`, `Threshold`, `Lop`, `Heart`) ported from the Arduino BioData library
- `biodata_pipeline/modules/feature_extractor.py`: extracts multiple features from raw EDA, PPQ and respiration using NeuroKit2/scipy

## Important Gotchas

- **Two sampling rates**: GAN pipeline and classifier use **1000 Hz**. The biodata_pipeline uses **100 Hz**. Do not mix them.
- **Biodata default file**: if `--path-to-biodata` is omitted in `train_with_biodata.py`, it looks for a hardcoded CSV filename in the dataset directory. Pass the flag explicitly to avoid this.
- **Emotion labels**: abbreviated in data files (e.g. `war`, `nul`). The `data/` directory uses longer prefixes (ANG, ARO, FEA, HAP).
- **feeling_it column**: binary (0/1) pedal press by the actress during recording — used for quality filtering in `biodata_pipeline`.
- **Git submodules**: `BioDataFeatureExtract/libraries/` contains Arduino library submodules. Run `git submodule update --init --recursive` after cloning.
- **Path conventions**: training scripts use `sys.path.insert(0, '..')` to import from `lib/`. Always run them from the repo root or their own directory.

## Biodata Signal Defaults

Default processing parameters (in `training_gan/train_with_biodata.py`):

| Parameter | Value |
|---|---|
| Sampling rate | 1000 Hz |
| Video FPS | 30000/1001 (~29.97) |
| Heart peak detection | distance=400, width=100, prominence=0.01 |
| EDA peak detection | distance=1800, width=600, prominence=0.0014 |
