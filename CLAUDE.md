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
| `latent_pipeline/` | W-space encoder: synthetic pre-training + real frame fine-tuning | `scripts/train_synthetic.py`, `scripts/train_frames.py` |

Each subdirectory has its own README.md with detailed documentation.

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
