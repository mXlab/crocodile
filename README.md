# Crocodile

Crocodile is an interactive installation: a participant's physiological signals
(heart rate, EDA, respiration) drive real-time generation of a face via a
StyleGAN2 model, trained on actress Laurence Dauphinais, that reacts to the
participant's emotional state. The project combines:

- **A W-space encoder pipeline** (`latent_pipeline/`) inverting Dauphinais' own
  video into a pretrained StyleGAN2's latent space, so a future biodata→W
  regressor can drive that frozen generator — the active approach
- **A biodata feature/alignment pipeline** (`biodata_pipeline/`) extracting
  physiological features and mapping a new participant's signal space onto
  Dauphinais' reference space
- **Emotion classification** from physiological signals (heart rate, EDA/skin
  conductance, respiration)
- **Arduino-based biodata collection** using wearable sensors
- *(legacy)* a from-scratch biodata-conditioned GAN (`training_gan/`),
  superseded by the approach above

See [PIPELINE.md](PIPELINE.md) for how these fit together and current status.

## Repository Structure

```
crocodile/
├── biodata_pipeline/      # Modular emotion recognition pipeline (feature extraction, analysis)
├── BioDataFeatureExtract/ # Arduino/Teensy real-time biodata collection system
├── cnn_emotion_classifier/ # CNN-based emotion classifier from physiological signals
├── conda/                 # Legacy conda environment (PyTorch 1.5.0, Python 3.7)
├── data/                  # Emotion-labeled physiological recordings (raw + CSV)
├── latent_pipeline/       # W-space encoder: invert video frames into StyleGAN2 latent space, attach biodata
├── lib/                   # Core Python library (datasets, models, signal processing, evaluation)
├── notebooks/             # Exploratory Jupyter notebooks for biodata feature engineering
├── requirements/          # Modern pip dependencies (PyTorch 2.1, Python 3.10+)
├── scripts/               # SLURM batch job scheduling
├── tools/                 # Utility scripts (signal testing, conversion, video generation)
└── training_gan/          # LEGACY — from-scratch conditional GAN training, superseded by latent_pipeline
```

Each major subdirectory has its own README with detailed documentation:

- [PIPELINE.md](PIPELINE.md) -- How biodata_pipeline and latent_pipeline fit together, current status
- [training_gan/README.md](training_gan/README.md) -- GAN training usage and configuration
- [biodata_pipeline/README.md](biodata_pipeline/README.md) -- Emotion recognition pipeline status and documentation
- [latent_pipeline/PLAN.md](latent_pipeline/PLAN.md) -- W-space encoder pipeline architecture and implementation plan
- [cnn_emotion_classifier/README.md](cnn_emotion_classifier/README.md) -- Emotion classifier training
- [BioDataFeatureExtract/README.md](BioDataFeatureExtract/README.md) -- Arduino sensor collection setup
- [lib/README.md](lib/README.md) -- Core library API overview
- [tools/README.md](tools/README.md) -- Utility scripts reference

## Environment Setup

**Option A -- Conda (legacy, PyTorch 1.5.0 / Python 3.7):**

```bash
conda env create -f conda/crocodile.yml
conda activate crocodile
```

**Option B -- pip (modern, PyTorch 2.1 / Python 3.10+):**

```bash
pip install -r requirements/biodata_features.txt
```

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
python latent_pipeline/scripts/train_synthetic.py --config latent_pipeline/configs/default.yaml
python latent_pipeline/scripts/train_frames.py --config latent_pipeline/configs/default.yaml
python latent_pipeline/scripts/stage3_validate.py --config latent_pipeline/configs/default.yaml
python latent_pipeline/scripts/stage4_assemble.py --config latent_pipeline/configs/default.yaml
```

See [latent_pipeline/PLAN.md](latent_pipeline/PLAN.md) for stage detail. Stage
5 (biodata→W regressor, needed for the runtime pipeline) is not yet built.

### 3. Train an emotion classifier

```bash
cd cnn_emotion_classifier
python train.py --path_to_dataset PATH --epochs 3 --batch_size_train 128
```

See [cnn_emotion_classifier/README.md](cnn_emotion_classifier/README.md) for details.

### 4. (Legacy) train a from-scratch biodata-conditioned GAN

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
