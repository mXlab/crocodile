# Crocodile

Crocodile is a multimodal emotion classification and generation system that combines:

- **GANs** for generating facial video frames conditioned on physiological (biodata) signals
- **Emotion classification** from physiological signals (heart rate, EDA/skin conductance, respiration)
- **Arduino-based biodata collection** using wearable sensors
- **Biodata processing pipeline** for feature extraction and emotion recognition

The system can generate synthetic facial videos conditioned on real physiological features associated with emotional states.

## Repository Structure

```
crocodile/
├── training_gan/          # GAN training scripts (unconditional, label-conditioned, biodata-conditioned)
├── lib/                   # Core Python library (datasets, models, signal processing, evaluation)
├── cnn_emotion_classifier/# CNN-based emotion classifier from physiological signals
├── biodata_pipeline/      # Modular emotion recognition pipeline (feature extraction, analysis)
├── BioDataFeatureExtract/ # Arduino/Teensy real-time biodata collection system
├── tools/                 # Utility scripts (signal testing, conversion, video generation)
├── scripts/               # SLURM batch job scheduling
├── notebooks/             # Exploratory Jupyter notebooks for biodata feature engineering
├── data/                  # Emotion-labeled physiological recordings (raw + CSV)
├── conda/                 # Legacy conda environment (PyTorch 1.5.0, Python 3.7)
└── requirements/          # Modern pip dependencies (PyTorch 2.1, Python 3.10+)
```

Each major subdirectory has its own README with detailed documentation:

- [training_gan/README.md](training_gan/README.md) -- GAN training usage and configuration
- [biodata_pipeline/README.md](biodata_pipeline/README.md) -- Emotion recognition pipeline status and documentation
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

### 1. Prepare a dataset from video

```bash
python dataset.py VIDEO_PATH DATASET_PATH [-r RESOLUTION]
```

### 2. Train a biodata-conditioned GAN

```bash
python training_gan/train_with_biodata.py OUTPUT_PATH -r 128 \
    --path-to-dataset DATASET_PATH \
    --path-to-biodata BIODATA_CSV
```

See [training_gan/README.md](training_gan/README.md) for all training options.

### 3. Train an emotion classifier

```bash
cd cnn_emotion_classifier
python train.py --path_to_dataset PATH --epochs 3 --batch_size_train 128
```

See [cnn_emotion_classifier/README.md](cnn_emotion_classifier/README.md) for details.

### 4. Run the biodata processing pipeline

See [biodata_pipeline/README.md](biodata_pipeline/README.md) for the modular pipeline (data slicing, feature extraction, analysis).

## Data Flow

1. **Acquisition**: Arduino/sensors collect heart rate, EDA, and respiration at 1000 Hz
2. **Preprocessing**: BioSPPy/NeuroKit2 extract features (BPM, HRV, SCR/SCL, breathing rate)
3. **Dataset**: Image frames + physiological features are aligned by timestamps
4. **Training**: Generator (latent + biodata) produces synthetic frames; Discriminator evaluates realism
5. **Evaluation**: FID score measures quality against real dataset

## Key Dependencies

- **Deep Learning**: PyTorch, TorchVision
- **Signal Processing**: BioSPPy, NeuroKit2, SciPy, PyWavelets
- **Data**: Pandas, NumPy, scikit-learn, imbalanced-learn
- **Visualization**: Matplotlib, TensorboardX
- **Video**: OpenCV, MoviePy

## License

See [LICENSE](LICENSE).
