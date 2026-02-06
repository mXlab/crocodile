# CNN Emotion Classifier

Classifies emotions from physiological signals (heart rate, EDA, respiration) using a 1D ResNet architecture.

## Usage

```bash
cd cnn_emotion_classifier
python train.py --path_to_dataset PATH --epochs 3 --batch_size_train 128 --optim adam
```

## Arguments

| Argument | Description | Default |
|---|---|---|
| `--epochs` | Number of training epochs | 3 |
| `--batch_size_train` | Training batch size | 128 |
| `--batch_size_test` | Test batch size | 1000 |
| `--learning_rate` | Learning rate | 1e-3 |
| `--momentum` | SGD momentum | 0.5 |
| `--optim` | Optimizer (`adam`, `sgd`, `sls`) | `adam` |
| `--path_to_dataset` | Path to dataset directory | None |
| `--output_path` | Path to save results | None |
| `--downsampling` | Downsampling factor | 1 |
| `--overlap` | Window overlap fraction | 0.0 |
| `--log_interval` | Logging frequency (batches) | 10 |

## Architecture

**ECGResNet** (`model.py`): A 1D ResNet with 16 residual blocks, adaptive average pooling, and a fully connected output layer. Designed for classifying time-series physiological signals.

The model uses `ResBlock` layers with 1D convolutions, batch normalization, and skip connections.

## Dataset

**EmotionDataset** (`dataset.py`): Loads labeled physiological recordings from CSV files. Each sample is a window of sensor readings with an associated emotion label. Preprocessing applies envelope filtering and normalization via [lib/biodata.py](../lib/biodata.py).

Training data (`sensor_data.csv`, `timestamps.csv`) maps physiological signal windows to emotion categories.

## Dependencies

Requires [imbalanced-learn](https://imbalanced-learn.org/) for handling class imbalance during training. Install via `pip install -r requirements/biodata_features.txt` from the repository root.
