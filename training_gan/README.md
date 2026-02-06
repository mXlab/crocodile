# GAN Training

This directory contains training scripts for Generative Adversarial Networks that generate facial video frames. Three modes are available, from simplest to most advanced.

## Dataset Preparation

Before training, extract frames from video:

```bash
python dataset.py VIDEO_PATH DATASET_PATH [-r RESOLUTION]
```

- `VIDEO_PATH`: Path to the source video.
- `DATASET_PATH`: Where to save extracted frames.
- `-r RESOLUTION` (optional): Resize frames to this resolution.

## Training Modes

### Unconditional GAN

Generates faces without any conditioning signal.

```bash
python training_gan/train.py -r 128 --output-path OUTPUT_PATH --path-to-dataset DATASET_PATH
```

### Label-Conditioned GAN

Conditions generation on discrete emotion labels.

```bash
python training_gan/train_conditional.py OUTPUT_PATH -r 128 --path-to-dataset DATASET_PATH
```

### Biodata-Conditioned GAN (main research focus)

Conditions generation on continuous physiological signals (heart rate, EDA).

```bash
python training_gan/train_with_biodata.py OUTPUT_PATH -r 128 \
    --path-to-dataset DATASET_PATH \
    --path-to-biodata BIODATA_CSV
```

If `--path-to-biodata` is omitted, it defaults to `LaurenceHBS-Nov919mins1000Hz-Heart+GSR-2channels.csv` inside `DATASET_PATH`.

## Common Arguments

| Argument | Description | Default |
|---|---|---|
| `-r, --resolution` | Image resolution | 128 |
| `-e, --num-epochs` | Number of training epochs | 1000 |
| `-bs, --batch-size` | Batch size | 64 |
| `-z, --num-latent` | Latent vector dimensions | 50 (unconditional), 5 (conditional/biodata) |
| `-nl, --num-layers` | Number of convolutional layers | 6 (unconditional), 4 (conditional/biodata) |
| `-f, --num-filters` | Number of filters | 128 (unconditional), 256 (conditional/biodata) |
| `-lrd` | Discriminator learning rate | 1e-3 (unconditional), 5e-3 (conditional/biodata) |
| `-lrg` | Generator learning rate | 1e-3 (unconditional), 2e-3 (conditional/biodata) |
| `-gp, --gradient-penalty` | Gradient penalty weight (0 = off) | 0 |
| `--spectral-norm-gen` | Enable spectral normalization for generator | off |
| `--seed` | Random seed | 1234 |
| `--eval-freq` | Evaluation frequency (epochs) | 1 |

### Biodata-specific arguments

| Argument | Description | Default |
|---|---|---|
| `--path-to-biodata` | Path to biodata CSV file | auto-detect in dataset dir |
| `--normalization` | Feature normalization (`normalized` or `standardized`) | `normalized` |
| `--length-sequence` | Sequence length for sequential sampling | 150 |
| `--num-sequences` | Number of sequences to sample | 10 |
| `--num-variations` | Number of latent variations per sequence | 10 |

Run `python training_gan/<script>.py --help` for the full list of options.

## Signal Processing Defaults

For the biodata-conditioned GAN, physiological signals are processed with:

- Sampling rate: 1000 Hz
- Video FPS: 30000/1001 (~29.97)
- Heart peak detection: distance=400, width=100, prominence=0.01
- EDA peak detection: distance=1800, width=600, prominence=0.0014

## Output

Each training run creates a timestamped subfolder inside the output path containing:

- `img/` -- Sample images generated at each evaluation epoch
- Tensorboard logs (for conditional and biodata modes)
- Training configuration (JSON)

## SLURM Batch Training

For large-scale hyperparameter sweeps on a cluster:

```bash
python scripts/launch_batch.py
```

This uses [submitit](https://github.com/facebookincubator/submitit) to schedule 25 configurations x 100 seeds with random search over filters, learning rates, latent dimensions, and layers. See [scripts/launch_batch.py](../scripts/launch_batch.py) for configuration.

## Architecture

The GAN models are defined in [lib/models/](../lib/models/):

- `SmallGenerator` / `SmallDiscriminator` -- Base unconditional GAN
- `ConditionalSmallGenerator` / `ConditionalSmallDiscriminator` -- Conditioned variants
- Loss functions: NSGAN, WGAN variants (see [lib/utils.py](../lib/utils.py))
- Evaluation: FID score (see [lib/fid/](../lib/fid/))
