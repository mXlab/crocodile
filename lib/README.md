# Core Library (`lib/`)

Shared Python modules used by the GAN training scripts, the emotion classifier, and the biodata pipeline.

## Modules

### `dataset.py`

PyTorch Dataset classes for loading data:

- **`CrocodileDataset`** -- Loads image frames aligned with biodata features by timestamp. Used by the GAN training scripts.
- **`EmotionDataset`** -- Loads physiological signal windows with emotion labels. Used by the emotion classifier.
- **`SequenceSampler`** -- Custom sampler for sequential batches (used in biodata-conditioned GAN).
- **`extract_video()`** / **`resize_images()`** -- Video frame extraction and resizing utilities.

### `biodata.py`

Signal processing utilities for physiological data:

- `enveloppe_filter()` -- Envelope filtering for raw signals
- `interpolate()`, `rate_of_change()`, `compute_intervals()` -- Signal analysis
- Heart rate detection classes ported from the Arduino [BioData library](../BioDataFeatureExtract/libraries/BioData/): `MinMax`, `Threshold`, `Lop`, `Heart`

### `models/`

Neural network architectures:

- `generator.py` -- Base `Generator` class with sampling method
- `discriminator.py` -- Base `Discriminator` class with gradient penalty
- `small_cnn.py` -- `SmallGenerator`, `SmallDiscriminator`, `ConditionalSmallGenerator`, `ConditionalSmallDiscriminator`
- `deep_cnn.py` -- `ECGResNet` (ResBlock-based 1D classifier)

### `fid/`

Frechet Inception Distance for evaluating GAN output quality:

- `fid.py` -- FID metric computation
- `inception.py` -- InceptionV3 feature extractor

### `utils.py`

Loss functions: NSGAN (non-saturating), WGAN variants.

### `logger.py`

Experiment logging with TensorboardX integration.
