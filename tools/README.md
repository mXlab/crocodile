# Tools

Utility scripts for data processing, testing, and visualization.

## Scripts

### `test_biosppy.py`

Test BioSPPy signal processing on a raw CSV file:

```bash
python tools/test_biosppy.py CSV_FILE [-s SAMPLING_RATE] [-m MIN_AMPLITUDE]
```

- `-s, --sampling-rate`: Sample rate in Hz (default: 1000)
- `-m, --min-amplitude`: Minimum EDA amplitude (default: 0.1)

Processes heart rate (BVP) and EDA channels using BioSPPy and prints the results.

### `convert_wav_to_csv.py`

Convert a WAV audio file to CSV format:

```bash
python tools/convert_wav_to_csv.py INPUT_WAV OUTPUT_CSV
```

### `generate_transition.py`

Generate smooth transition videos by interpolating through the GAN latent space:

```bash
python tools/generate_transition.py --model-path MODEL_PATH --output OUTPUT_VIDEO
```

Creates a video showing continuous morphing between generated frames.
