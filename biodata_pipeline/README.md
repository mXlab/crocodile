# Biodata Pipeline

Pipeline for extracting physiological features from raw sensor data (EDA, cardiac, respiration), classifying emotions, and aligning feature spaces across subjects.

## Requirements

Python 3.10+ with a virtual environment:

```bash
pip install -r requirements.txt
```

## Modules

| Module | Description |
|--------|-------------|
| `modules/continuous_feature_extractor.py` | `EnhancedContinuousFeatureExtractor` — extracts 67 features (15 EDA, 17 cardiac, 30 respiratory, 5 multimodal) sample-by-sample, maintaining filter state across the entire session. This is the main extractor. |
| `modules/data_slicer.py` | `DataSlicer` — slices raw physiological data into emotion-labeled windows (segment, sliding, or hybrid mode). Also provides `slice_features_into_windows()` for windowing pre-extracted feature DataFrames. |
| `modules/feature_extractor.py` | `EmotionFeatureExtractor` — per-segment feature extraction (resets filters at emotion boundaries). Used by older scripts. |
| `modules/feature_analyzer.py` | `FeatureAnalyzer` — statistical analysis of feature discriminability (ANOVA, feature importance ranking, visualizations). |
| `modules/pipeline_config.py` | `PipelineConfig` — YAML-based configuration with defaults and validation. |
| `modules/continuous_feature_extractor_basic.py` | `ContinuousFeatureExtractor` — earlier version with 15 features. Kept for comparison. |

## Input Data Format

CSV files at 100 Hz with columns: `heart`, `gsr`, `respiration`, `emotion`, `feeling_it`.

## Workflows

### 1. Extract continuous features

Extracts features from raw sensor CSVs. Produces one row per time interval (default 1s) with 67+ features plus metadata columns (`timestamp`, `sample_idx`, `emotion`, `feeling_it`, `session_id`).

```bash
# Single file
python scripts/extract_continuous_features.py \
    --input "emotion_biodata_subject_session.csv" \
    --output subject_session_features.csv

# Multiple files combined into one output
python scripts/extract_continuous_features.py \
    --input "emotion_biodata_laurence_main_*.csv" \
    --output laurence_main_features.csv

# Separate output per session
python scripts/extract_continuous_features.py \
    --input "emotion_biodata_laurence_main_*.csv" \
    --per-session
```

Output is saved to `data/processed/`. Raw CSVs are read from `data/raw/` by default (override with `--data-dir`).

### 2. Evaluate classification with cross-validation

Slice the continuous features into time windows, then run GroupKFold cross-validation with a Random Forest classifier.

```bash
# Use super-segments for CV grouping (prevents leakage from overlapping windows)
python scripts/slice_and_evaluate.py \
    --features data/processed/laurence_main_features.csv \
    --window-size 20 --stride 2 \
    --super-segment-size 45

# Select emotions, window size, stride
python scripts/slice_and_evaluate.py \
    --features data/processed/laurence_main_features.csv \
    --include neu sad anx \
    --window-size 20 --stride 2 \
    --super-segment-size 45

# Exclude specific emotions instead
python scripts/slice_and_evaluate.py \
    --features data/processed/laurence_main_features.csv \
    --exclude nul \
    --window-size 20 --stride 2 \
    --super-segment-size 45

# Create dataset only, evaluate separately
python scripts/slice_and_evaluate.py \
    --features data/processed/laurence_main_features.csv \
    --include fea sad \
    --window-size 30 --stride 5  \
    --super-segment-size 45 \
    --no-evaluate
```

This produces windowed feature CSVs in `data/processed/` and reports in `reports/cross_validation/` (per-fold metrics, summary, confusion matrix).

You can also evaluate a pre-built windowed CSV directly:

```bash
# GroupKFold CV (groups by parent_segment_id or session_id)
python scripts/evaluate_with_cv.py \
    --features data/processed/windowed_neu_sad_anx_w30s5.csv

# Standard KFold CV (for independent/non-overlapping windows)
python scripts/train_and_evaluate.py \
    --features data/processed/windowed_neu_sad_anx_w30s5.csv
```

### 3. Cross-subject alignment

When a new subject records calibration data with shared emotions, their feature space can be aligned to the reference subject's (the actress') space so the GAN sees physiologically plausible conditioning vectors.

The goal is not emotion classification but a good geometric mapping: transformed subject features should land close to the correct reference emotion prototypes, in the same region of feature space the GAN was trained on.

**Step 1: Extract features for both subjects** (see workflow 1).

**Step 2: Train an alignment transformer.** Three methods are available:

| Method | `--method` | Description |
|--------|-----------|-------------|
| Ridge regression | `ridge` | Fits a linear map on per-emotion prototype pairs (means only). Fast, minimal data required. |
| Global linear OT | `ot_global` | Gaussian Monge map fitted on all samples from both domains, ignoring emotion labels. Aligns full distributions but conflates emotion classes. |
| Class-conditional OT | `ot_classconditional` | Separate Gaussian Monge map per emotion, fitted on matching emotion samples. Best alignment quality in practice. |

```bash
# Class-conditional OT (recommended)
python scripts/train_transformer.py \
    --reference data/processed/laurence_main_features.csv \
    --subject data/processed/subject_session_features.csv \
    --method ot_classconditional \
    --output models/transformer_ot_classconditional.pkl

# Ridge (fast baseline)
python scripts/train_transformer.py \
    --reference data/processed/laurence_main_features.csv \
    --subject data/processed/subject_session_features.csv \
    --method ridge \
    --output models/transformer_ridge.pkl
```

Options:
- `--method` — `ridge` (default), `ot_global`, or `ot_classconditional`
- `--alpha 10.0` — Ridge regularization strength (Ridge only)
- `--reg 1e-5` — OT regularization strength (OT methods only)
- `--n-features 20` — keep only top N features by ANOVA F-test (Ridge only)
- `--output models/custom_transformer.pkl` — output path

All three methods share the same `transform()` interface and are saved as `.pkl` files that load automatically with the correct class.

At inference, class-conditional OT assigns each sample to its nearest subject emotion prototype, then applies that emotion's specific map — no emotion label is required at runtime.

**Step 3: Validate.** Four metrics are reported:

| Metric | What it measures |
|--------|-----------------|
| Separation ratio | `dist(correct proto) / mean dist(wrong protos)` — below 1.0 is good |
| Normalized Prototype RMSE | Prototype alignment error as a fraction of inter-class spread — 0 is perfect, 1 is useless |
| Nearest-Prototype Accuracy (NPA) | Fraction of transformed windows nearest to the correct reference prototype — method-agnostic, random baseline = 1/n\_emotions |
| RF accuracy | Random Forest trained on reference, tested on transformed subject — useful but affected by between-session physiological drift |

NPA and RMSE\_norm are the recommended metrics for comparing alignment methods as they are method-agnostic and directly measure geometric quality.

```bash
python scripts/validate_transformer.py \
    --reference data/processed/laurence_main_features.csv \
    --subject data/processed/subject_session_features.csv \
    --model models/transformer_ot_classconditional.pkl
```

Options:
- `--model` — path to any trained transformer `.pkl` (method detected automatically)
- `--output-dir reports/` — where to save results (default)

Outputs:
- `reports/validation_results.json` — all metrics including per-emotion NPA and prototype errors
- `reports/transformation_validation.png` — PCA visualization of prototype alignment

**Benchmark on Erin → Actress (anx, neu, sad):**

| Method | RMSE\_norm | NPA | RF accuracy |
|--------|-----------|-----|-------------|
| Ridge | 0.397 | 52.8% | 59.3% |
| OT global | 0.455 | 37.4% | 30.2% |
| OT class-conditional | **0.000** | **54.2%** | **77.0%** |

Class-conditional OT wins on all metrics. OT global underperforms Ridge because a single map conflates emotion-specific structure.

## Directory Structure

```
biodata_pipeline/
├── modules/                  # Core library
├── scripts/                  # CLI entry points
├── configs/                  # YAML presets for slice_and_evaluate
├── data/
│   ├── raw/                  # Raw sensor CSVs
│   └── processed/            # Extracted feature CSVs
├── models/                   # Trained transformer .pkl files
└── reports/                  # Validation results, CV reports, plots
```
