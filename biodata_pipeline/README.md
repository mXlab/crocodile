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

### 3. Cross-subject alignment and classification

When a new subject records calibration data with shared emotions, you can align their feature space to the reference subject and reuse the reference subject's classifier.

**Step 1: Extract features for both subjects** (see workflow 1).

**Step 2: Train a prototype alignment transformer.** Computes per-emotion mean feature vectors (prototypes) for both subjects, then learns a Ridge regression mapping from the new subject's prototypes to the reference subject's prototypes. Features are standardized before fitting.

```bash
python scripts/train_transformer.py \
    --reference data/processed/laurence_main_features.csv \
    --subject data/processed/subject_session_features.csv
```

Options:
- `--alpha 10.0` — Ridge regularization strength (default)
- `--n-features 20` — optional: keep only top N features ranked by ANOVA F-test
- `--output models/custom_transformer.pkl` — output path (default: `models/subject_alignment_transformer.pkl`)

**Step 3: Validate.** Runs two tests: (1) separation ratio — transformed samples should land closer to the correct emotion prototype than to wrong ones (ratio < 1.0 is good), (2) RF classifier — trains a Random Forest on the reference data, classifies the transformed subject data, and compares accuracy with and without the transformation.

```bash
python scripts/validate_transformer.py \
    --reference data/processed/laurence_main_features.csv \
    --subject data/processed/subject_session_features.csv
```

Options:
- `--model models/subject_alignment_transformer.pkl` — path to trained transformer (default)
- `--output-dir reports/` — where to save results (default)

Outputs:
- `reports/validation_results.json` — per-emotion separation ratios and RF classification accuracy
- `reports/transformation_validation.png` — PCA visualization of prototype alignment

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
