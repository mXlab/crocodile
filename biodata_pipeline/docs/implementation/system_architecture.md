# Crocodile Emotion Classification Pipeline - Repository Architecture

## Complete Directory Structure

```
biodata_pipeline/
│
├── README.md                          # Project overview and quick start
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
├── .gitignore                         # Git ignore file
│
├── modules/                           # Core pipeline modules
│   ├── __init__.py
│   ├── data_loader.py                 # Module 1: Load and validate CSV files
│   ├── data_slicer.py                 # Module 2: Slice and filter data ✅ DONE
│   ├── feature_extractor.py           # Module 3: Extract 54 features ✅ DONE
│   ├── feature_analyzer.py            # Module 4: Analyze feature discriminability
│   ├── classifier.py                  # Module 5: Train emotion classifier
│   └── realtime_classifier.py         # Module 6: Real-time deployment
│
├── utils/                             # Helper utilities
│   ├── __init__.py
│   ├── signal_processing.py           # Signal filtering, event detection
│   ├── visualization.py               # Plotting functions
│   ├── osc_communication.py           # OSC/UDP for Ossia Score
│   └── io_helpers.py                  # File I/O utilities
│
├── scripts/                           # Executable workflow scripts
│   ├── 01_explore_data.py             # Explore raw data, summarize sessions
│   ├── 02_slice_data.py               # Apply Module 2 filtering
│   ├── 03_extract_features.py         # Apply Module 3 feature extraction
│   ├── 04_analyze_features.py         # Apply Module 4 analysis
│   ├── 05_train_classifier.py         # Train and validate classifier
│   ├── 06_run_realtime.py             # Real-time classification
│   └── 00_process_all.py              # End-to-end pipeline
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb      # Interactive data exploration
│   ├── 02_feature_validation.ipynb    # Feature discriminability analysis
│   ├── 03_classifier_tuning.ipynb     # Hyperparameter tuning
│   └── 04_results_visualization.ipynb # Final results and plots
│
├── configs/                           # Configuration files
│   ├── emotions.yaml                  # Emotion label mappings
│   ├── pipeline_config.yaml           # Pipeline parameters
│   ├── feature_config.yaml            # Feature extraction settings
│   └── classifier_config.yaml         # Classifier hyperparameters
│
├── data/                              # Data directory (gitignored)
│   ├── raw/                           # Original CSV files from recording sessions
│   │   ├── session_001.csv
│   │   ├── session_002.csv
│   │   └── ...
│   │
│   ├── processed/                     # Sliced and filtered segments
│   │   ├── segments_all.pkl           # All segments from all sessions
│   │   ├── segments_filtered.pkl      # After emotion + feeling_it filtering
│   │   └── windows_30s.pkl            # Fixed-size training windows
│   │
│   ├── features/                      # Extracted features
│   │   ├── features_all.csv           # All features from all windows
│   │   ├── features_normalized.csv    # Normalized per-session
│   │   └── feature_importance.csv     # Ranked features
│   │
│   └── metadata/                      # Session metadata
│       ├── session_info.csv           # Recording dates, participants, etc.
│       └── emotion_mappings.json      # Emotion abbreviation → full name
│
├── models/                            # Trained models (gitignored)
│   ├── trained_classifiers/
│   │   ├── template_matcher.pkl       # Template matching classifier
│   │   ├── svm_classifier.pkl         # SVM classifier
│   │   ├── rf_classifier.pkl          # Random Forest classifier
│   │   └── ensemble_classifier.pkl    # Ensemble model
│   │
│   └── baselines/                     # Personal baseline data per participant
│       ├── baseline_actress_LD.pkl
│       └── baseline_participant_001.pkl
│
├── reports/                           # Analysis reports and figures
│   ├── feature_analysis/
│   │   ├── feature_importance.png
│   │   ├── feature_distributions.png
│   │   ├── correlation_matrix.png
│   │   └── discriminability_report.pdf
│   │
│   ├── classifier_evaluation/
│   │   ├── confusion_matrix.png
│   │   ├── per_emotion_performance.png
│   │   ├── cross_validation_results.csv
│   │   └── evaluation_report.pdf
│   │
│   └── data_exploration/
│       ├── session_summary.csv
│       ├── emotion_distribution.png
│       └── feeling_it_coverage.png
│
├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_data_slicer.py
│   ├── test_feature_extractor.py
│   ├── test_classifier.py
│   └── test_integration.py
│
├── docs/                              # Documentation
│   ├── architecture.md                # This file
│   ├── module_guides/
│   │   ├── module1_data_loader.md
│   │   ├── module2_data_slicer.md     ✅ DONE
│   │   ├── module3_feature_extractor.md ✅ DONE
│   │   ├── module4_feature_analyzer.md
│   │   ├── module5_classifier.md
│   │   └── module6_realtime.md
│   │
│   ├── api_reference.md               # API documentation
│   ├── emotion_framework.md           # Custom emotion definitions
│   └── installation.md                # Setup instructions
│
└── examples/                          # Example scripts and tutorials
    ├── example_01_load_and_slice.py
    ├── example_02_extract_features.py
    ├── example_03_train_classifier.py
    └── example_04_realtime_demo.py
```

---

## Module Breakdown

### ✅ **Implemented Modules**

#### Module 2: Data Slicer (`modules/data_slicer.py`)
- **Status**: ✅ Complete and tested
- **Lines**: ~500
- **Key Classes**: `DataSlicer`, `EmotionSegment`
- **Dependencies**: pandas, numpy
- **Documentation**: `docs/module_guides/module2_data_slicer.md`

#### Module 3: Feature Extractor (`modules/feature_extractor.py`)
- **Status**: ✅ Complete and tested
- **Lines**: ~600
- **Key Classes**: `EmotionFeatureExtractor`
- **Features**: 54 total (17 EDA, 13 cardiac, 19 respiratory, 5 multimodal)
- **Dependencies**: pandas, numpy, scipy
- **Documentation**: `docs/module_guides/module3_feature_extractor.md`

---

### 🚧 **To Be Implemented**

#### Module 1: Data Loader (`modules/data_loader.py`)
```python
class Session:
    """Represents a recording session with metadata"""
    
class DataLoader:
    def load_sessions(filepaths: List[str]) -> List[Session]
    def validate_session(session: Session) -> bool
    def get_summary_stats(sessions: List[Session]) -> dict
```

**Purpose**: 
- Load multiple CSV files
- Validate structure (required columns)
- Handle different naming conventions (PPG/heart, EDA/gsr, etc.)
- Aggregate session metadata

**Priority**: Medium (can use pandas directly for now)

---

#### Module 4: Feature Analyzer (`modules/feature_analyzer.py`)
```python
class FeatureAnalyzer:
    def compute_feature_importance(method='random_forest') -> pd.DataFrame
    def statistical_tests(test='anova') -> pd.DataFrame
    def visualize_distributions(top_n=20) -> None
    def discriminability_matrix() -> pd.DataFrame
    def generate_report(output_path: str) -> None
```

**Purpose**:
- Rank features by discriminative power
- Statistical significance tests (ANOVA, t-tests)
- Visualizations (distributions, correlations, PCA)
- Generate analysis reports

**Priority**: **HIGH** (next critical step for validation)

---

#### Module 5: Classifier (`modules/classifier.py`)
```python
class EmotionClassifier:
    def train(features_df: pd.DataFrame) -> None
    def cross_validate(cv_method='loo') -> dict
    def predict(features: dict) -> tuple[str, dict]
    def save(filepath: str) -> None
    def load(filepath: str) -> None
```

**Methods**:
- Template matching (recommended)
- SVM, Random Forest
- Ensemble

**Purpose**:
- Train emotion classifier
- Cross-validation
- Performance metrics
- Model persistence

**Priority**: **HIGH** (needed for deployment)

---

#### Module 6: Real-Time Classifier (`modules/realtime_classifier.py`)
```python
class RealTimeClassifier:
    def process_sample(heart, gsr, respiration) -> None
    def classify_current_window() -> tuple[str, dict]
    def send_to_ossia(emotion, confidences) -> None
```

**Purpose**:
- Buffer incoming sensor data
- Real-time feature extraction
- Emotion prediction
- OSC/UDP communication to Ossia Score

**Priority**: Medium (after classifier training)

---

## Configuration Files

### `configs/emotions.yaml`
```yaml
# Emotion label mappings
emotions:
  abbreviations:
    war: "Wariness"          # Example mapping
    nul: "Neutral"
    joy: "Joy"
    sad: "Sadness"
    fea: "Fear"
    ang: "Anger"
    # Add your 23 emotions here
    
  target_emotions:
    # Emotions to use for training
    - joy
    - sad
    - fear
    - calm
    - anger
    - surprise
    
  exclude_emotions:
    # Emotions to exclude (baselines, transitions)
    - nul
    - baseline
    - transition
```

### `configs/pipeline_config.yaml`
```yaml
# Pipeline configuration
data_slicing:
  sampling_rate: 100
  
  emotion_filter:
    include: ['joy', 'sad', 'fear', 'calm', 'anger', 'surprise']
    exclude: ['nul', 'baseline']
  
  feeling_it:
    require: true
    min_ratio: 0.7
    time_tolerance_s: 2.0
  
  windowing:
    window_size_s: 30.0
    overlap_s: 15.0
    min_feeling_ratio: 0.6
  
  quality:
    min_duration_s: 10.0
    max_flat_ratio: 0.9
    check_signal_validity: true

feature_extraction:
  normalize_per_session: true
  
classifier:
  method: 'template_matching'
  feature_subset: 20  # Use top 20 features
  cv_method: 'leave_one_out'
```

---

## Workflow Scripts

### `scripts/01_explore_data.py`
```python
"""
Explore raw data from all sessions.
Output: Summary statistics, emotion distributions, feeling_it coverage
"""

# Load all sessions
# Print summary per session
# Visualize emotion distributions
# Check feeling_it coverage per emotion
```

### `scripts/02_slice_data.py`
```python
"""
Apply Module 2 filtering to create training segments.
Output: processed/segments_filtered.pkl, processed/windows_30s.pkl
"""

# Load config
# Load all sessions
# Apply emotion filtering
# Apply feeling_it filtering
# Create fixed windows
# Save processed segments
```

### `scripts/03_extract_features.py`
```python
"""
Apply Module 3 feature extraction to all windows.
Output: features/features_all.csv
"""

# Load processed windows
# For each window:
#   - Extract features
#   - Add metadata (emotion, feeling_it_ratio, etc.)
# Save features DataFrame
```

### `scripts/04_analyze_features.py`
```python
"""
Apply Module 4 analysis to validate features.
Output: reports/feature_analysis/*
"""

# Load features
# Compute feature importance
# Statistical tests
# Generate visualizations
# Create report
```

### `scripts/05_train_classifier.py`
```python
"""
Train and validate emotion classifier.
Output: models/trained_classifiers/*, reports/classifier_evaluation/*
"""

# Load features
# Select top features
# Train classifier
# Cross-validate
# Generate evaluation report
# Save trained model
```

### `scripts/00_process_all.py`
```python
"""
End-to-end pipeline: raw data → trained classifier
"""

# Run scripts 01-05 in sequence
# Generate final report
```

---

## Example Usage Patterns

### Pattern 1: Interactive Exploration (Jupyter)
```python
# In notebooks/01_data_exploration.ipynb

from modules.data_slicer import DataSlicer
import pandas as pd

# Load single session
data = pd.read_csv('../data/raw/session_001.csv')

# Explore
slicer = DataSlicer(sampling_rate=100)
segments = slicer.session_to_segments(data, session_id='s1')
slicer.print_summary(segments)

# Visualize
# ... interactive plots
```

### Pattern 2: Batch Processing (Scripts)
```bash
# Command line workflow
python scripts/02_slice_data.py --config configs/pipeline_config.yaml
python scripts/03_extract_features.py --input data/processed/windows_30s.pkl
python scripts/04_analyze_features.py --input data/features/features_all.csv
python scripts/05_train_classifier.py --features data/features/features_all.csv
```

### Pattern 3: Programmatic Pipeline
```python
# In scripts/00_process_all.py

from modules.data_slicer import DataSlicer
from modules.feature_extractor import EmotionFeatureExtractor
from modules.classifier import EmotionClassifier
import yaml

# Load config
config = yaml.safe_load(open('configs/pipeline_config.yaml'))

# Step 1: Slice data
# Step 2: Extract features
# Step 3: Analyze features
# Step 4: Train classifier
# Step 5: Save everything
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAW DATA                                    │
│  data/raw/session_001.csv, session_002.csv, ...                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────────┐
                  │  Module 1: Loader  │
                  └────────┬───────────┘
                           │ List[Session]
                           ▼
                  ┌────────────────────┐
                  │  Module 2: Slicer  │ ✅ DONE
                  └────────┬───────────┘
                           │ List[EmotionSegment]
                           ▼
         ┌─────────────────────────────────────┐
         │  data/processed/segments_filtered.pkl│
         └──────────────┬──────────────────────┘
                        │
                        ▼
               ┌────────────────────┐
               │ Module 3: Features │ ✅ DONE
               └────────┬───────────┘
                        │ features_df
                        ▼
         ┌─────────────────────────────┐
         │  data/features/features_all.csv │
         └──────────┬──────────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
  ┌───────────────┐   ┌──────────────┐
  │ Module 4:     │   │ Module 5:    │
  │ Analyzer      │   │ Classifier   │
  └───────┬───────┘   └──────┬───────┘
          │                  │
          ▼                  ▼
  ┌───────────────┐   ┌──────────────────┐
  │ reports/      │   │ models/          │
  │ feature_      │   │ trained_         │
  │ analysis/     │   │ classifiers/     │
  └───────────────┘   └──────┬───────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Module 6:        │
                     │ Real-Time        │
                     │ Classifier       │
                     └────────┬─────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Ossia Score        │
                   │  (via OSC/UDP)      │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │  Autolume           │
                   │  (StyleGAN)         │
                   └─────────────────────┘
```

---

## Dependencies (requirements.txt)

```txt
# Core
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Machine Learning
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Configuration
pyyaml>=6.0

# Real-time Communication (Module 6)
python-osc>=1.8.0

# Optional: Advanced features
# tslearn>=0.6.0        # Time series analysis
# tsfresh>=0.20.0       # Automated feature engineering
# optuna>=3.0.0         # Hyperparameter optimization

# Development
pytest>=7.4.0
jupyter>=1.0.0
black>=23.0.0
```

---

## Git Setup (.gitignore)

```gitignore
# Data (too large for git)
data/raw/
data/processed/
data/features/
models/trained_classifiers/
models/baselines/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Reports (optional - might want to track some)
reports/*.png
reports/*.pdf

# Temporary
*.tmp
*.log
```

---

## Installation & Setup

### Option 1: Standard Install
```bash
# Clone repository
git clone <repository_url>
cd crocodile_emotion_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Option 2: Package Install
```bash
# Install as package
pip install -e .

# Now modules are importable anywhere
from modules.data_slicer import DataSlicer
from modules.feature_extractor import EmotionFeatureExtractor
```

---

## Current Status

### ✅ Completed (Ready to Use)
- Module 2: Data Slicer (100%)
- Module 3: Feature Extractor (100%)
- Documentation for both modules
- Example workflows
- Tested on sample data

### 🚧 In Progress
- Documentation structure

### 📋 To Do
- Module 1: Data Loader
- Module 4: Feature Analyzer (HIGH PRIORITY)
- Module 5: Classifier (HIGH PRIORITY)
- Module 6: Real-Time Classifier
- Workflow scripts (01-06)
- Configuration files
- Unit tests

---

## Next Implementation Priority

**Recommended Order:**

1. **Module 4: Feature Analyzer** ← NEXT
   - Validate which of 54 features actually work
   - Critical for understanding your data
   - ~300 lines, 1-2 days

2. **Module 5: Classifier**
   - Template matching implementation
   - Cross-validation
   - ~400 lines, 2-3 days

3. **Workflow Scripts**
   - 02_slice_data.py
   - 03_extract_features.py
   - 04_analyze_features.py
   - ~200 lines each, 1 day

4. **Module 1: Data Loader** (optional)
   - Can use pandas directly for now
   - Implement later if needed

5. **Module 6: Real-Time Classifier**
   - After classifier is trained
   - Integration with Ossia Score
   - ~300 lines, 2-3 days

---

## Questions?

- **Where should I put my CSV files?** → `data/raw/`
- **Where will processed data go?** → `data/processed/`
- **Where are trained models saved?** → `models/trained_classifiers/`
- **How do I run the pipeline?** → See `scripts/` or use modules directly
- **Where's the documentation?** → `docs/module_guides/`
