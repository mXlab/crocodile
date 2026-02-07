# Flexible Windowing System - User Guide

## Overview

The flexible windowing system allows you to control the **temporal resolution** of your training data by adjusting how physiological data is sliced into samples.

**Key Insight:** By varying window size and stride, you can generate anywhere from **76 samples** (one per segment) to **5,000+ samples** (dense sliding windows) from the same raw data.

---

## Three Windowing Modes

### **1. Segment Mode** (Original)
**One window per emotion segment**

```python
windows = slicer.create_windows(data, 'session1', window_mode='segment')
# Result: ~76 windows
```

**Characteristics:**
- ✅ Simple and interpretable
- ✅ No overlap concerns
- ✅ Fast to compute
- ❌ Very few training samples
- ❌ Cannot capture temporal dynamics

**Use for:** Feature analysis, quick validation

---

### **2. Sliding Mode** (Maximum Data)
**Fixed windows that slide across entire session, may cross emotion boundaries**

```python
windows = slicer.create_windows(
    data, 'session1',
    window_mode='sliding',
    window_size_s=30,    # 30-second windows
    stride_s=5,          # 5-second stride (83% overlap)
    min_purity=0.8       # Only use windows that are 80%+ one emotion
)
# Result: ~1,000-1,500 windows
```

**Characteristics:**
- ✅ Maximum training data
- ✅ Good for real-time simulation
- ✅ Captures temporal dynamics
- ⚠️ May cross emotion boundaries (use min_purity filter)
- ⚠️ High autocorrelation

**Use for:** Real-time deployment, maximum training data

---

### **3. Hybrid Mode** (Recommended)
**Multiple windows per segment, never crosses boundaries**

```python
windows = slicer.create_windows(
    data, 'session1',
    window_mode='hybrid',
    window_size_s=30,
    stride_s=5
)
# Result: ~800-1,200 windows
```

**Characteristics:**
- ✅ Much more data than segment mode
- ✅ Maintains emotion purity (no boundary crossing)
- ✅ Good balance of data vs quality
- ✅ Proven effective in emotion recognition
- ⚠️ Moderate autocorrelation

**Use for:** Classifier training (recommended!)

---

## Configuration Presets

We provide pre-configured windowing strategies in `configs/windowing_config.yaml`:

### **Training Presets**

```yaml
# Conservative (50% overlap)
training_conservative:
  mode: 'hybrid'
  window_size_s: 30
  stride_s: 15
  overlap_ratio: 0.50
  expected_samples: 300-500

# Standard (83% overlap) ← RECOMMENDED
training_standard:
  mode: 'hybrid'
  window_size_s: 30
  stride_s: 5
  overlap_ratio: 0.833
  expected_samples: 800-1500

# Dense (93% overlap)
training_dense:
  mode: 'hybrid'
  window_size_s: 30
  stride_s: 2
  overlap_ratio: 0.933
  expected_samples: 2000-3000
```

### **Real-Time Presets**

```yaml
# Smooth (update every second)
realtime_smooth:
  mode: 'sliding'
  window_size_s: 30
  stride_s: 1
  overlap_ratio: 0.967
  expected_samples: 5000-6000

# Balanced (update every 5 seconds)
realtime_balanced:
  mode: 'sliding'
  window_size_s: 30
  stride_s: 5
  overlap_ratio: 0.833
  expected_samples: 1000-1500
```

---

## Using the System

### **Method 1: Use Presets (Easiest)**

```bash
# Use training_standard preset (recommended)
python scripts/process_all_sessions_v2.py --preset training_standard

# Use different preset
python scripts/process_all_sessions_v2.py --preset training_dense

# Use segment mode (original behavior)
python scripts/process_all_sessions_v2.py --preset segment_mode
```

### **Method 2: Custom Parameters**

```bash
# Specify parameters directly
python scripts/process_all_sessions_v2.py \
    --mode hybrid \
    --window-size 30 \
    --stride 5

# Fine-grained sliding windows
python scripts/process_all_sessions_v2.py \
    --mode sliding \
    --window-size 30 \
    --stride 1
```

### **Method 3: Python API**

```python
from modules.data_slicer import DataSlicer
import pandas as pd

data = pd.read_csv('session.csv')
slicer = DataSlicer(sampling_rate=100)

# Hybrid mode (recommended for training)
windows = slicer.create_windows(
    data,
    session_id='session1',
    window_mode='hybrid',
    window_size_s=30,
    stride_s=5
)

print(f"Created {len(windows)} training samples")
```

---

## Understanding Overlap

**Overlap ratio** = `1 - (stride / window_size)`

Examples with 30-second windows:

| Stride | Overlap | Training Samples | Use Case |
|--------|---------|------------------|----------|
| 30s | 0% | ~190 | Independent windows, cross-validation |
| 15s | 50% | ~350 | Conservative training |
| 10s | 67% | ~500 | Moderate training |
| 5s | 83% | ~1,100 | **Standard training (recommended)** |
| 2s | 93% | ~2,500 | Dense training |
| 1s | 97% | ~5,600 | Real-time simulation |

---

## Window Object Structure

Each window is an `EmotionWindow` object:

```python
window = EmotionWindow(
    window_id='session1_hybrid_000',    # Unique identifier
    session_id='session1',               # Session ID
    emotion='joy',                       # Emotion label
    emotion_purity=1.0,                  # Proportion of window with this emotion
    start_idx=0,                         # Starting sample index
    end_idx=3000,                        # Ending sample index
    start_time=0.0,                      # Starting time (seconds)
    end_time=30.0,                       # Ending time (seconds)
    duration=30.0,                       # Duration (seconds)
    signals={                            # Signal arrays
        'heart': array([...]),
        'gsr': array([...]),
        'respiration': array([...])
    },
    feeling_it=True,                     # Feeling_it pressed?
    feeling_it_ratio=0.85,               # % of window with feeling_it
    parent_segment_id='session1_seg002', # Parent segment (hybrid mode)
    metadata={'window_mode': 'hybrid'}   # Additional info
)
```

---

## Filtering Windows

After creating windows, you can filter them:

### **By Emotion**
```python
# Include only specific emotions
windows = slicer.filter_by_emotions(
    windows,
    include_emotions=['joy', 'sad', 'fear', 'calm']
)

# Exclude emotions
windows = slicer.filter_by_emotions(
    windows,
    exclude_emotions=['nul', 'baseline']
)
```

### **By Purity** (sliding mode)
```python
# Only use windows that are 90%+ one emotion
windows = slicer.filter_by_purity(
    windows,
    min_purity=0.9
)
```

### **By Feeling_it**
```python
# Only use windows where actress pressed pedal
windows = slicer.filter_by_feeling_it(
    windows,
    require_feeling_it=True,
    min_feeling_ratio=0.7
)
```

### **By Quality**
```python
# Filter by duration and signal quality
windows = slicer.filter_by_quality(
    windows,
    min_duration_s=10.0,
    max_duration_s=60.0,
    check_signal_validity=True
)
```

---

## Cross-Validation Considerations

**Critical:** With overlapping windows, you must use **time-aware cross-validation**:

```python
from sklearn.model_selection import GroupKFold

# Create groups: windows from same segment get same group ID
group_ids = [window.parent_segment_id or window.window_id for window in windows]

# Use GroupKFold instead of regular KFold
cv = GroupKFold(n_splits=5)

for train_idx, test_idx in cv.split(X, y, groups=group_ids):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train and evaluate
    # Ensures no data leakage from overlapping windows
```

**Why this matters:**
- Overlapping windows from same segment are highly correlated
- Regular K-Fold would put overlapping windows in both train and test sets
- This causes **data leakage** and **inflated accuracy**
- GroupKFold ensures all windows from a segment stay together

---

## Choosing the Right Configuration

### **Decision Tree:**

```
What are you doing?
│
├─ Feature analysis (Module 4)?
│  └─ Use: segment_mode
│     Result: 76 samples, fast, interpretable
│
├─ Training classifier (Module 5)?
│  │
│  ├─ Want maximum data?
│  │  └─ Use: training_dense
│  │     Result: 2,000-3,000 samples
│  │
│  ├─ Balanced (recommended)?
│  │  └─ Use: training_standard
│  │     Result: 800-1,500 samples
│  │
│  └─ Worried about autocorrelation?
│     └─ Use: training_conservative
│        Result: 300-500 samples
│
├─ Real-time deployment (Module 6)?
│  │
│  ├─ Need smooth predictions?
│  │  └─ Use: realtime_smooth
│  │     Result: Update every 1 second
│  │
│  └─ Balanced latency/computation?
│     └─ Use: realtime_balanced
│        Result: Update every 5 seconds
│
└─ Research/experiments?
   └─ Try multiple configs, plot accuracy vs sample count
```

---

## Output Files

Processing with different modes creates separate output files:

```bash
# Segment mode
data/processed/
├── all_features_segment.csv
├── all_windows_segment.pkl
└── windowing_config_segment.yaml

# Hybrid mode (training_standard)
data/processed/
├── all_features_hybrid.csv
├── all_windows_hybrid.pkl
└── windowing_config_hybrid.yaml

# Sliding mode (realtime)
data/processed/
├── all_features_sliding.csv
├── all_windows_sliding.pkl
└── windowing_config_sliding.yaml
```

This lets you compare different windowing strategies!

---

## Comparing Configurations

### **Example: Compare Training Modes**

```bash
# Generate data with different configurations
python scripts/process_all_sessions_v2.py --preset training_conservative
python scripts/process_all_sessions_v2.py --preset training_standard
python scripts/process_all_sessions_v2.py --preset training_dense

# Train classifier on each
python scripts/train_classifier.py --features data/processed/all_features_hybrid.csv

# Compare results
python scripts/compare_windowing_configs.py
```

**Expected results:**
```
Configuration         Samples    Accuracy    Training Time
------------------------------------------------------------
training_conservative    350       68%         2 min
training_standard       1,150       72%         5 min
training_dense          2,400       73%        12 min
```

**Sweet spot:** Usually around **800-1,500 samples** (training_standard)

---

## Best Practices

### **1. Start with Standard**
```bash
# Always start here
python scripts/process_all_sessions_v2.py --preset training_standard
```

### **2. Validate with Independent Windows**
```bash
# For cross-validation, use non-overlapping windows
python scripts/process_all_sessions_v2.py --preset experimental_independent
```

### **3. Match Training and Deployment**
- Train with `training_standard` (30s window, 5s stride)
- Deploy with `realtime_balanced` (30s window, 5s stride)
- **Same window size** ensures consistency!

### **4. Document Your Choice**
```python
# Always save windowing config with your model
config_used = {
    'mode': 'hybrid',
    'window_size_s': 30,
    'stride_s': 5,
    'preset': 'training_standard'
}

with open('model_windowing_config.yaml', 'w') as f:
    yaml.dump(config_used, f)
```

---

## Troubleshooting

### **Too Few Samples After Filtering**
```python
# Check windowing info
info = slicer.get_windowing_info(windows)
print(info)

# Try less restrictive filters
windows = slicer.create_windows(..., min_purity=0.5)  # Lower threshold
```

### **Out of Memory**
```python
# Use coarser stride
windows = slicer.create_windows(..., stride_s=10)  # Instead of stride_s=2

# Or process in batches
for csv_file in csv_files:
    windows = process_single_file(csv_file)
    features = extract_and_save(windows)
    del windows  # Free memory
```

### **Cross-Validation Gives Inflated Accuracy**
```python
# Use GroupKFold, not regular KFold!
from sklearn.model_selection import GroupKFold

groups = [w.parent_segment_id for w in windows]
cv = GroupKFold(n_splits=5)
```

---

## Summary

**Flexible windowing gives you control over the data quantity/quality trade-off:**

- **segment_mode**: 76 samples, fast, interpretable
- **training_standard**: ~1,000 samples, balanced ← **Use this!**
- **realtime_smooth**: ~5,000 samples, smooth predictions

**Key parameters:**
- `window_size_s`: How much history to use (typical: 30s)
- `stride_s`: How often to sample (typical: 5s for training, 1s for real-time)
- `mode`: Where windows can cross boundaries

**Remember:** Use GroupKFold for cross-validation with overlapping windows!

---

## Next Steps

1. ✅ Install updated Module 2 with flexible windowing
2. ▶️ Process data with `training_standard` preset
3. ▶️ Compare sample counts: segment vs hybrid
4. ▶️ Train classifier (Module 5) with windowed data
5. ▶️ Deploy with matching window configuration (Module 6)
