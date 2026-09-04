# Module 2: Flexible Windowing - Installation Checklist

## Files to Install

You have 4 files to install:

1. **modules/data_slicer_v2.py** - Updated DataSlicer with flexible windowing
2. **configs/windowing_config.yaml** - Pre-configured windowing presets
3. **scripts/process_all_sessions_v2.py** - Updated processing script
4. **docs/guides/Flexible_Windowing_Guide.md** - Complete user guide

---

## Installation Steps

### **Step 1: Backup Your Current Module**

```bash
cd biodata_pipeline

# Backup existing data_slicer.py
cp modules/data_slicer.py modules/data_slicer_backup.py
echo "✓ Backed up original data_slicer.py"
```

### **Step 2: Install Updated Module**

```bash
# Replace with new version
cp data_slicer_v2.py modules/data_slicer.py

echo "✓ Installed flexible windowing module"
```

**Or keep both versions:**
```bash
# Keep old version, add new as separate file
cp data_slicer_v2.py modules/data_slicer_windowing.py

# Then import as: from modules.data_slicer_windowing import DataSlicer
```

### **Step 3: Install Configuration File**

```bash
# Create configs directory if it doesn't exist
mkdir -p configs

# Copy windowing config
cp windowing_config.yaml configs/

echo "✓ Installed windowing configuration"
```

### **Step 4: Install Updated Processing Script**

```bash
# Keep old script, add new one
cp process_all_sessions_v2.py scripts/

echo "✓ Installed windowing-enabled processing script"
```

### **Step 5: Install Documentation**

```bash
# Copy guide to docs
cp Flexible_Windowing_Guide.md docs/guides/

echo "✓ Installed windowing guide"
```

### **Step 6: Install PyYAML (if not already installed)**

```bash
pip install pyyaml

echo "✓ Installed dependencies"
```

---

## Verify Installation

### **Test 1: Import Module**

```bash
python << 'EOF'
from modules.data_slicer import DataSlicer
print("✓ Module imports successfully")

# Check for windowing methods
slicer = DataSlicer(sampling_rate=100)
assert hasattr(slicer, 'create_windows'), "Missing create_windows method"
print("✓ Flexible windowing methods present")
EOF
```

### **Test 2: Check Configuration**

```bash
python << 'EOF'
import yaml
with open('configs/windowing_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f"✓ Loaded {len(config)} windowing presets")
print(f"  Presets: {list(config.keys())}")
EOF
```

### **Test 3: Run Processing Script (Help)**

```bash
python scripts/process_all_sessions_v2.py --help

# Should show:
#   --preset PRESET
#   --mode {segment,sliding,hybrid}
#   --window-size WINDOW_SIZE
#   --stride STRIDE
```

---

## Quick Test Run

### **Generate Different Windowing Resolutions**

```bash
cd biodata_pipeline

# 1. Segment mode (original, 76 samples)
python scripts/process_all_sessions_v2.py --preset segment_mode

# 2. Training standard (recommended, ~1,000 samples)
python scripts/process_all_sessions_v2.py --preset training_standard

# 3. Training dense (maximum data, ~2,000 samples)
python scripts/process_all_sessions_v2.py --preset training_dense

# Check outputs
ls data/processed/
# Should see:
#   all_features_segment.csv
#   all_features_hybrid.csv (from training_standard)
#   all_features_hybrid.csv (or _dense suffix)
```

---

## Compare Results

### **Check Sample Counts**

```bash
python << 'EOF'
import pandas as pd

segment = pd.read_csv('data/processed/all_features_segment.csv')
hybrid = pd.read_csv('data/processed/all_features_hybrid.csv')

print(f"Segment mode:  {len(segment):4d} samples")
print(f"Hybrid mode:   {len(hybrid):4d} samples")
print(f"Improvement:   {len(hybrid)/len(segment):.1f}x more data")
EOF
```

**Expected output:**
```
Segment mode:    76 samples
Hybrid mode:   1150 samples
Improvement:   15.1x more data
```

---

## File Structure After Installation

```
biodata_pipeline/
├── modules/
│   ├── data_slicer.py              ✅ Updated with flexible windowing
│   ├── data_slicer_backup.py       (Optional) Original version
│   ├── feature_extractor.py
│   └── feature_analyzer.py
│
├── scripts/
│   ├── process_all_sessions.py      Original script
│   ├── process_all_sessions_v2.py   ✅ NEW: Windowing support
│   └── analyze_features.py
│
├── configs/
│   └── windowing_config.yaml        ✅ NEW: Windowing presets
│
├── docs/
│   └── guides/
│       ├── DataSlicer_Usage_Guide.md
│       └── Flexible_Windowing_Guide.md  ✅ NEW
│
└── data/
    └── processed/
        ├── all_features_segment.csv     76 samples
        └── all_features_hybrid.csv      ~1,000 samples
```

---

## Usage Examples

### **Example 1: Quick Start (Use Preset)**

```bash
# Use recommended training configuration
python scripts/process_all_sessions_v2.py --preset training_standard

# Output: data/processed/all_features_hybrid.csv (~1,000 samples)
```

### **Example 2: Custom Configuration**

```bash
# Specify parameters directly
python scripts/process_all_sessions_v2.py \
    --mode hybrid \
    --window-size 30 \
    --stride 5 \
    --output-suffix "_custom"

# Output: data/processed/all_features_custom.csv
```

### **Example 3: Python API**

```python
from modules.data_slicer import DataSlicer
import pandas as pd

# Load data
data = pd.read_csv('data/raw/emotion_biodata_1S.csv')

# Create slicer
slicer = DataSlicer(sampling_rate=100)

# Create windows with hybrid mode
windows = slicer.create_windows(
    data,
    session_id='session1',
    window_mode='hybrid',
    window_size_s=30,
    stride_s=5,
    emotion_col='emotion',
    signal_cols=['heart', 'gsr', 'respiration']
)

print(f"Created {len(windows)} windows")
print(f"First window: {windows[0]}")

# Get windowing info
info = slicer.get_windowing_info(windows)
print(f"Windowing config: {info}")
```

---

## Troubleshooting

### **Issue: "No module named 'yaml'"**

**Solution:**
```bash
pip install pyyaml
```

### **Issue: "FileNotFoundError: windowing_config.yaml"**

**Solution:**
```bash
# Make sure config is in right place
ls configs/windowing_config.yaml

# Or copy it
cp /path/to/windowing_config.yaml configs/
```

### **Issue: "AttributeError: 'DataSlicer' object has no attribute 'create_windows'"**

**Solution:**
```bash
# You're using the old module, update it
cp data_slicer_v2.py modules/data_slicer.py
```

### **Issue: ImportError when replacing data_slicer.py**

**Solution:**
```bash
# The old EmotionSegment is now EmotionWindow
# But we kept backwards compatibility, so it should work
# If you have import issues, check:

python << 'EOF'
from modules.data_slicer import EmotionWindow, EmotionSegment
print(EmotionSegment == EmotionWindow)  # Should be True
EOF
```

---

## Backwards Compatibility

The updated module is **100% backwards compatible**:

```python
# Old code still works!
segments = slicer.session_to_segments(data, 'session1')

# This is equivalent to:
windows = slicer.create_windows(data, 'session1', window_mode='segment')
```

**EmotionSegment** is now an alias for **EmotionWindow**, so existing code continues to work.

---

## Next Steps

After installation:

### **1. Test Different Windowing Modes**

```bash
# Compare sample counts
python scripts/process_all_sessions_v2.py --preset segment_mode
python scripts/process_all_sessions_v2.py --preset training_standard
python scripts/process_all_sessions_v2.py --preset training_dense

# Check results
python << 'EOF'
import pandas as pd
segment = pd.read_csv('data/processed/all_features_segment.csv')
hybrid = pd.read_csv('data/processed/all_features_hybrid.csv')
print(f"Segment: {len(segment)}, Hybrid: {len(hybrid)}")
EOF
```

### **2. Run Feature Analysis on Windowed Data**

```bash
# Analyze the increased dataset
python scripts/analyze_features.py --features data/processed/all_features_hybrid.csv
```

### **3. Proceed to Module 5: Classifier**

With ~1,000 training samples (instead of 76), you can now:
- Train more robust classifier
- Support 23 emotions better
- Use cross-validation properly

---

## Key Configuration Presets

Quick reference for choosing a preset:

| Preset | Samples | Use Case |
|--------|---------|----------|
| `segment_mode` | ~76 | Feature analysis, quick tests |
| `training_conservative` | ~350 | Low autocorrelation |
| `training_standard` | ~1,100 | **Training (recommended)** |
| `training_dense` | ~2,400 | Maximum training data |
| `realtime_smooth` | ~5,600 | Deployment (1s updates) |
| `experimental_independent` | ~190 | Cross-validation |

---

## Documentation

- **User Guide:** `docs/guides/Flexible_Windowing_Guide.md`
- **Config Reference:** `configs/windowing_config.yaml`
- **Module Docstrings:** `modules/data_slicer.py`

---

## Summary

✅ **Flexible windowing installed!**

You can now generate:
- **76 samples** (segment mode)
- **~1,000 samples** (hybrid mode, recommended)
- **~5,000 samples** (sliding mode, real-time)

**All from the same raw data!**

Next: Process your data with `training_standard` preset and see the difference! 🚀
