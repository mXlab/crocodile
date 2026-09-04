# Migration Guide: --preset to --config

## Overview

The old `--preset` approach has been replaced with a more flexible `--config` system using YAML files.

---

## Quick Migration

### Old Commands → New Commands

**Old:**
```bash
python scripts/process_with_supersegments.py --preset training_standard
```

**New:**
```bash
python scripts/slice_and_evaluate.py --config configs/training_standard.yaml
```

---

## Preset Mapping

| Old Preset | New Config File | Description |
|------------|----------------|-------------|
| `training_standard` | `configs/training_standard.yaml` | 30s windows, 5s stride (medium) |
| `training_dense` | `configs/training_dense.yaml` | 30s windows, 2s stride (dense) |
| `training_moderate` | `configs/training_moderate.yaml` | 30s windows, 10s stride (light) |
| `training_independent` | `configs/training_independent.yaml` | 30s windows, 30s stride (no overlap) |

---

## Why Migrate?

### Old Approach Limitations

```python
# Hardcoded in script
if preset == 'training_standard':
    window_size = 30
    stride = 5
    # ... can't easily customize
```

**Problems:**
- ❌ Can't modify without editing code
- ❌ Can't add new presets easily
- ❌ Can't document reasoning
- ❌ Can't version control separately
- ❌ Can't override easily

### New Approach Benefits

```yaml
# configs/training_standard.yaml
window_size_s: 30.0
stride_s: 5.0
# Easy to read, modify, document
```

**Benefits:**
- ✅ External config files
- ✅ Easy to create new configs
- ✅ Well documented
- ✅ Version control friendly
- ✅ Easy overrides: `--config file.yaml --stride 3`

---

## Step-by-Step Migration

### Step 1: Install PyYAML

```bash
pip install pyyaml
```

### Step 2: Copy Config Files

```bash
cd ~/Documents/workspace/crocodile/biodata_pipeline
mkdir -p configs
# Copy all .yaml files to configs/
```

### Step 3: Use New Script

**Old workflow:**
```bash
# Step 1: Extract features with preset
python scripts/process_with_supersegments.py --preset training_dense

# Output: data/processed/all_features_dense.csv
```

**New workflow:**
```bash
# Step 1: Extract features continuously (once)
python scripts/extract_continuous_features.py

# Step 2: Slice with config
python scripts/slice_and_evaluate.py --config configs/training_dense.yaml
```

---

## Detailed Examples

### Example 1: Standard Training

**Old:**
```bash
python scripts/process_with_supersegments.py \
    --preset training_standard
```

**New:**
```bash
# Option A: Use config
python scripts/slice_and_evaluate.py \
    --config configs/training_standard.yaml

# Option B: Customize
python scripts/slice_and_evaluate.py \
    --config configs/training_standard.yaml \
    --include fea sad \
    --window-size 25
```

---

### Example 2: Dense Training

**Old:**
```bash
python scripts/process_with_supersegments.py \
    --preset training_dense \
    --emotions fea sad
```

**New:**
```bash
python scripts/slice_and_evaluate.py \
    --config configs/training_dense.yaml \
    --include fea sad
```

---

### Example 3: Custom Parameters

**Old:**
```bash
# Had to create new preset in code!
# Or use many command-line flags
python scripts/process_with_supersegments.py \
    --window-size 25 \
    --stride 3 \
    --min-purity 0.6 \
    --super-segment-size 40
```

**New:**
```bash
# Create custom config once
cat > configs/my_custom.yaml << EOF
features_file: data/processed/continuous_features.csv
include_emotions: [fea, sad]
window_size_s: 25.0
stride_s: 3.0
min_purity: 0.6
super_segment_size_s: 40.0
EOF

# Reuse easily
python scripts/slice_and_evaluate.py --config configs/my_custom.yaml
```

---

## Feature Comparison

| Feature | Old (--preset) | New (--config) |
|---------|---------------|----------------|
| **Ease of use** | ✅ Simple | ✅ Simple |
| **Flexibility** | ❌ Limited | ✅ Unlimited |
| **Documentation** | ❌ In code | ✅ In YAML |
| **Customization** | ❌ Edit code | ✅ Edit YAML |
| **Sharing configs** | ❌ Hard | ✅ Easy |
| **Version control** | ❌ Mixed with code | ✅ Separate files |
| **Override params** | ⚠️ Cumbersome | ✅ Easy |
| **Add new presets** | ❌ Edit code | ✅ Create YAML |

---

## Config File Structure

### Equivalent to Old Preset

**Old (hardcoded in Python):**
```python
PRESETS = {
    'training_standard': {
        'window_size': 30,
        'stride': 5,
        'min_purity': 0.7,
        'super_segment_size': 35
    }
}
```

**New (YAML file):**
```yaml
# configs/training_standard.yaml
window_size_s: 30.0
stride_s: 5.0
min_purity: 0.7
super_segment_size_s: 35.0

# Plus you can add:
include_emotions: [fea, sad]
n_folds: 5
n_estimators: 100
# ... and document everything!
```

---

## Backwards Compatibility

### Old Scripts Still Work

If you prefer, the old command-line interface still works:

```bash
# No config file needed
python scripts/slice_and_evaluate.py \
    --features data/processed/continuous_features.csv \
    --include fea sad \
    --window-size 30 \
    --stride 2 \
    --super-segment-size 45
```

### Mixing Approaches

You can use configs AND overrides:

```bash
# Start with preset, customize one parameter
python scripts/slice_and_evaluate.py \
    --config configs/training_standard.yaml \
    --window-size 25
```

---

## Recommended Workflow

### For Development

```bash
# Use quick_test config for fast iteration
python scripts/slice_and_evaluate.py --config configs/quick_test.yaml
```

### For Experiments

```bash
# Create experiment config
cp configs/training_standard.yaml configs/exp_2024-02-07.yaml
# Edit configs/exp_2024-02-07.yaml
python scripts/slice_and_evaluate.py --config configs/exp_2024-02-07.yaml
```

### For Production

```bash
# Use production config
python scripts/slice_and_evaluate.py --config configs/production.yaml
```

---

## Common Patterns

### Pattern 1: Compare Old Presets

**Old:**
```bash
for preset in training_standard training_dense training_moderate; do
    python scripts/process_with_supersegments.py --preset $preset
done
```

**New:**
```bash
for config in configs/training_*.yaml; do
    python scripts/slice_and_evaluate.py --config $config
done
```

---

### Pattern 2: Emotion Sweep

**Old:**
```bash
# Had to repeat preset for each emotion set
python scripts/process_with_supersegments.py --preset training_standard --emotions fea sad
python scripts/process_with_supersegments.py --preset training_standard --emotions sad ang
```

**New:**
```bash
# Keep windowing config, vary emotions
for emo in "fea sad" "sad ang" "fea sad joy"; do
    python scripts/slice_and_evaluate.py \
        --config configs/training_standard.yaml \
        --include $emo
done
```

---

## Summary

### Migration Checklist

- [x] Install PyYAML: `pip install pyyaml`
- [x] Copy config files to `configs/`
- [x] Copy `pipeline_config.py` to `modules/`
- [x] Copy updated `slice_and_evaluate.py` to `scripts/`
- [x] Test: `python scripts/slice_and_evaluate.py --config configs/training_standard.yaml`
- [x] Create custom configs as needed

### Key Changes

1. **Two-step workflow:**
   - Step 1: `extract_continuous_features.py` (once)
   - Step 2: `slice_and_evaluate.py --config` (many times)

2. **Config files replace presets:**
   - `training_standard.yaml` replaces `--preset training_standard`
   - Much more flexible and maintainable

3. **Command-line still works:**
   - Can use configs, command-line, or both
   - Command-line args override config values

---

## Need Help?

**Q: Can I keep using old scripts?**
A: Yes, but the new workflow is more flexible. Old scripts still work.

**Q: Do I have to use configs?**
A: No, command-line args still work fine. Configs are optional but recommended.

**Q: Can I mix old and new?**
A: Yes, `--config file.yaml --window-size 25` works great.

**Q: What if I want a new preset?**
A: Just create a new YAML file! Much easier than editing code.

---

**The config system is a strict superset of the old preset system - everything you could do before, plus much more!** 🚀
