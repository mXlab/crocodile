# Data Slicer Module - Usage Guide

## Overview

The DataSlicer module handles slicing and filtering of emotion-labeled physiological data with `feeling_it` pedal integration. It's the core of your data preprocessing pipeline.

---

## Quick Start

```python
from data_slicer import DataSlicer
import pandas as pd

# Load your data
data = pd.read_csv('session_001.csv')

# Initialize slicer
slicer = DataSlicer(sampling_rate=100)

# Convert session to emotion segments
segments = slicer.session_to_segments(
    data,
    session_id='session_001',
    emotion_col='emotion',
    feeling_col='feeling_it',
    signal_cols=['heart', 'gsr', 'respiration']
)

# Get summary
slicer.print_summary(segments)
```

---

## Common Workflows

### Workflow 1: Filter by Emotions + Feeling_it Pedal

```python
# 1. Convert session to segments
segments = slicer.session_to_segments(data, session_id='session_001')

# 2. Exclude baseline/neutral emotions
segments = slicer.filter_by_emotions(
    segments,
    exclude_emotions=['nul', 'baseline', 'neutral']
)

# 3. Keep only data where actress was "feeling it"
segments = slicer.filter_by_feeling_it(
    segments,
    require_feeling_it=True,
    min_feeling_ratio=0.5,  # At least 50% of segment
    time_tolerance_s=2.0     # Include 2s before/after
)

# 4. Quality control
segments = slicer.filter_by_quality(
    segments,
    min_duration_s=10.0,
    max_flat_ratio=0.9  # Adjusted for low-variation EDA
)

# 5. Summary
slicer.print_summary(segments)
```

**Output:**
```
================================================================================
SEGMENT SUMMARY
================================================================================
emotion  n_segments  total_duration_s  avg_duration_s  n_with_feeling  feeling_duration_s  avg_feeling_ratio
    joy           3             45.2           15.07               3                45.2               0.85
    sad           2             32.1           16.05               2                32.1               0.92
   fear           4             58.3           14.58               4                58.3               0.78
  TOTAL           9            135.6           15.07               9               135.6               0.85
================================================================================
```

---

### Workflow 2: Extract Continuous Feeling_it Zones

```python
# Extract only continuous zones where feeling_it==1
feeling_zones = slicer.extract_feeling_zones(
    segments,
    min_zone_duration_s=5.0  # Minimum 5s continuous
)

print(f"Extracted {len(feeling_zones)} continuous feeling_it zones")

# Each zone is a separate segment with feeling_it_ratio=1.0
for zone in feeling_zones[:3]:
    print(f"  {zone}")
```

**Example Output:**
```
Extracted 12 continuous feeling_it zones
  EmotionSegment(id=session_001_seg003_zone00, emotion=joy, duration=8.5s, feeling_it=1.00)
  EmotionSegment(id=session_001_seg003_zone01, emotion=joy, duration=12.3s, feeling_it=1.00)
  EmotionSegment(id=session_001_seg005_zone00, emotion=sad, duration=15.2s, feeling_it=1.00)
```

---

### Workflow 3: Create Fixed-Size Training Windows

```python
# Create 30-second windows with 15-second overlap
windows = slicer.create_fixed_windows(
    segments,
    window_size_s=30.0,
    overlap_s=15.0,
    min_feeling_ratio=0.7  # At least 70% feeling_it in window
)

print(f"Created {len(windows)} training windows")
slicer.print_summary(windows)
```

**Use Case:** Perfect for feature extraction where you need consistent 30s windows for analysis.

---

### Workflow 4: Multi-Session Processing

```python
# Load multiple session files
session_files = [
    'session_001.csv',
    'session_002.csv',
    'session_003.csv'
]

all_segments = []

for filepath in session_files:
    data = pd.read_csv(filepath)
    session_id = filepath.split('/')[-1].replace('.csv', '')
    
    segments = slicer.session_to_segments(
        data,
        session_id=session_id,
        metadata={'participant': 'actress_LD', 'date': '2026-02-06'}
    )
    
    all_segments.extend(segments)

print(f"Loaded {len(all_segments)} total segments from {len(session_files)} sessions")

# Now filter across all sessions
filtered = slicer.filter_by_emotions(all_segments, include_emotions=['joy', 'sad', 'fear', 'calm'])
filtered = slicer.filter_by_feeling_it(filtered, require_feeling_it=True, min_feeling_ratio=0.6)

slicer.print_summary(filtered)
```

---

## API Reference

### DataSlicer Class

```python
DataSlicer(sampling_rate=100)
```

**Parameters:**
- `sampling_rate` (int): Sampling frequency in Hz (default: 100)

---

### 1. session_to_segments()

Convert continuous session data into discrete emotion segments.

```python
segments = slicer.session_to_segments(
    data,                          # pd.DataFrame with signals + labels
    session_id='session_001',      # Unique session identifier
    emotion_col='emotion',         # Column name for emotion labels
    feeling_col='feeling_it',      # Column name for pedal
    signal_cols=None,              # Auto-detect if None
    metadata={'participant': 'LD'} # Optional metadata
)
```

**Returns:** `List[EmotionSegment]`

**What it does:**
- Detects emotion label changes → creates segment boundaries
- Extracts signals for each segment
- Computes feeling_it statistics (ratio, indices)

---

### 2. filter_by_emotions()

Filter segments by emotion labels.

```python
# Include only specific emotions
filtered = slicer.filter_by_emotions(
    segments,
    include_emotions=['joy', 'sad', 'fear']
)

# Exclude baseline/neutral
filtered = slicer.filter_by_emotions(
    segments,
    exclude_emotions=['nul', 'baseline']
)
```

**Returns:** `List[EmotionSegment]`

---

### 3. filter_by_feeling_it()

Filter based on feeling_it pedal.

```python
filtered = slicer.filter_by_feeling_it(
    segments,
    require_feeling_it=True,    # Keep only segments with pedal pressed
    min_feeling_ratio=0.5,      # At least 50% of segment
    time_tolerance_s=2.0        # Expand 2s before/after pedal zones
)
```

**Parameters:**
- `require_feeling_it` (bool): If True, keep only segments with feeling_it==1
- `min_feeling_ratio` (float): Minimum proportion (0.0 to 1.0)
- `time_tolerance_s` (float): Extend zones by N seconds before/after

**Returns:** `List[EmotionSegment]`

---

### 4. extract_feeling_zones()

Extract continuous zones where feeling_it==1.

```python
zones = slicer.extract_feeling_zones(
    segments,
    min_zone_duration_s=5.0  # Minimum continuous zone length
)
```

**What it does:**
- Splits segments at feeling_it boundaries
- Each zone is continuous feeling_it==1
- Filters out zones shorter than minimum

**Returns:** `List[EmotionSegment]` where each has `feeling_it_ratio=1.0`

---

### 5. create_fixed_windows()

Create fixed-size overlapping windows.

```python
windows = slicer.create_fixed_windows(
    segments,
    window_size_s=30.0,      # Window size
    overlap_s=15.0,          # Overlap (50% in this case)
    min_feeling_ratio=0.7    # Minimum feeling_it in window
)
```

**Parameters:**
- `window_size_s` (float): Window duration in seconds
- `overlap_s` (float): Overlap between windows (must be < window_size_s)
- `min_feeling_ratio` (float): Minimum feeling_it proportion in window

**Returns:** `List[EmotionSegment]` with fixed duration

---

### 6. filter_by_quality()

Quality control filtering.

```python
quality_segments = slicer.filter_by_quality(
    segments,
    min_duration_s=10.0,       # Minimum segment length
    max_duration_s=120.0,      # Maximum segment length (optional)
    check_signal_validity=True, # Check for flat/invalid signals
    max_flat_ratio=0.9         # Maximum flatness (0.9 = 90% flat allowed)
)
```

**Signal Validity Checks:**
- Not all zeros
- Standard deviation > 1e-6
- Flat ratio < max_flat_ratio

**Note:** EDA/GSR signals can have low variation naturally, so use `max_flat_ratio=0.9` to avoid false rejections.

**Returns:** `List[EmotionSegment]`

---

### 7. get_summary() / print_summary()

Get statistics about segments.

```python
# DataFrame summary
summary_df = slicer.get_summary(segments)

# Print formatted summary
slicer.print_summary(segments)
```

**Summary includes:**
- Number of segments per emotion
- Total and average duration
- Number of segments with feeling_it
- Average feeling_it ratio

---

## EmotionSegment Object

Each segment is an `EmotionSegment` object with:

```python
segment = EmotionSegment(
    segment_id='session_001_seg003',       # Unique identifier
    session_id='session_001',              # Parent session
    emotion='joy',                         # Emotion label
    start_idx=4500,                        # Start sample index
    end_idx=7500,                          # End sample index
    start_time=45.0,                       # Start time (seconds)
    end_time=75.0,                         # End time (seconds)
    duration=30.0,                         # Duration (seconds)
    signals={                              # Signal data
        'heart': np.array([...]),
        'gsr': np.array([...]),
        'respiration': np.array([...])
    },
    feeling_it=True,                       # Any pedal press in segment
    feeling_it_ratio=0.85,                 # Proportion with pedal pressed
    feeling_it_indices=[120, 121, ...],    # Sample indices with pedal
    metadata={'participant': 'LD'}         # Additional info
)
```

**Access signals:**
```python
heart_signal = segment.signals['heart']
print(f"Heart signal: {len(heart_signal)} samples")
```

---

## Advanced Usage

### Custom Signal Column Names

If your CSV has different column names:

```python
segments = slicer.session_to_segments(
    data,
    session_id='custom',
    emotion_col='emotion_label',      # Your column name
    feeling_col='pedal_press',        # Your column name
    signal_cols=['ppg', 'eda', 'temp'] # Your column names
)
```

### Chain Multiple Filters

```python
# Pipeline approach
segments = slicer.session_to_segments(data, session_id='s1')
segments = slicer.filter_by_emotions(segments, include_emotions=['joy', 'sad'])
segments = slicer.filter_by_feeling_it(segments, require_feeling_it=True, min_feeling_ratio=0.7)
segments = slicer.filter_by_quality(segments, min_duration_s=15.0)
windows = slicer.create_fixed_windows(segments, window_size_s=30.0, overlap_s=15.0)

print(f"Final pipeline: {len(windows)} training windows")
```

### Export Segments for Feature Extraction

```python
# After slicing, prepare for Module 3 (Feature Extraction)
for segment in segments:
    print(f"Segment: {segment.segment_id}")
    print(f"  Emotion: {segment.emotion}")
    print(f"  Duration: {segment.duration}s")
    print(f"  Signals: {list(segment.signals.keys())}")
    print(f"  Feeling_it: {segment.feeling_it_ratio:.2f}")
    
    # Ready for feature extraction
    # features = feature_extractor.extract_features_single(segment)
```

---

## Common Issues & Solutions

### Issue 1: Quality filter removes all segments

**Problem:** `max_flat_ratio=0.5` is too strict for EDA signals

**Solution:** Use `max_flat_ratio=0.9` or disable signal validity check:
```python
segments = slicer.filter_by_quality(
    segments,
    min_duration_s=10.0,
    check_signal_validity=False  # Disable flatness check
)
```

---

### Issue 2: No feeling_it zones found

**Problem:** Sample doesn't have pedal presses (feeling_it always 0)

**Solution:** Use `min_feeling_ratio=0.0` or skip feeling_it filter:
```python
# Don't require feeling_it
segments = slicer.filter_by_feeling_it(
    segments,
    require_feeling_it=False,
    min_feeling_ratio=0.0
)
```

---

### Issue 3: Windows too short for emotion induction

**Problem:** Fixed windows split across emotion changes

**Solution:** Use `extract_feeling_zones()` first, then window:
```python
# First extract continuous feeling_it zones
zones = slicer.extract_feeling_zones(segments, min_zone_duration_s=30.0)

# Then create windows from zones (won't cross emotion boundaries)
windows = slicer.create_fixed_windows(zones, window_size_s=30.0, overlap_s=0.0)
```

---

## Integration with Module 3 (Feature Extraction)

```python
from data_slicer import DataSlicer
from feature_extractor import EmotionFeatureExtractor

# 1. Slice data
slicer = DataSlicer(sampling_rate=100)
segments = slicer.session_to_segments(data, session_id='s1')
segments = slicer.filter_by_emotions(segments, include_emotions=['joy', 'sad', 'fear'])
windows = slicer.create_fixed_windows(segments, window_size_s=30.0)

# 2. Extract features
extractor = EmotionFeatureExtractor(sampling_rate=100)

features_list = []
for window in windows:
    # Extract features from window
    features = extractor.extract_all_features(
        eda_raw=window.signals['gsr'],
        ppg_raw=window.signals['heart'],
        resp_raw=window.signals['respiration'],
        window_size_s=window.duration
    )
    
    # Add metadata
    features['segment_id'] = window.segment_id
    features['emotion'] = window.emotion
    features['feeling_it_ratio'] = window.feeling_it_ratio
    
    features_list.append(features)

print(f"Extracted features for {len(features_list)} windows")
```

---

## Next Steps

Now that you have Module 2 (Data Slicer), you can:

1. **Process your full dataset** - Load all session CSV files
2. **Extract quality segments** - Filter by emotions and feeling_it
3. **Feed to Module 3** - Extract features from segments
4. **Analyze features** - Use Module 4 to find discriminative features
5. **Train classifier** - Use Module 5 for emotion recognition

**Ready to process your full dataset?**
