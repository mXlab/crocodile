"""
Test script for Feature Extractor module
"""
import pandas as pd
from modules.data_slicer import DataSlicer
from modules.feature_extractor import EmotionFeatureExtractor

# Load sample data
data = pd.read_csv('data/raw/emotion_biodata_1S.csv')

print("="*80)
print("FEATURE EXTRACTOR TEST")
print("="*80)

# First, get a segment using DataSlicer
slicer = DataSlicer(sampling_rate=100)
segments = slicer.session_to_segments(
    data,
    session_id='test',
    emotion_col='emotion',
    feeling_col='feeling_it',
    signal_cols=['heart', 'gsr', 'respiration']
)

# Get the longest segment for testing
test_segment = max(segments, key=lambda s: s.duration)
print(f"Testing with segment: {test_segment.emotion}, duration: {test_segment.duration:.1f}s")

# Initialize feature extractor
extractor = EmotionFeatureExtractor(sampling_rate=100)

# Extract features
print("\nExtracting features...")
features = extractor.extract_all_features(
    eda_raw=test_segment.signals['gsr'],
    ppg_raw=test_segment.signals['heart'],
    resp_raw=test_segment.signals['respiration'],
    window_size_s=test_segment.duration
)

# Convert to vector
feature_vector, feature_names = extractor.features_to_vector(features)

print(f"\n✓ Extracted {len(feature_vector)} features")

# Show summary by modality
print("\nFeatures by modality:")
for modality in ['eda', 'cardiac', 'respiratory', 'multimodal']:
    count = len(features[modality])
    print(f"  {modality:15s}: {count} features")

# Show a few example features
print("\nExample features:")
for i, (name, value) in enumerate(zip(feature_names[:10], feature_vector[:10])):
    print(f"  {name:40s}: {value:10.4f}")

print("\n✓ Feature Extractor test complete!")
