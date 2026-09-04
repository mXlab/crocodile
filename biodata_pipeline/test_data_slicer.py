"""
Test script for Data Slicer module
"""
import pandas as pd
from modules.data_slicer import DataSlicer

# Load sample data
data = pd.read_csv('data/raw/emotion_biodata_1S.csv')

print("="*80)
print("DATA SLICER TEST")
print("="*80)

# Initialize slicer
slicer = DataSlicer(sampling_rate=100)

# Convert to segments
segments = slicer.session_to_segments(
    data,
    session_id='test_session',
    emotion_col='emotion',
    feeling_col='feeling_it',
    signal_cols=['heart', 'gsr', 'respiration']
)

print(f"\n✓ Created {len(segments)} segments")

# Print summary
slicer.print_summary(segments)

# Test filtering
print("\n" + "="*80)
print("TESTING FILTERS")
print("="*80)

# Filter by emotions (exclude 'nul')
filtered = slicer.filter_by_emotions(
    segments,
    exclude_emotions=['nul']
)

print(f"After emotion filter: {len(filtered)} segments")

# Try creating windows
windows = slicer.create_fixed_windows(
    segments,
    window_size_s=10.0,  # Small for test data
    overlap_s=5.0
)

print(f"Created {len(windows)} windows")

print("\n✓ Data Slicer test complete!")
