"""
Batch process all session CSV files: Slice → Extract Features → Combine
"""
import pandas as pd
import pickle
from pathlib import Path
from modules.data_slicer import DataSlicer
from modules.feature_extractor import EmotionFeatureExtractor
import sys

def process_session(csv_path, session_id, slicer, extractor, 
                   emotion_filter=None, feeling_it_filter=True):
    """Process a single session file"""
    
    print(f"\n{'='*80}")
    print(f"PROCESSING: {session_id}")
    print(f"{'='*80}")
    
    # Load data
    print(f"Loading {csv_path}...")
    data = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(data):,} samples ({len(data)/100:.1f}s)")
    
    # Slice into segments
    print("Slicing data...")
    segments = slicer.session_to_segments(
        data,
        session_id=session_id,
        emotion_col='emotion',
        feeling_col='feeling_it',
        signal_cols=['heart', 'gsr', 'respiration']
    )
    
    # Optional: Filter by emotions
    if emotion_filter:
        segments = slicer.filter_by_emotions(
            segments,
            exclude_emotions=emotion_filter['exclude']
        )
    
    # Optional: Filter by feeling_it
    if feeling_it_filter:
        segments = slicer.filter_by_feeling_it(
            segments,
            require_feeling_it=True,
            min_feeling_ratio=0.5,
            time_tolerance_s=2.0
        )
    
    # Optional: Quality filter
    segments = slicer.filter_by_quality(
        segments,
        min_duration_s=5.0,
        check_signal_validity=False,
        max_flat_ratio=0.98
    )
    
    slicer.print_summary(segments)
    
    if len(segments) == 0:
        print("⚠ No segments after filtering, skipping feature extraction")
        return [], pd.DataFrame()
    
    # Extract features
    print(f"\nExtracting features from {len(segments)} segments...")
    features_list = []
    
    for i, segment in enumerate(segments, 1):
        if i % 10 == 0:
            print(f"  Processing segment {i}/{len(segments)}...", end='\r')
        
        try:
            features = extractor.extract_all_features(
                eda_raw=segment.signals['gsr'],
                ppg_raw=segment.signals['heart'],
                resp_raw=segment.signals['respiration'],
                window_size_s=segment.duration
            )
            
            feature_vector, feature_names = extractor.features_to_vector(features)
            
            # Build feature dict
            features_dict = {
                'segment_id': segment.segment_id,
                'session_id': segment.session_id,
                'emotion': segment.emotion,
                'duration': segment.duration,
                'feeling_it': segment.feeling_it,
                'feeling_it_ratio': segment.feeling_it_ratio
            }
            
            for name, value in zip(feature_names, feature_vector):
                features_dict[name] = value
            
            features_list.append(features_dict)
            
        except Exception as e:
            print(f"\n⚠ Error processing segment {segment.segment_id}: {e}")
            continue
    
    print(f"\n  ✓ Extracted features from {len(features_list)} segments")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    return segments, features_df


def main():
    """Process all session files"""
    
    print("="*80)
    print("BATCH PROCESSING: ALL SESSIONS")
    print("="*80)
    
    # Configuration
    data_dir = Path('data/raw')
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = sorted(data_dir.glob('*.csv'))
    
    if len(csv_files) == 0:
        print(f"❌ No CSV files found in {data_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Initialize processors
    slicer = DataSlicer(sampling_rate=100)
    extractor = EmotionFeatureExtractor(sampling_rate=100)
    
    # Optional filtering
    emotion_filter = {
        'exclude': ['nul']  # Exclude neutral/baseline
        # Or use 'include': ['joy', 'sad', 'fear', 'calm']
    }
    
    feeling_it_filter = False  # Set to True to only use feeling_it data
    
    # Process each file
    all_segments = []
    all_features = []
    
    for csv_path in csv_files:
        session_id = csv_path.stem  # Filename without extension
        
        try:
            segments, features_df = process_session(
                csv_path=csv_path,
                session_id=session_id,
                slicer=slicer,
                extractor=extractor,
                emotion_filter=emotion_filter,
                feeling_it_filter=feeling_it_filter
            )
            
            all_segments.extend(segments)
            all_features.append(features_df)
            
        except Exception as e:
            print(f"\n❌ Failed to process {session_id}: {e}")
            continue
    
    # Combine all features
    print("\n" + "="*80)
    print("COMBINING RESULTS")
    print("="*80)
    
    if len(all_features) == 0:
        print("❌ No features extracted from any session")
        sys.exit(1)
    
    combined_features = pd.concat(all_features, ignore_index=True)
    
    print(f"\nTotal segments: {len(all_segments)}")
    print(f"Total feature rows: {len(combined_features)}")
    print(f"\nEmotion distribution:")
    print(combined_features['emotion'].value_counts())
    
    # Save outputs
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # Save combined segments
    segments_path = output_dir / 'all_segments.pkl'
    with open(segments_path, 'wb') as f:
        pickle.dump(all_segments, f)
    print(f"✓ Saved segments: {segments_path}")
    
    # Save combined features
    features_path = output_dir / 'all_features.csv'
    combined_features.to_csv(features_path, index=False)
    print(f"✓ Saved features: {features_path} ({len(combined_features):,} rows)")
    
    # Save summary
    summary = {
        'total_samples': sum(len(pd.read_csv(f)) for f in csv_files),
        'total_segments': len(all_segments),
        'total_features': len(combined_features),
        'emotions': combined_features['emotion'].value_counts().to_dict(),
        'sessions': len(csv_files)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / 'processing_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary: {summary_path}")
    
    print("\n" + "="*80)
    print("✓ BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"\nNext step: Run feature analysis")
    print(f"  python scripts/analyze_features.py")


if __name__ == "__main__":
    main()
