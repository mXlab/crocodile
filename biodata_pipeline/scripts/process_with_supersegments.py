"""
Hybrid Windowing Strategy: Independent Super-Segments with Overlapping Windows

Strategy:
1. Split emotion segments into independent 60s "super-segments"
2. Within each super-segment, create overlapping windows (30s with 5s stride)
3. Group by super-segment for cross-validation

Benefits:
- More independent groups for stable CV (vs original segments)
- More training samples (from overlapping within each group)
- No data leakage (super-segments are independent)

Usage:
    # Use default settings
    python scripts/process_with_supersegments.py

    # Use a windowing preset for window-size and stride
    python scripts/process_with_supersegments.py --preset training_dense

    # Custom super-segment size with preset
    python scripts/process_with_supersegments.py --preset training_standard --super-segment-size 90

    # Filter emotions
    python scripts/process_with_supersegments.py --include-emotions ang,joy,rlx
    python scripts/process_with_supersegments.py --exclude-emotions nul,war
"""

import sys
from pathlib import Path
import argparse
import yaml

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
from modules.data_slicer import DataSlicer
from modules.feature_extractor import EmotionFeatureExtractor


def load_windowing_config(config_path='configs/windowing_config.yaml'):
    """Load windowing configuration from YAML file."""
    config_file = PROJECT_ROOT / config_path

    if not config_file.exists():
        print(f"Warning: Config file not found: {config_file}")
        print("Using default configuration")
        return {}

    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def parse_labels(raw_list):
    """Parse comma- and/or space-separated label lists."""
    labels = []
    for item in raw_list:
        labels.extend(item.split(','))
    return [l.strip() for l in labels if l.strip()]


def create_super_segments(data, session_id, slicer, 
                         super_segment_size_s=60,
                         emotion_col='emotion',
                         feeling_col='feeling_it',
                         signal_cols=None):
    """
    Split data into independent super-segments.
    
    Instead of using natural emotion boundaries, we split into fixed-duration
    independent chunks. This creates more groups for cross-validation.
    
    Parameters
    ----------
    super_segment_size_s : float
        Duration of each super-segment in seconds (e.g., 60)
    
    Returns
    -------
    super_segments : list of dict
        Each dict contains: {'data': DataFrame, 'super_segment_id': str, 'emotion': str}
    """
    
    if signal_cols is None:
        signal_cols = ['heart', 'gsr', 'respiration']
    
    sampling_rate = slicer.sampling_rate
    super_segment_samples = int(super_segment_size_s * sampling_rate)
    
    super_segments = []
    
    # Split into fixed-size chunks
    for i in range(0, len(data), super_segment_samples):
        # Extract chunk
        chunk_data = data.iloc[i:i+super_segment_samples].copy()
        
        # Skip if too short
        if len(chunk_data) < super_segment_samples * 0.5:  # At least 50% of desired size
            continue
        
        # Determine dominant emotion (majority vote)
        emotion_counts = chunk_data[emotion_col].value_counts()
        dominant_emotion = emotion_counts.index[0]
        emotion_purity = emotion_counts.iloc[0] / len(chunk_data)
        
        # Skip if emotion purity is too low (mixed emotions)
        if emotion_purity < 0.7:  # At least 70% one emotion
            continue
        
        # Create super-segment ID
        super_segment_id = f"{session_id}_superseg{len(super_segments):03d}"
        
        super_segments.append({
            'data': chunk_data,
            'super_segment_id': super_segment_id,
            'emotion': dominant_emotion,
            'emotion_purity': emotion_purity,
            'start_time': i / sampling_rate,
            'duration': len(chunk_data) / sampling_rate
        })
    
    print(f"  Created {len(super_segments)} super-segments of ~{super_segment_size_s}s each")
    
    return super_segments


def create_windows_in_super_segment(super_segment, slicer,
                                   window_size_s=30,
                                   stride_s=5,
                                   signal_cols=None):
    """
    Create overlapping windows within a single super-segment.
    
    All windows from this super-segment will share the same group ID
    for cross-validation.
    """
    
    if signal_cols is None:
        signal_cols = ['heart', 'gsr', 'respiration']
    
    data = super_segment['data']
    super_segment_id = super_segment['super_segment_id']
    emotion = super_segment['emotion']
    
    sampling_rate = slicer.sampling_rate
    window_samples = int(window_size_s * sampling_rate)
    stride_samples = int(stride_s * sampling_rate)
    
    windows = []
    
    # Create sliding windows within this super-segment
    for start_idx in range(0, len(data) - window_samples + 1, stride_samples):
        end_idx = start_idx + window_samples
        
        window_data = data.iloc[start_idx:end_idx]
        
        # Extract signals
        signals = {col: window_data[col].values for col in signal_cols}
        
        # Feeling_it stats
        if 'feeling_it' in window_data.columns:
            feeling_it_ratio = window_data['feeling_it'].sum() / len(window_data)
        else:
            feeling_it_ratio = 0.0
        
        # Create window object
        from modules.data_slicer import EmotionWindow
        
        window = EmotionWindow(
            window_id=f"{super_segment_id}_win{len(windows):03d}",
            session_id=super_segment_id.split('_superseg')[0],
            emotion=emotion,
            emotion_purity=1.0,  # Within super-segment, emotion is pure
            start_idx=start_idx,
            end_idx=end_idx,
            start_time=super_segment['start_time'] + (start_idx / sampling_rate),
            end_time=super_segment['start_time'] + (end_idx / sampling_rate),
            duration=window_size_s,
            signals=signals,
            feeling_it=feeling_it_ratio > 0,
            feeling_it_ratio=feeling_it_ratio,
            parent_segment_id=super_segment_id,  # KEY: All windows from same super-segment
            metadata={
                'window_mode': 'hybrid_supersegment',
                'super_segment_size_s': super_segment['duration'],
                'stride_s': stride_s
            }
        )
        
        windows.append(window)
    
    return windows


def main():
    """Main processing workflow with super-segment strategy."""
    
    parser = argparse.ArgumentParser(
        description='Process with independent super-segments + overlapping windows'
    )
    parser.add_argument(
        '--preset',
        type=str,
        default=None,
        help='Windowing preset from windowing_config.yaml (sets window-size and stride)'
    )
    parser.add_argument(
        '--super-segment-size',
        type=float,
        default=60,
        help='Super-segment duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--window-size',
        type=float,
        default=None,
        help='Window size in seconds (default: 30, overrides preset)'
    )
    parser.add_argument(
        '--stride',
        type=float,
        default=None,
        help='Stride in seconds (default: 5, overrides preset)'
    )
    parser.add_argument(
        '--min-purity',
        type=float,
        default=0.7,
        help='Minimum emotion purity for super-segments (default: 0.7)'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='',
        help='Suffix for output files (e.g., "_dense" -> all_features_supersegment_dense.csv)'
    )

    emotion_group = parser.add_mutually_exclusive_group()
    emotion_group.add_argument(
        '--exclude-emotions',
        type=str,
        nargs='+',
        metavar='LABEL',
        default=['nul'],
        help='Emotion labels to exclude (default: nul)'
    )
    emotion_group.add_argument(
        '--include-emotions',
        type=str,
        nargs='+',
        metavar='LABEL',
        help='Only include these emotion labels (excludes all others)'
    )

    args = parser.parse_args()

    # Resolve window-size and stride from preset or defaults
    window_size = args.window_size
    stride = args.stride

    if args.preset:
        windowing_configs = load_windowing_config()
        if args.preset in windowing_configs:
            preset = windowing_configs[args.preset]
            print(f"Using preset: {args.preset}")
            print(f"  Description: {preset.get('description', 'N/A')}")
            if window_size is None:
                window_size = preset.get('window_size_s', 30)
            if stride is None:
                stride = preset.get('stride_s', 5)
        else:
            print(f"Warning: Preset '{args.preset}' not found, using defaults")

    # Apply defaults if still unset
    if window_size is None:
        window_size = 30
    if stride is None:
        stride = 5

    # Resolve emotion filter
    if args.include_emotions:
        labels = parse_labels(args.include_emotions)
        emotion_filter = {'include': labels}
    elif args.exclude_emotions:
        labels = parse_labels(args.exclude_emotions)
        emotion_filter = {'exclude': labels}
    else:
        emotion_filter = None

    print("="*80)
    print("HYBRID WINDOWING: SUPER-SEGMENTS + OVERLAPPING WINDOWS")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  Super-segment size: {args.super_segment_size}s (independent chunks)")
    print(f"  Window size: {window_size}s")
    print(f"  Stride: {stride}s (overlap: {(1-stride/window_size)*100:.1f}%)")
    print(f"  Min emotion purity: {args.min_purity*100:.0f}%")
    if emotion_filter:
        mode = 'include' if 'include' in emotion_filter else 'exclude'
        print(f"  Emotion filter ({mode}): {emotion_filter[mode]}")
    
    # Setup paths
    data_dir = PROJECT_ROOT / 'data' / 'raw'
    output_dir = PROJECT_ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CSV files
    csv_files = sorted(data_dir.glob('*.csv'))
    
    if len(csv_files) == 0:
        print(f"\n❌ No CSV files found in {data_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(csv_files)} CSV files")
    
    # Initialize processors
    slicer = DataSlicer(sampling_rate=100)
    extractor = EmotionFeatureExtractor(sampling_rate=100)
    
    # Process each session
    all_windows = []
    all_features = []
    
    total_super_segments = 0
    
    for csv_path in csv_files:
        session_id = csv_path.stem
        
        print(f"\n{'='*80}")
        print(f"PROCESSING: {session_id}")
        print(f"{'='*80}")
        
        try:
            # Load data
            data = pd.read_csv(csv_path)
            print(f"Loaded {len(data):,} samples ({len(data)/100:.1f}s)")
            
            # Step 1: Create super-segments (independent chunks)
            super_segments = create_super_segments(
                data, session_id, slicer,
                super_segment_size_s=args.super_segment_size,
                emotion_col='emotion',
                feeling_col='feeling_it',
                signal_cols=['heart', 'gsr', 'respiration']
            )
            
            total_super_segments += len(super_segments)
            
            # Filter super-segments by emotion
            if emotion_filter:
                before_count = len(super_segments)
                if 'include' in emotion_filter:
                    super_segments = [s for s in super_segments
                                      if s['emotion'] in emotion_filter['include']]
                elif 'exclude' in emotion_filter:
                    super_segments = [s for s in super_segments
                                      if s['emotion'] not in emotion_filter['exclude']]
                print(f"  Emotion filter: kept {len(super_segments)}/{before_count} super-segments")

            # Step 2: Create overlapping windows within each super-segment
            session_windows = []

            for super_segment in super_segments:
                windows = create_windows_in_super_segment(
                    super_segment, slicer,
                    window_size_s=window_size,
                    stride_s=stride,
                    signal_cols=['heart', 'gsr', 'respiration']
                )
                session_windows.extend(windows)
            
            print(f"  Total windows: {len(session_windows)}")
            if len(super_segments) > 0:
                print(f"  Windows per super-segment: {len(session_windows)/len(super_segments):.1f}")
            
            # Step 3: Extract features from all windows
            print(f"\nExtracting features...")
            
            for i, window in enumerate(session_windows, 1):
                if i % 50 == 0 or i == len(session_windows):
                    print(f"  Processing window {i}/{len(session_windows)}...", end='\r')
                
                try:
                    features = extractor.extract_all_features(
                        eda_raw=window.signals['gsr'],
                        ppg_raw=window.signals['heart'],
                        resp_raw=window.signals['respiration'],
                        window_size_s=window.duration
                    )
                    
                    feature_vector, feature_names = extractor.features_to_vector(features)
                    
                    # Build feature dict
                    features_dict = {
                        'window_id': window.window_id,
                        'parent_segment_id': window.parent_segment_id,  # Super-segment ID
                        'session_id': window.session_id,
                        'emotion': window.emotion,
                        'duration': window.duration,
                        'emotion_purity': window.emotion_purity,
                        'feeling_it': window.feeling_it,
                        'feeling_it_ratio': window.feeling_it_ratio,
                        'window_mode': 'supersegment_hybrid'
                    }
                    
                    for name, value in zip(feature_names, feature_vector):
                        features_dict[name] = value
                    
                    all_features.append(features_dict)
                    
                except Exception as e:
                    print(f"\n⚠ Error processing window {window.window_id}: {e}")
                    continue
            
            all_windows.extend(session_windows)
            print(f"\n  ✓ Extracted features from {len(session_windows)} windows")
            
        except Exception as e:
            print(f"\n❌ Failed to process {session_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine and save results
    print("\n" + "="*80)
    print("COMBINING RESULTS")
    print("="*80)
    
    if len(all_features) == 0:
        print("❌ No features extracted")
        sys.exit(1)
    
    combined_features = pd.DataFrame(all_features)
    
    print(f"\nTotal super-segments: {total_super_segments}")
    print(f"Total windows: {len(all_windows)}")
    print(f"Total feature rows: {len(combined_features)}")
    print(f"Windows per super-segment: {len(all_windows)/total_super_segments:.1f}")
    
    # Get unique super-segments for CV
    n_groups = combined_features['parent_segment_id'].nunique()
    print(f"\nUnique groups for CV: {n_groups}")
    
    print(f"\nEmotion distribution:")
    emotion_counts = combined_features['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion:15s}: {count:4d} windows")
    
    print(f"\nSuper-segments per emotion:")
    for emotion in emotion_counts.index:
        n_super = combined_features[combined_features['emotion'] == emotion]['parent_segment_id'].nunique()
        print(f"  {emotion:15s}: {n_super:3d} super-segments")
    
    # Save outputs
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # Determine output suffix
    suffix = args.output_suffix
    if suffix and not suffix.startswith('_'):
        suffix = '_' + suffix

    # Save windows
    windows_path = output_dir / f'all_windows_supersegment{suffix}.pkl'
    with open(windows_path, 'wb') as f:
        pickle.dump(all_windows, f)
    print(f"✓ Saved windows: {windows_path}")

    # Save features
    features_path = output_dir / f'all_features_supersegment{suffix}.csv'
    combined_features.to_csv(features_path, index=False)
    print(f"✓ Saved features: {features_path}")
    
    print("\n" + "="*80)
    print("✓ PROCESSING COMPLETE")
    print("="*80)
    
    print(f"\nGenerated {len(combined_features):,} training samples")
    print(f"  From {n_groups} independent super-segments")
    print(f"  Average {len(combined_features)/n_groups:.1f} windows per super-segment")
    
    print(f"\nNext steps:")
    print(f"  1. Evaluate with GroupKFold:")
    print(f"     python scripts/train_and_evaluate.py --features data/processed/all_features_supersegment{suffix}.csv")
    print(f"\nGroupKFold will use {n_groups} groups for stable cross-validation")


if __name__ == "__main__":
    main()
