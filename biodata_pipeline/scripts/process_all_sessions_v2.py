"""
Process all sessions with flexible windowing support

This script supports multiple windowing modes via configuration or command-line.

Usage:
    # Use default (training_standard)
    python scripts/process_all_sessions_v2.py
    
    # Use specific preset
    python scripts/process_all_sessions_v2.py --preset training_dense
    
    # Custom parameters
    python scripts/process_all_sessions_v2.py --mode hybrid --window-size 30 --stride 5
    
    # Segment mode (original behavior)
    python scripts/process_all_sessions_v2.py --preset segment_mode
"""

import sys
from pathlib import Path
import argparse
import yaml

# Fix Python path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pickle
from modules.data_slicer import DataSlicer
from modules.feature_extractor import EmotionFeatureExtractor


def load_windowing_config(config_path='configs/windowing_config.yaml'):
    """Load windowing configuration from YAML file."""
    config_file = PROJECT_ROOT / config_path
    
    if not config_file.exists():
        print(f"⚠ Config file not found: {config_file}")
        print("Using default configuration")
        return {}
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def process_session(csv_path, session_id, slicer, extractor,
                   windowing_config,
                   emotion_filter=None,
                   feeling_it_filter=False):
    """Process a single session file with flexible windowing."""
    
    print(f"\n{'='*80}")
    print(f"PROCESSING: {session_id}")
    print(f"{'='*80}")
    
    # Load data
    print(f"Loading {csv_path}...")
    data = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(data):,} samples ({len(data)/100:.1f}s)")
    
    # Create windows using configuration
    print(f"\nCreating windows...")
    print(f"  Mode: {windowing_config.get('mode', 'segment')}")
    if 'window_size_s' in windowing_config:
        print(f"  Window size: {windowing_config['window_size_s']}s")
        print(f"  Stride: {windowing_config['stride_s']}s")
        print(f"  Overlap: {windowing_config.get('overlap_ratio', 0)*100:.1f}%")
    
    windows = slicer.create_windows(
        data,
        session_id=session_id,
        window_mode=windowing_config.get('mode', 'segment'),
        window_size_s=windowing_config.get('window_size_s', 30),
        stride_s=windowing_config.get('stride_s', 5),
        min_purity=windowing_config.get('min_purity', 0.5),
        emotion_col='emotion',
        feeling_col='feeling_it',
        signal_cols=['heart', 'gsr', 'respiration']
    )
    
    # Apply filters
    if emotion_filter:
        if 'exclude' in emotion_filter:
            windows = slicer.filter_by_emotions(
                windows,
                exclude_emotions=emotion_filter['exclude']
            )
        elif 'include' in emotion_filter:
            windows = slicer.filter_by_emotions(
                windows,
                include_emotions=emotion_filter['include']
            )
    
    if feeling_it_filter:
        windows = slicer.filter_by_feeling_it(
            windows,
            require_feeling_it=True,
            min_feeling_ratio=0.5,
            time_tolerance_s=2.0
        )
    
    # Quality filter (relaxed)
    windows = slicer.filter_by_quality(
        windows,
        min_duration_s=5.0,
        check_signal_validity=False,  # Disabled to avoid filtering too many
        max_flat_ratio=0.9
    )
    
    # Print summary
    slicer.print_summary(windows)
    
    if len(windows) == 0:
        print("⚠ No windows after filtering, skipping feature extraction")
        return [], pd.DataFrame()
    
    # Extract features
    print(f"\nExtracting features from {len(windows)} windows...")
    features_list = []
    
    for i, window in enumerate(windows, 1):
        if i % 50 == 0 or i == len(windows):
            print(f"  Processing window {i}/{len(windows)}...", end='\r')
        
        try:
            features = extractor.extract_all_features(
                eda_raw=window.signals['gsr'],
                ppg_raw=window.signals['heart'],
                resp_raw=window.signals['respiration'],
                window_size_s=window.duration
            )
            
            feature_vector, feature_names = extractor.features_to_vector(features)
            
            # Build feature dict with metadata
            features_dict = {
                'window_id': window.window_id,
                'session_id': window.session_id,
                'emotion': window.emotion,
                'duration': window.duration,
                'emotion_purity': window.emotion_purity,
                'feeling_it': window.feeling_it,
                'feeling_it_ratio': window.feeling_it_ratio,
                'window_mode': window.metadata.get('window_mode', 'unknown')
            }
            
            # Add parent segment ID if available (for hybrid mode)
            if window.parent_segment_id:
                features_dict['parent_segment_id'] = window.parent_segment_id
            
            for name, value in zip(feature_names, feature_vector):
                features_dict[name] = value
            
            features_list.append(features_dict)
            
        except Exception as e:
            print(f"\n⚠ Error processing window {window.window_id}: {e}")
            continue
    
    print(f"\n  ✓ Extracted features from {len(features_list)} windows")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    return windows, features_df


def main():
    """Main processing workflow with flexible windowing."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process sessions with flexible windowing'
    )
    parser.add_argument(
        '--preset',
        type=str,
        default='training_standard',
        help='Windowing preset from windowing_config.yaml (default: training_standard)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['segment', 'sliding', 'hybrid'],
        help='Windowing mode (overrides preset)'
    )
    parser.add_argument(
        '--window-size',
        type=float,
        help='Window size in seconds (overrides preset)'
    )
    parser.add_argument(
        '--stride',
        type=float,
        help='Stride in seconds (overrides preset)'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='',
        help='Suffix for output files (e.g., "_hybrid" → all_features_hybrid.csv)'
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
    
    print("="*80)
    print("BATCH PROCESSING WITH FLEXIBLE WINDOWING")
    print("="*80)
    
    # Load windowing configuration
    windowing_configs = load_windowing_config()
    
    # Select or build configuration
    if args.mode:
        # Manual configuration
        windowing_config = {
            'mode': args.mode,
            'window_size_s': args.window_size or 30,
            'stride_s': args.stride or 5
        }
        print(f"\nUsing manual configuration:")
    elif args.preset in windowing_configs:
        # Preset configuration
        windowing_config = windowing_configs[args.preset]
        print(f"\nUsing preset: {args.preset}")
        print(f"Description: {windowing_config.get('description', 'N/A')}")
    else:
        # Default
        print(f"\n⚠ Preset '{args.preset}' not found, using training_standard")
        windowing_config = windowing_configs.get('training_standard', {
            'mode': 'hybrid',
            'window_size_s': 30,
            'stride_s': 5
        })
    
    # Print configuration
    print(f"\nWindowing configuration:")
    for key, value in windowing_config.items():
        if key not in ['description', 'use_case', 'pros', 'cons', 'expected_samples']:
            print(f"  {key}: {value}")
    
    # Setup paths
    data_dir = PROJECT_ROOT / 'data' / 'raw'
    output_dir = PROJECT_ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CSV files
    csv_files = sorted(data_dir.glob('*.csv'))
    
    if len(csv_files) == 0:
        print(f"\n❌ No CSV files found in {data_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    # Initialize processors
    slicer = DataSlicer(sampling_rate=100)
    extractor = EmotionFeatureExtractor(sampling_rate=100)
    
    # Emotion filter (from CLI) — handle both comma- and space-separated labels
    def parse_labels(raw_list):
        labels = []
        for item in raw_list:
            labels.extend(item.split(','))
        return [l.strip() for l in labels if l.strip()]

    if args.include_emotions:
        labels = parse_labels(args.include_emotions)
        emotion_filter = {'include': labels}
        print(f"\nEmotion filter: include only {labels}")
    elif args.exclude_emotions:
        labels = parse_labels(args.exclude_emotions)
        emotion_filter = {'exclude': labels}
        print(f"\nEmotion filter: exclude {labels}")
    else:
        emotion_filter = None
    
    feeling_it_filter = False
    
    # Process each file
    all_windows = []
    all_features = []
    
    for csv_path in csv_files:
        session_id = csv_path.stem
        
        try:
            windows, features_df = process_session(
                csv_path=csv_path,
                session_id=session_id,
                slicer=slicer,
                extractor=extractor,
                windowing_config=windowing_config,
                emotion_filter=emotion_filter,
                feeling_it_filter=feeling_it_filter
            )
            
            all_windows.extend(windows)
            if len(features_df) > 0:
                all_features.append(features_df)
            
        except Exception as e:
            print(f"\n❌ Failed to process {session_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine results
    print("\n" + "="*80)
    print("COMBINING RESULTS")
    print("="*80)
    
    if len(all_features) == 0:
        print("❌ No features extracted from any session")
        sys.exit(1)
    
    combined_features = pd.concat(all_features, ignore_index=True)
    
    print(f"\nTotal windows: {len(all_windows)}")
    print(f"Total feature rows: {len(combined_features)}")
    
    # Windowing info
    if len(all_windows) > 0:
        info = slicer.get_windowing_info(all_windows)
        print(f"\nWindowing info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print(f"\nEmotion distribution:")
    emotion_counts = combined_features['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion:15s}: {count:4d} windows")
    
    # Save outputs
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # Determine output filenames
    suffix = args.output_suffix or f"_{windowing_config['mode']}"
    if suffix and not suffix.startswith('_'):
        suffix = '_' + suffix
    
    # Save windows
    windows_path = output_dir / f'all_windows{suffix}.pkl'
    with open(windows_path, 'wb') as f:
        pickle.dump(all_windows, f)
    print(f"✓ Saved windows: {windows_path}")
    
    # Save features
    features_path = output_dir / f'all_features{suffix}.csv'
    combined_features.to_csv(features_path, index=False)
    print(f"✓ Saved features: {features_path} ({len(combined_features):,} rows)")
    
    # Save configuration used
    config_path = output_dir / f'windowing_config{suffix}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(windowing_config, f)
    print(f"✓ Saved config: {config_path}")
    
    # Save summary
    summary = {
        'windowing_mode': windowing_config['mode'],
        'window_size_s': windowing_config.get('window_size_s'),
        'stride_s': windowing_config.get('stride_s'),
        'total_samples': sum(len(pd.read_csv(f)) for f in csv_files),
        'total_windows': len(all_windows),
        'total_features': len(combined_features),
        'sessions': len(csv_files),
        'emotions': emotion_counts.to_dict()
    }
    
    summary_path = output_dir / f'processing_summary{suffix}.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f)
    print(f"✓ Saved summary: {summary_path}")
    
    print("\n" + "="*80)
    print("✓ BATCH PROCESSING COMPLETE")
    print("="*80)
    
    print(f"\nGenerated {len(combined_features):,} training samples")
    print(f"  (vs {len(all_windows)} windows before feature extraction)")
    
    print(f"\nNext steps:")
    print(f"  1. Run feature analysis:")
    print(f"     python scripts/analyze_features.py --features data/processed/all_features{suffix}.csv")
    print(f"  2. Train classifier with this data (Module 5)")


if __name__ == "__main__":
    main()
