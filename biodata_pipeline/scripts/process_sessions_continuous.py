"""
Process Sessions with Continuous Feature Extraction (Option 3)

NEW ARCHITECTURE:
    Raw Data → Features (continuous) → Slice into windows

This script inverts the traditional approach:
- OLD: Slice data → Extract features per segment
- NEW: Extract features continuously → Slice by emotion labels

Advantages:
1. Filter states maintained across emotion boundaries
2. Captures transition dynamics between emotions
3. Matches real-time deployment (filters never reset)
4. More efficient (compute once, slice many ways)

Usage:
    python scripts/process_sessions_continuous.py --window-size 30 --stride 2
"""

import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import argparse
from typing import Dict, List, Optional

from modules.data_slicer import slice_features_into_windows
from modules.continuous_feature_extractor import EnhancedContinuousFeatureExtractor as ContinuousFeatureExtractor


def process_session_continuous(session_path: Path,
                              sampling_rate: int = 100,
                              feature_interval_s: float = 1.0,
                              signal_cols: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Process one session: Raw data → Continuous features
    
    Parameters
    ----------
    session_path : Path
        Path to raw CSV file
    sampling_rate : int
        Sampling rate in Hz
    feature_interval_s : float
        Extract features every N seconds (1.0 = every second)
    signal_cols : dict, optional
        Signal column mapping
    
    Returns
    -------
    pd.DataFrame
        Continuous features with timestamps and emotion labels
    """
    
    if signal_cols is None:
        signal_cols = {
            'eda': 'gsr',
            'ppg': 'heart',
            'resp': 'respiration'
        }
    
    print(f"\n{'='*80}")
    print(f"PROCESSING SESSION: {session_path.stem}")
    print(f"{'='*80}")
    
    # Load raw data
    print(f"\nLoading raw data...")
    session_df = pd.read_csv(session_path)
    print(f"  ✓ Loaded {len(session_df):,} samples ({len(session_df)/sampling_rate:.1f}s)")
    
    # Check required columns
    required_cols = list(signal_cols.values()) + ['emotion']
    missing_cols = [col for col in required_cols if col not in session_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Extract features continuously
    print(f"\nExtracting features continuously (every {feature_interval_s}s)...")
    extractor = ContinuousFeatureExtractor(sampling_rate=sampling_rate)
    
    features_df = extractor.process_session(
        session_df,
        feature_interval_s=feature_interval_s,
        signal_cols=signal_cols
    )
    
    print(f"  ✓ Extracted {len(features_df)} feature samples")
    print(f"  ✓ Feature columns: {len([c for c in features_df.columns if '.' in c])}")
    
    # Add session metadata
    features_df['session_id'] = session_path.stem
    
    return features_df


def slice_session_features_into_windows(features_df: pd.DataFrame,
                                       window_size_s: float = 30,
                                       stride_s: float = 5,
                                       min_purity: float = 0.7,
                                       create_super_segments: bool = True,
                                       super_segment_size_s: float = 35) -> List[Dict]:
    """
    Slice continuous features into training windows.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Continuous features from process_session_continuous()
    window_size_s : float
        Window size for feature aggregation
    stride_s : float
        Stride between windows (overlap control)
    min_purity : float
        Minimum emotion purity per window
    create_super_segments : bool
        If True, group windows into independent super-segments for CV
    super_segment_size_s : float
        Size of super-segments (for grouping in CV)
    
    Returns
    -------
    list of dict
        Windows with aggregated features
    """
    
    print(f"\n{'='*80}")
    print(f"SLICING FEATURES INTO WINDOWS")
    print(f"{'='*80}")
    
    print(f"\nConfiguration:")
    print(f"  Window size: {window_size_s}s")
    print(f"  Stride: {stride_s}s (overlap: {(1 - stride_s/window_size_s)*100:.1f}%)")
    print(f"  Min emotion purity: {min_purity*100:.0f}%")
    
    if create_super_segments:
        print(f"  Super-segment size: {super_segment_size_s}s")
    
    windows = []
    
    # Get session ID
    session_id = features_df['session_id'].iloc[0]
    
    if create_super_segments:
        # Create independent super-segments first, then windows within each
        max_time = features_df['timestamp'].max()
        super_segment_id = 0
        
        # Split timeline into super-segments
        for super_start in np.arange(0, max_time, super_segment_size_s):
            super_end = super_start + super_segment_size_s
            
            # Get features in this super-segment
            super_mask = ((features_df['timestamp'] >= super_start) & 
                         (features_df['timestamp'] < super_end))
            super_features = features_df[super_mask]
            
            if len(super_features) < 10:  # Too short
                continue
            
            # Check emotion purity of super-segment
            emotion_counts = super_features['emotion'].value_counts()
            if len(emotion_counts) > 0:
                dominant_emotion = emotion_counts.index[0]
                super_purity = emotion_counts.iloc[0] / len(super_features)
                
                if super_purity < min_purity:
                    continue  # Skip mixed super-segments
                
                # Create windows within this super-segment
                super_segment_windows = _create_windows_in_range(
                    super_features,
                    start_time=super_start,
                    end_time=super_end,
                    window_size_s=window_size_s,
                    stride_s=stride_s,
                    parent_id=f"{session_id}_superseg{super_segment_id:03d}",
                    dominant_emotion=dominant_emotion
                )
                
                windows.extend(super_segment_windows)
                super_segment_id += 1
    else:
        # Create windows across entire session (no super-segments)
        max_time = features_df['timestamp'].max()
        
        windows = _create_windows_in_range(
            features_df,
            start_time=0,
            end_time=max_time,
            window_size_s=window_size_s,
            stride_s=stride_s,
            parent_id=session_id,
            dominant_emotion=None  # Will determine per window
        )
    
    print(f"\n  ✓ Created {len(windows)} windows")
    
    if create_super_segments:
        n_super_segments = len(set(w['parent_segment_id'] for w in windows))
        print(f"  ✓ Organized into {n_super_segments} super-segments")
    
    return windows


def _create_windows_in_range(features_df: pd.DataFrame,
                             start_time: float,
                             end_time: float,
                             window_size_s: float,
                             stride_s: float,
                             parent_id: str,
                             dominant_emotion: Optional[str] = None) -> List[Dict]:
    """Helper: Create windows within a time range."""
    
    windows = []
    window_counter = 0
    
    for win_start in np.arange(start_time, end_time - window_size_s, stride_s):
        win_end = win_start + window_size_s
        
        # Get features in this window
        mask = ((features_df['timestamp'] >= win_start) & 
               (features_df['timestamp'] < win_end))
        window_features = features_df[mask]
        
        if len(window_features) < 2:
            continue
        
        # Determine emotion (use provided or compute)
        if dominant_emotion is None:
            emotion_counts = window_features['emotion'].value_counts()
            window_emotion = emotion_counts.index[0]
            emotion_purity = emotion_counts.iloc[0] / len(window_features)
        else:
            window_emotion = dominant_emotion
            emotion_purity = 1.0  # Assume pure within super-segment
        
        # Aggregate features (mean over window)
        feature_cols = [c for c in window_features.columns 
                       if '.' in c and c not in ['session_id']]
        
        window_dict = {
            'window_id': f"{parent_id}_win{window_counter:03d}",
            'parent_segment_id': parent_id,
            'session_id': features_df['session_id'].iloc[0],
            'emotion': window_emotion,
            'duration': window_size_s,
            'emotion_purity': emotion_purity,
            'start_time': win_start,
            'end_time': win_end,
        }
        
        # Aggregate features
        for col in feature_cols:
            window_dict[col] = window_features[col].mean()
        
        # Add feeling_it if available
        if 'feeling_it' in window_features.columns:
            window_dict['feeling_it'] = window_features['feeling_it'].mean() > 0.5
            window_dict['feeling_it_ratio'] = window_features['feeling_it'].mean()
        
        window_dict['window_mode'] = 'continuous_hybrid'
        
        windows.append(window_dict)
        window_counter += 1
    
    return windows


def main():
    """Main processing workflow with continuous feature extraction."""
    
    parser = argparse.ArgumentParser(
        description='Process sessions with continuous feature extraction (Option 3)'
    )
    parser.add_argument(
        '--window-size',
        type=float,
        default=30,
        help='Window size in seconds (default: 30)'
    )
    parser.add_argument(
        '--stride',
        type=float,
        default=2,
        help='Stride in seconds (default: 2 for dense overlap)'
    )
    parser.add_argument(
        '--feature-interval',
        type=float,
        default=1.0,
        help='Extract features every N seconds (default: 1.0)'
    )
    parser.add_argument(
        '--min-purity',
        type=float,
        default=0.5,
        help='Minimum emotion purity (default: 0.5)'
    )
    parser.add_argument(
        '--super-segment-size',
        type=float,
        default=35,
        help='Super-segment size for CV grouping (default: 35s)'
    )
    parser.add_argument(
        '--no-super-segments',
        action='store_true',
        help='Disable super-segment grouping'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='continuous',
        help='Suffix for output files (default: continuous)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("CONTINUOUS FEATURE EXTRACTION PIPELINE (OPTION 3)")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Feature extraction: Every {args.feature_interval}s")
    print(f"  Window size: {args.window_size}s")
    print(f"  Stride: {args.stride}s (overlap: {(1-args.stride/args.window_size)*100:.1f}%)")
    print(f"  Min purity: {args.min_purity*100:.0f}%")
    print(f"  Super-segments: {'Enabled' if not args.no_super_segments else 'Disabled'}")
    if not args.no_super_segments:
        print(f"  Super-segment size: {args.super_segment_size}s")
    
    # Setup paths
    data_dir = PROJECT_ROOT / 'data' / 'raw'
    output_dir = PROJECT_ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CSV files
    csv_files = sorted(data_dir.glob('*.csv'))
    
    if len(csv_files) == 0:
        print(f"\n❌ No CSV files found in {data_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(csv_files)} session files")
    
    # ========================================================================
    # STEP 1: EXTRACT FEATURES CONTINUOUSLY (per session)
    # ========================================================================
    
    all_session_features = []
    
    for session_path in csv_files:
        try:
            features_df = process_session_continuous(
                session_path,
                sampling_rate=100,
                feature_interval_s=args.feature_interval,
                signal_cols={'eda': 'gsr', 'ppg': 'heart', 'resp': 'respiration'}
            )
            
            all_session_features.append(features_df)
            
        except Exception as e:
            print(f"\n❌ Error processing {session_path.stem}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_session_features) == 0:
        print("\n❌ No sessions processed successfully")
        sys.exit(1)
    
    # Combine all session features
    combined_features = pd.concat(all_session_features, ignore_index=True)
    
    print(f"\n{'='*80}")
    print(f"COMBINED CONTINUOUS FEATURES")
    print(f"{'='*80}")
    print(f"\nTotal feature samples: {len(combined_features):,}")
    print(f"Total sessions: {combined_features['session_id'].nunique()}")
    print(f"Total duration: {combined_features['timestamp'].max():.1f}s")
    
    # Save continuous features (optional, for inspection)
    continuous_path = output_dir / f'continuous_features_{args.output_suffix}.csv'
    combined_features.to_csv(continuous_path, index=False)
    print(f"\n✓ Saved continuous features: {continuous_path}")
    
    # ========================================================================
    # STEP 2: SLICE FEATURES INTO WINDOWS
    # ========================================================================
    
    all_windows = []
    
    for session_id in combined_features['session_id'].unique():
        session_features = combined_features[combined_features['session_id'] == session_id]
        
        windows = slice_session_features_into_windows(
            session_features,
            window_size_s=args.window_size,
            stride_s=args.stride,
            min_purity=args.min_purity,
            create_super_segments=not args.no_super_segments,
            super_segment_size_s=args.super_segment_size
        )
        
        all_windows.extend(windows)
    
    # ========================================================================
    # STEP 3: COMBINE AND SAVE
    # ========================================================================
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    
    if len(all_windows) == 0:
        print("\n❌ No windows created")
        sys.exit(1)
    
    # Convert to DataFrame
    windows_df = pd.DataFrame(all_windows)
    
    print(f"\nTotal windows: {len(windows_df)}")
    print(f"Unique groups (for CV): {windows_df['parent_segment_id'].nunique()}")
    print(f"Windows per group: {len(windows_df) / windows_df['parent_segment_id'].nunique():.1f}")
    
    print(f"\nEmotion distribution:")
    emotion_counts = windows_df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion:15s}: {count:4d} windows")
    
    print(f"\nGroups per emotion:")
    for emotion in emotion_counts.index:
        n_groups = windows_df[windows_df['emotion'] == emotion]['parent_segment_id'].nunique()
        print(f"  {emotion:15s}: {n_groups:3d} groups")
    
    # Save windows
    windows_path = output_dir / f'all_features_{args.output_suffix}.csv'
    windows_df.to_csv(windows_path, index=False)
    print(f"\n✓ Saved windows: {windows_path}")
    
    print(f"\n{'='*80}")
    print(f"✓ PROCESSING COMPLETE")
    print(f"{'='*80}")
    
    print(f"\nGenerated {len(windows_df):,} training samples")
    print(f"  From {windows_df['parent_segment_id'].nunique()} independent groups")
    print(f"  Using CONTINUOUS feature extraction (filters never reset!)")
    
    print(f"\nNext steps:")
    print(f"  1. Filter to desired emotions:")
    print(f"     python scripts/filter_emotions.py --emotions fea sad")
    print(f"  2. Evaluate with GroupKFold:")
    print(f"     python scripts/evaluate_with_cv.py --features {windows_path}")
    print(f"\nKey advantage: Features maintain continuity across emotion boundaries! 🚀")


if __name__ == "__main__":
    main()
