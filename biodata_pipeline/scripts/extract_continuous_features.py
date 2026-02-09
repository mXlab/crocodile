"""
Step 1: Extract Continuous Features

This script ONLY extracts features continuously from raw sensor data.
It does NOT create windows or filter emotions.

Output: CSV file with continuous features (one row per time interval)

Usage:
    python scripts/extract_continuous_features.py
    python scripts/extract_continuous_features.py --feature-interval 1.0 --output features_1s.csv
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import argparse
from typing import Dict, Optional

from modules.continuous_feature_extractor import EnhancedContinuousFeatureExtractor as FeatureExtractor
FEATURE_COUNT = 75


def extract_features_from_session(session_path: Path,
                                  sampling_rate: int = 100,
                                  feature_interval_s: float = 1.0,
                                  signal_cols: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Extract continuous features from one session.
    
    Parameters
    ----------
    session_path : Path
        Path to raw CSV file
    sampling_rate : int
        Sampling rate in Hz
    feature_interval_s : float
        Extract features every N seconds
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
    
    print(f"\n{'─'*80}")
    print(f"Processing: {session_path.stem}")
    print(f"{'─'*80}")
    
    # Load raw data
    session_df = pd.read_csv(session_path)
    print(f"  Loaded: {len(session_df):,} samples ({len(session_df)/sampling_rate:.1f}s)")
    
    # Check required columns
    required_cols = list(signal_cols.values()) + ['emotion']
    missing_cols = [col for col in required_cols if col not in session_df.columns]
    if missing_cols:
        print(f"  ❌ Missing columns: {missing_cols}")
        return None
    
    # Extract features continuously
    print(f"  Extracting features (every {feature_interval_s}s)...")
    extractor = FeatureExtractor(sampling_rate=sampling_rate)
    
    features_df = extractor.process_session(
        session_df,
        feature_interval_s=feature_interval_s,
        signal_cols=signal_cols
    )
    
    print(f"  ✓ Extracted {len(features_df)} feature samples")
    
    # Add session metadata
    features_df['session_id'] = session_path.stem
    
    return features_df


def main():
    """Extract continuous features from all sessions."""
    
    parser = argparse.ArgumentParser(
        description='Step 1: Extract continuous features from raw sensor data'
    )
    parser.add_argument(
        '--feature-interval',
        type=float,
        default=1.0,
        help='Extract features every N seconds (default: 1.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='continuous_features.csv',
        help='Output filename (default: continuous_features.csv)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing raw CSV files (default: data/raw)'
    )
    parser.add_argument(
        '--per-session',
        action='store_true',
        help='Save separate CSV per session instead of combined'
    )
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        help='File patterns to process, relative to --data-dir (default: *.csv). '
             'Examples: --input erin_labeled.csv, --input "laurence_*.csv"'
    )

    args = parser.parse_args()

    print("="*80)
    print("STEP 1: CONTINUOUS FEATURE EXTRACTION")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  Feature interval: {args.feature_interval}s")
    print(f"  Expected features: ~{FEATURE_COUNT}")
    print(f"  Output mode: {'Per-session' if args.per_session else 'Combined'}")

    # Setup paths
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find CSV files
    if args.input:
        csv_files = []
        for pattern in args.input:
            matches = sorted(data_dir.glob(pattern))
            if not matches:
                print(f"  ⚠ No files matched: {pattern}")
            csv_files.extend(matches)
        csv_files = sorted(set(csv_files))
    else:
        csv_files = sorted(data_dir.glob('*.csv'))
    
    if len(csv_files) == 0:
        print(f"\n❌ No CSV files found in {data_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(csv_files)} session files:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # ========================================================================
    # EXTRACT FEATURES FROM EACH SESSION
    # ========================================================================
    
    all_features = []
    
    for session_path in csv_files:
        try:
            features_df = extract_features_from_session(
                session_path,
                sampling_rate=100,
                feature_interval_s=args.feature_interval,
                signal_cols={'eda': 'gsr', 'ppg': 'heart', 'resp': 'respiration'}
            )
            
            if features_df is not None:
                # Save per-session if requested
                if args.per_session:
                    session_output = output_dir / f"{session_path.stem}_features.csv"
                    features_df.to_csv(session_output, index=False)
                    print(f"  ✓ Saved: {session_output.name}")
                
                all_features.append(features_df)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_features) == 0:
        print("\n❌ No features extracted")
        sys.exit(1)
    
    # ========================================================================
    # COMBINE AND SAVE
    # ========================================================================
    
    combined_features = pd.concat(all_features, ignore_index=True)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nTotal feature samples: {len(combined_features):,}")
    print(f"Sessions: {combined_features['session_id'].nunique()}")
    print(f"Total duration: {combined_features['timestamp'].max():.1f}s")
    
    # Feature columns
    feature_cols = [c for c in combined_features.columns if '.' in c]
    print(f"Features extracted: {len(feature_cols)}")
    
    # Emotion distribution
    print(f"\nEmotion distribution:")
    emotion_counts = combined_features['emotion'].value_counts()
    for emotion, count in emotion_counts.head(10).items():
        duration = count * args.feature_interval
        print(f"  {emotion:15s}: {count:4d} samples ({duration:.1f}s)")
    
    if len(emotion_counts) > 10:
        print(f"  ... and {len(emotion_counts) - 10} more emotions")
    
    # Save combined
    output_path = output_dir / args.output
    combined_features.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved combined features: {output_path}")
    print(f"  Shape: {combined_features.shape}")
    print(f"  Columns: {list(combined_features.columns[:5])} ... (showing first 5)")
    
    print("\n" + "="*80)
    print("✓ FEATURE EXTRACTION COMPLETE")
    print("="*80)
    
    print(f"\nNext step:")
    print(f"  python scripts/slice_and_evaluate.py \\")
    print(f"      --features {output_path.relative_to(PROJECT_ROOT)} \\")
    print(f"      --include fea sad \\")
    print(f"      --window-size 30 \\")
    print(f"      --stride 2")


if __name__ == "__main__":
    main()
