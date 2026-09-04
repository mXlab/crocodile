"""
Step 2: Slice and Evaluate Continuous Features

Takes continuous features and:
1. Filters emotions (--include or --exclude)
2. Creates windows (aggregates features over time)
3. Optionally creates super-segments for CV grouping
4. Evaluates with cross-validation

Usage:
    # Evaluate fea vs sad with 30s windows
    python scripts/slice_and_evaluate.py \\
        --features data/processed/continuous_features.csv \\
        --include fea sad \\
        --window-size 30 --stride 2
    
    # Evaluate all emotions except 'nul'
    python scripts/slice_and_evaluate.py \\
        --features data/processed/continuous_features.csv \\
        --exclude nul \\
        --window-size 20 --stride 5 \\
        --super-segment-size 35
    
    # Just create dataset, don't evaluate yet
    python scripts/slice_and_evaluate.py \\
        --features data/processed/continuous_features.csv \\
        --include fea sad joy \\
        --window-size 30 --stride 2 \\
        --output data/processed/windowed_fea_sad_joy.csv \\
        --no-evaluate
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import argparse
from typing import List, Optional
import subprocess

# Import config system
try:
    from modules.pipeline_config import PipelineConfig, load_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠ Warning: Config system not available (modules/pipeline_config.py not found)")


def filter_emotions(features_df: pd.DataFrame,
                   include: Optional[List[str]] = None,
                   exclude: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filter continuous features by emotion.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Continuous features
    include : list of str, optional
        Include only these emotions
    exclude : list of str, optional
        Exclude these emotions
    
    Returns
    -------
    pd.DataFrame
        Filtered features
    """
    
    original_count = len(features_df)
    original_emotions = features_df['emotion'].unique()
    
    print(f"\n{'='*80}")
    print(f"EMOTION FILTERING")
    print(f"{'='*80}")
    
    print(f"\nOriginal: {len(original_emotions)} emotions, {original_count:,} samples")
    
    if include:
        print(f"\n➜ Including only: {', '.join(include)}")
        features_df = features_df[features_df['emotion'].isin(include)]
        
        missing = set(include) - set(features_df['emotion'].unique())
        if missing:
            print(f"  ⚠ Warning: Emotions not found: {', '.join(missing)}")
    
    elif exclude:
        print(f"\n➜ Excluding: {', '.join(exclude)}")
        features_df = features_df[~features_df['emotion'].isin(exclude)]
    
    else:
        print(f"\n➜ No filtering (using all emotions)")
    
    filtered_count = len(features_df)
    filtered_emotions = features_df['emotion'].unique()
    
    print(f"\nAfter filtering: {len(filtered_emotions)} emotions, {filtered_count:,} samples")
    print(f"  Removed: {original_count - filtered_count:,} samples ({(original_count - filtered_count)/original_count*100:.1f}%)")
    
    print(f"\nEmotion distribution:")
    emotion_counts = features_df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion:15s}: {count:4d} samples")
    
    return features_df


def normalize_per_session(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each feature independently within each session.

    Removes session-level baseline differences so the classifier sees
    emotion-relative deviations rather than absolute physiological values.
    Computed on the full session (before emotion filtering) so the mean/std
    reflect the complete recording baseline.
    """

    print(f"\n{'='*80}")
    print(f"PER-SESSION NORMALISATION")
    print(f"{'='*80}")

    metadata_cols = {'timestamp', 'sample_idx', 'emotion', 'session_id', 'feeling_it'}
    feature_cols = [c for c in features_df.columns if c not in metadata_cols]

    normalized = features_df.copy()

    for session_id in features_df['session_id'].unique():
        mask = features_df['session_id'] == session_id
        session_data = features_df.loc[mask, feature_cols]

        mean = session_data.mean()
        std = session_data.std().replace(0, 1)  # avoid division by zero for constant features

        normalized.loc[mask, feature_cols] = (session_data - mean) / std

    print(f"\n  Sessions normalised: {features_df['session_id'].nunique()}")
    print(f"  Features normalised: {len(feature_cols)}")
    print(f"  Each session now has mean≈0, std≈1 per feature")

    return normalized


def create_windows(features_df: pd.DataFrame,
                  window_size_s: float = 30,
                  stride_s: float = 5,
                  min_purity: float = 0.7,
                  super_segment_size_s: Optional[float] = None) -> pd.DataFrame:
    """
    Create windows from continuous features.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Continuous features (filtered by emotion)
    window_size_s : float
        Window size in seconds
    stride_s : float
        Stride between windows
    min_purity : float
        Minimum emotion purity (fraction of window with same emotion)
    super_segment_size_s : float, optional
        If provided, group windows into super-segments for CV
    
    Returns
    -------
    pd.DataFrame
        Windowed features ready for CV
    """
    
    print(f"\n{'='*80}")
    print(f"WINDOWING")
    print(f"{'='*80}")
    
    print(f"\nConfiguration:")
    print(f"  Window size: {window_size_s}s")
    print(f"  Stride: {stride_s}s (overlap: {(1 - stride_s/window_size_s)*100:.1f}%)")
    print(f"  Min purity: {min_purity*100:.0f}%")
    
    if super_segment_size_s:
        print(f"  Super-segments: {super_segment_size_s}s (for CV grouping)")
    else:
        print(f"  Super-segments: Disabled")
    
    windows = []
    
    # Process each session
    for session_id in features_df['session_id'].unique():
        session_features = features_df[features_df['session_id'] == session_id]
        
        print(f"\n  Session: {session_id}")
        print(f"    Samples: {len(session_features)}")
        
        if super_segment_size_s:
            session_windows = _create_super_segment_windows(
                session_features,
                session_id,
                window_size_s,
                stride_s,
                min_purity,
                super_segment_size_s
            )
        else:
            session_windows = _create_simple_windows(
                session_features,
                session_id,
                window_size_s,
                stride_s,
                min_purity
            )
        
        windows.extend(session_windows)
        print(f"    Windows: {len(session_windows)}")
    
    if len(windows) == 0:
        print(f"\n❌ No windows created!")
        return None
    
    windows_df = pd.DataFrame(windows)
    
    print(f"\n{'='*80}")
    print(f"WINDOWING COMPLETE")
    print(f"{'='*80}")
    
    print(f"\nTotal windows: {len(windows_df)}")
    print(f"Groups (for CV): {windows_df['parent_segment_id'].nunique()}")
    print(f"Windows per group: {len(windows_df) / windows_df['parent_segment_id'].nunique():.1f}")
    
    print(f"\nWindows per emotion:")
    for emotion in windows_df['emotion'].unique():
        emo_windows = windows_df[windows_df['emotion'] == emotion]
        n_groups = emo_windows['parent_segment_id'].nunique()
        print(f"  {emotion:15s}: {len(emo_windows):3d} windows, {n_groups:3d} groups")
    
    return windows_df


def _create_super_segment_windows(features_df: pd.DataFrame,
                                 session_id: str,
                                 window_size_s: float,
                                 stride_s: float,
                                 min_purity: float,
                                 super_segment_size_s: float) -> List[dict]:
    """Create windows grouped by super-segments."""
    
    windows = []
    max_time = features_df['timestamp'].max()
    super_seg_id = 0
    
    for super_start in np.arange(0, max_time, super_segment_size_s):
        super_end = super_start + super_segment_size_s
        
        mask = ((features_df['timestamp'] >= super_start) & 
               (features_df['timestamp'] < super_end))
        super_features = features_df[mask]
        
        if len(super_features) < 10:
            continue
        
        # Check purity
        emotion_counts = super_features['emotion'].value_counts()
        if len(emotion_counts) == 0:
            continue
        
        dominant_emotion = emotion_counts.index[0]
        purity = emotion_counts.iloc[0] / len(super_features)
        
        if purity < min_purity:
            continue
        
        # Create windows within super-segment
        parent_id = f"{session_id}_ss{super_seg_id:03d}"
        
        for win_start in np.arange(super_start, super_end - window_size_s, stride_s):
            win_end = win_start + window_size_s
            
            mask = ((super_features['timestamp'] >= win_start) & 
                   (super_features['timestamp'] < win_end))
            win_features = super_features[mask]
            
            if len(win_features) < 2:
                continue
            
            window = _aggregate_window(win_features, parent_id, session_id, 
                                      dominant_emotion, win_start, win_end)
            windows.append(window)
        
        super_seg_id += 1
    
    return windows


def _create_simple_windows(features_df: pd.DataFrame,
                          session_id: str,
                          window_size_s: float,
                          stride_s: float,
                          min_purity: float) -> List[dict]:
    """Create windows without super-segment grouping."""
    
    windows = []
    max_time = features_df['timestamp'].max()
    win_id = 0
    
    for win_start in np.arange(0, max_time - window_size_s, stride_s):
        win_end = win_start + window_size_s
        
        mask = ((features_df['timestamp'] >= win_start) & 
               (features_df['timestamp'] < win_end))
        win_features = features_df[mask]
        
        if len(win_features) < 2:
            continue
        
        # Check purity
        emotion_counts = win_features['emotion'].value_counts()
        dominant_emotion = emotion_counts.index[0]
        purity = emotion_counts.iloc[0] / len(win_features)
        
        if purity < min_purity:
            continue
        
        # Each window is its own group
        parent_id = f"{session_id}_w{win_id:04d}"
        
        window = _aggregate_window(win_features, parent_id, session_id,
                                  dominant_emotion, win_start, win_end)
        windows.append(window)
        win_id += 1
    
    return windows


def _aggregate_window(features: pd.DataFrame, 
                     parent_id: str,
                     session_id: str,
                     emotion: str,
                     start_time: float,
                     end_time: float) -> dict:
    """Aggregate features over a window (mean)."""
    
    metadata_cols = ['timestamp', 'sample_idx', 'emotion', 'session_id', 'feeling_it']
    feature_cols = [c for c in features.columns if c not in metadata_cols]
    
    window = {
        'window_id': f"{parent_id}_t{int(start_time)}",
        'parent_segment_id': parent_id,
        'session_id': session_id,
        'emotion': emotion,
        'start_time': start_time,
        'end_time': end_time,
        'duration': end_time - start_time,
        'emotion_purity': 1.0,  # Already filtered by purity
    }
    
    # Mean of features
    for col in feature_cols:
        window[col] = features[col].mean()
    
    if 'feeling_it' in features.columns:
        window['feeling_it_ratio'] = features['feeling_it'].mean()
    
    window['window_mode'] = 'continuous'
    
    return window


def main():
    """Slice and evaluate continuous features."""
    
    parser = argparse.ArgumentParser(
        description='Step 2: Slice continuous features and evaluate',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Config file (highest priority after command-line)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file (e.g., configs/best_2emo.yaml)'
    )
    
    # Input
    parser.add_argument(
        '--features',
        type=str,
        required=False,  # Not required if using --config
        default=None,
        help='Path to continuous features CSV (from step 1)'
    )
    
    # Emotion filtering
    emotion_group = parser.add_mutually_exclusive_group()
    emotion_group.add_argument(
        '--include',
        type=str,
        nargs='+',
        help='Include only these emotions (e.g., --include fea sad)'
    )
    emotion_group.add_argument(
        '--exclude',
        type=str,
        nargs='+',
        help='Exclude these emotions (e.g., --exclude nul)'
    )
    
    # Windowing
    parser.add_argument(
        '--window-size',
        type=float,
        default=30,
        help='Window size in seconds (default: 30)'
    )
    parser.add_argument(
        '--stride',
        type=float,
        default=5,
        help='Stride in seconds (default: 5)'
    )
    parser.add_argument(
        '--min-purity',
        type=float,
        default=0.7,
        help='Minimum emotion purity (default: 0.7)'
    )
    parser.add_argument(
        '--super-segment-size',
        type=float,
        default=None,
        help='Super-segment size for CV grouping (optional)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save windowed features (optional, auto-generated if not specified)'
    )
    parser.add_argument(
        '--no-evaluate',
        action='store_true',
        help='Skip evaluation (only create dataset)'
    )
    parser.add_argument(
        '--group-by',
        type=str,
        default=None,
        help='Column to use for CV grouping. Use session_id for honest '
             'leave-one-session-out CV, or parent_segment_id for super-segment '
             'grouping (default).'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Z-score each feature per session before windowing, to remove '
             'session-level baseline differences.'
    )
    
    args = parser.parse_args()
    
    # Validate: must have either --config or --features
    if not args.config and not args.features:
        parser.error("Either --config or --features must be provided")
    
    print("="*80)
    print("STEP 2: SLICE AND EVALUATE")
    print("="*80)
    
    # ========================================================================
    # LOAD CONFIGURATION
    # ========================================================================
    
    if CONFIG_AVAILABLE and args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"\n❌ Config file not found: {config_path}")
            sys.exit(1)
        
        config = load_config(config_path, args)
        
        if not config.validate():
            sys.exit(1)
        
        print(f"\n{config.summary()}")
        
        # Use config values
        features_path = Path(config['features_file'])
        include_emotions = config['include_emotions']
        exclude_emotions = config['exclude_emotions']
        window_size_s = config['window_size_s']
        stride_s = config['stride_s']
        min_purity = config['min_purity']
        super_segment_size_s = config['super_segment_size_s']
        output_path_str = config['output_file']
        no_evaluate = not config['evaluate']
        
    else:
        # Use command-line args directly
        if args.config:
            print("\n⚠ Config file specified but pipeline_config.py not available")
            print("  Using command-line arguments instead")
        
        features_path = Path(args.features)
        include_emotions = args.include
        exclude_emotions = args.exclude
        window_size_s = args.window_size
        stride_s = args.stride
        min_purity = args.min_purity
        super_segment_size_s = args.super_segment_size
        output_path_str = args.output
        no_evaluate = args.no_evaluate
    
    # ========================================================================
    # LOAD CONTINUOUS FEATURES
    # ========================================================================
    
    if not features_path.exists():
        print(f"\n❌ File not found: {features_path}")
        sys.exit(1)
    
    print(f"\nLoading: {features_path.name}")
    features_df = pd.read_csv(features_path)
    
    feature_cols = [c for c in features_df.columns if '.' in c]
    print(f"  Samples: {len(features_df):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Sessions: {features_df['session_id'].nunique()}")
    print(f"  Emotions: {features_df['emotion'].nunique()}")

    # ========================================================================
    # PER-SESSION NORMALISATION (optional)
    # ========================================================================

    if args.normalize:
        features_df = normalize_per_session(features_df)

    # ========================================================================
    # FILTER EMOTIONS
    # ========================================================================
    
    features_df = filter_emotions(features_df, include_emotions, exclude_emotions)
    
    if len(features_df) == 0:
        print("\n❌ No data after filtering!")
        sys.exit(1)
    
    # ========================================================================
    # CREATE WINDOWS
    # ========================================================================
    
    windows_df = create_windows(
        features_df,
        window_size_s=window_size_s,
        stride_s=stride_s,
        min_purity=min_purity,
        super_segment_size_s=super_segment_size_s
    )
    
    if windows_df is None or len(windows_df) == 0:
        print("\n❌ No windows created!")
        sys.exit(1)
    
    # ========================================================================
    # SAVE WINDOWED FEATURES
    # ========================================================================
    
    if output_path_str:
        output_path = Path(output_path_str)
    else:
        # Auto-generate filename
        if include_emotions:
            emo_str = '_'.join(include_emotions[:3])  # Max 3 emotions in name
        elif exclude_emotions:
            emo_str = f"excl_{'_'.join(exclude_emotions[:2])}"
        else:
            emo_str = "all"
        
        output_name = f"windowed_{emo_str}_w{int(window_size_s)}s{int(stride_s)}.csv"
        output_path = PROJECT_ROOT / 'data' / 'processed' / output_name
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Shape: {windows_df.shape}")
    
    # ========================================================================
    # EVALUATE WITH CROSS-VALIDATION
    # ========================================================================
    
    if not no_evaluate:
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION")
        print(f"{'='*80}")
        
        eval_script = SCRIPT_DIR / 'evaluate_with_cv.py'
        
        if eval_script.exists():
            cmd = [sys.executable, str(eval_script), '--features', str(output_path)]
            if args.group_by:
                cmd += ['--group-by', args.group_by]

            print(f"\nRunning: {' '.join(cmd)}\n")
            subprocess.run(cmd)
        else:
            print(f"\n⚠ Evaluation script not found: {eval_script}")
            print(f"\nManually evaluate with:")
            print(f"  python scripts/evaluate_with_cv.py --features {output_path}")
    else:
        print(f"\n✓ Dataset created (evaluation skipped)")
        print(f"\nTo evaluate:")
        print(f"  python scripts/evaluate_with_cv.py --features {output_path}")


if __name__ == "__main__":
    main()
