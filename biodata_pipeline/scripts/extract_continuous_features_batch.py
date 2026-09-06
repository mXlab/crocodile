"""
Step 1 (batch variant): Extract Continuous Features via NeuroKit2

Offline/non-causal counterpart to extract_continuous_features.py -- see
modules/batch_feature_extractor.py's module docstring for the full
rationale (cold-start artifacts, EDA raw-signal normalization gap).

This does NOT overwrite continuous_features.csv. Output goes to a
separate file (default continuous_features_batch.csv) so both the
causal/streaming-compatible extractor and this batch one remain available
to compare directly -- switching between them for Stage 4/5 is just a
config path change (paths.continuous_features in latent_pipeline's yaml).

Usage:
    python scripts/extract_continuous_features_batch.py
    python scripts/extract_continuous_features_batch.py --output continuous_features_batch.csv
"""

import argparse
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from modules.batch_feature_extractor import BatchFeatureExtractor


def extract_features_from_session(session_path, sampling_rate=100, feature_interval_s=1.0,
                                   signal_cols=None):
    if signal_cols is None:
        signal_cols = {'eda': 'gsr', 'ppg': 'heart', 'resp': 'respiration'}

    print(f"\n{'-'*80}")
    print(f"Processing: {session_path.stem}")
    print(f"{'-'*80}")

    session_df = pd.read_csv(session_path)
    print(f"  Loaded: {len(session_df):,} samples ({len(session_df)/sampling_rate:.1f}s)")

    required_cols = list(signal_cols.values()) + ['emotion']
    missing_cols = [c for c in required_cols if c not in session_df.columns]
    if missing_cols:
        print(f"  Missing columns: {missing_cols}")
        return None

    extractor = BatchFeatureExtractor(sampling_rate=sampling_rate)
    features_df = extractor.process_session(session_df, feature_interval_s=feature_interval_s,
                                             signal_cols=signal_cols)
    print(f"  Extracted {len(features_df)} feature rows, "
          f"{len([c for c in features_df.columns if '.' in c])} features")

    features_df['session_id'] = session_path.stem
    return features_df


def main():
    parser = argparse.ArgumentParser(description='Step 1 (batch): NeuroKit2-based feature extraction')
    parser.add_argument('--feature-interval', type=float, default=1.0)
    parser.add_argument('--output', type=str, default='continuous_features_batch.csv')
    parser.add_argument('--data-dir', type=str, default='data/raw')
    parser.add_argument('--input', type=str, nargs='+',
                        help='File patterns relative to --data-dir (default: laurence sessions only)')
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 1 (BATCH): NEUROKIT2-BASED FEATURE EXTRACTION")
    print("=" * 80)

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        csv_files = []
        for pattern in args.input:
            csv_files.extend(sorted(data_dir.glob(pattern)))
        csv_files = sorted(set(csv_files))
    else:
        # Default to the 4 real sessions used throughout the pipeline, not
        # every raw CSV in the directory (e.g. skip erin_*, source/*).
        csv_files = sorted(data_dir.glob('emotion_biodata_laurence_main_*.csv'))

    if not csv_files:
        print(f"\nNo CSV files found in {data_dir}")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} session files:")
    for f in csv_files:
        print(f"  - {f.name}")

    all_features = []
    for session_path in csv_files:
        try:
            features_df = extract_features_from_session(
                session_path, sampling_rate=100, feature_interval_s=args.feature_interval,
            )
            if features_df is not None:
                all_features.append(features_df)
        except Exception as e:
            print(f"  Error: {e}")
            traceback.print_exc()
            continue

    if not all_features:
        print("\nNo features extracted")
        sys.exit(1)

    combined = pd.concat(all_features, ignore_index=True)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total feature samples: {len(combined):,}")
    print(f"Sessions: {combined['session_id'].nunique()}")
    feature_cols = [c for c in combined.columns if '.' in c]
    print(f"Features extracted: {len(feature_cols)}")
    print(f"NaN values: {combined[feature_cols].isna().sum().sum()}")

    output_path = output_dir / args.output
    combined.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
