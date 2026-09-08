"""
Train Ridge regression to align a new subject's biodata features
to the reference subject's emotional feature space.

Algorithm:
1. Load reference and subject feature CSVs
2. Find emotions present in both datasets
3. Compute per-emotion mean feature vectors (prototypes)
4. Train Ridge regression: subject prototypes → reference prototypes
5. Save transformer model

The transformer classes themselves live in modules/alignment_transformer.py
(reusable by apply_transformer.py, validate_transformer.py,
validate_heldout_emotion.py, and live_pipeline/live_pipeline.py) -- this
script is just the training CLI around them.

Usage (from biodata_pipeline/):
    python scripts/train_transformer.py --reference data/processed/continuous_features.csv --subject data/processed/erin_features.csv
    python scripts/train_transformer.py --reference data/processed/continuous_features.csv --subject data/processed/new_subject.csv --alpha 5.0
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.alignment_transformer import create_transformer


def main():
    parser = argparse.ArgumentParser(
        description="Train subject-to-reference prototype alignment transformer"
    )
    parser.add_argument(
        '--reference', required=True,
        help='Reference subject features CSV'
    )
    parser.add_argument(
        '--subject', required=True,
        help='New subject features CSV'
    )
    parser.add_argument(
        '--method',
        choices=['ridge', 'ot_global', 'ot_classconditional', 'coral', 'zscore'],
        default='ridge',
        help='Alignment method: ridge (default), ot_global, ot_classconditional, coral, '
             'zscore (diagonal-only mean+variance, no covariance -- no emotion labels '
             'needed on --subject, robust from very little data; see live_pipeline.py)'
    )
    parser.add_argument(
        '--emotion', default=None,
        help='zscore only: restrict --reference rows to this single emotion label '
             '(e.g. neu) instead of pooling all of --reference. Ignored by other methods.'
    )
    parser.add_argument(
        '--alpha', type=float, default=10.0,
        help='Ridge regularization strength (default: 10.0, Ridge only)'
    )
    parser.add_argument(
        '--reg', type=float, default=1e-5,
        help='OT/CORAL regularization strength (default: 1e-5, OT and CORAL methods only)'
    )
    parser.add_argument(
        '--n-features', type=int, default=None,
        help='Keep only top N features by ANOVA F-test (default: use all, Ridge only)'
    )
    parser.add_argument(
        '--output',
        default='models/subject_alignment_transformer.pkl',
        help='Output path for trained transformer'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Subject-to-Reference Prototype Alignment")
    print(f"Method: {args.method}")
    print("=" * 60)

    ref_df = pd.read_csv(args.reference)
    sub_df = pd.read_csv(args.subject)
    print(f"Reference: {len(ref_df)} samples from {args.reference}")
    print(f"Subject:   {len(sub_df)} samples from {args.subject}")

    transformer = create_transformer(
        args.method, alpha=args.alpha, n_features=args.n_features, reg=args.reg)

    if args.method == 'zscore':
        transformer.fit(ref_df, sub_df, emotion=args.emotion)
    else:
        transformer.fit(ref_df, sub_df)
    transformer.save(args.output)

    print("\nDone.")


if __name__ == '__main__':
    main()
