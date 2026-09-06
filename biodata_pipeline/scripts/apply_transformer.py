"""
Apply a trained subject alignment transformer to a subject's feature CSV,
producing an aligned CSV in the reference (actress) feature space.

This performs the `[user-actress alignment]` step of PIPELINE.md's
user-to-latent chain: a new subject's biodata features only make sense to the
actress-features-to-W regressor once mapped through this transform. This
script runs it offline/in batch, over a whole pre-recorded feature CSV --
not the live "runtime pipeline" (continuously incoming data), which would
apply the same `.transform()` call per incoming sample instead of writing a
CSV. Unlike validate_transformer.py (which checks separation/classification
metrics), this script materializes the transformed features to a CSV so they
can be fed directly into latent_pipeline's Stage 5 regressor.

Usage (from biodata_pipeline/):
    python scripts/apply_transformer.py \
        --subject data/processed/erin_features_batch.csv \
        --model models/transformer_ot_classconditional_batch.pkl \
        --output data/processed/erin_features_aligned.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_transformer import load_transformer, METADATA_COLS


def main():
    parser = argparse.ArgumentParser(
        description='Apply a trained alignment transformer to a subject feature CSV')
    parser.add_argument('--subject', required=True, help='Subject features CSV (untransformed)')
    parser.add_argument('--model', required=True, help='Trained transformer .pkl')
    parser.add_argument('--output', required=True, help='Output CSV path (aligned features)')
    args = parser.parse_args()

    subject_df = pd.read_csv(args.subject)
    transformer = load_transformer(args.model)
    print(f"Loaded {transformer.__class__.__name__} ({len(transformer.feature_cols)} features, "
          f"emotions: {transformer.common_emotions})")

    aligned = transformer.transform(subject_df[transformer.feature_cols])

    metadata_cols = [c for c in METADATA_COLS if c in subject_df.columns]
    out_df = subject_df[metadata_cols].copy()
    for i, col in enumerate(transformer.feature_cols):
        out_df[col] = aligned[:, i]

    n_before = len(out_df)
    n_nan = out_df[transformer.feature_cols].isna().sum().sum()
    print(f"Transformed {n_before} rows, {len(transformer.feature_cols)} features, NaN: {n_nan}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
