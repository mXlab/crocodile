#!/usr/bin/env python3
"""Rebuild the training manifest by filtering out invalid (nul) frames.

Reads the existing cnn_training_manifest.csv, removes frames annotated as
'nul' (invalid: makeup, clapperboard, tech issues), and writes a new manifest.
Also deletes the invalid frame files from disk.

Usage:
    python latent_pipeline/scripts/rebuild_manifest.py [--delete-files]
"""

import argparse
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)

from data.annotations import get_emotion, is_invalid

MANIFEST = os.path.join(REPO_ROOT, 'latent_pipeline/data/metadata/cnn_training_manifest.csv')
BIODATA_MANIFEST = os.path.join(REPO_ROOT, 'latent_pipeline/data/metadata/biodata_mapping_manifest.csv')


def main():
    parser = argparse.ArgumentParser(description='Rebuild manifest without invalid frames')
    parser.add_argument('--delete-files', action='store_true',
                        help='Also delete invalid frame PNG files from disk')
    args = parser.parse_args()

    df = pd.read_csv(MANIFEST)
    print(f"Original manifest: {len(df)} rows")

    # Identify invalid frames in session pools
    invalid_mask = pd.Series(False, index=df.index)
    for idx, row in df.iterrows():
        if row['pool_name'].startswith('session_'):
            session = row['pool_name'].replace('session_', '')
            frame_num = int(row['frame_number'])
            if is_invalid(session, frame_num):
                invalid_mask[idx] = True

    invalid_df = df[invalid_mask]
    valid_df = df[~invalid_mask]

    print(f"Invalid (nul) frames found: {len(invalid_df)}")
    print(f"Valid frames remaining: {len(valid_df)}")

    # Per-session breakdown
    for pool in sorted(invalid_df['pool_name'].unique()):
        count = (invalid_df['pool_name'] == pool).sum()
        print(f"  {pool}: {count} invalid removed")

    # Optionally delete invalid frame files
    if args.delete_files:
        deleted = 0
        for _, row in invalid_df.iterrows():
            path = os.path.join(REPO_ROOT, row['frame_path'])
            if os.path.exists(path):
                os.remove(path)
                deleted += 1
        print(f"\nDeleted {deleted} invalid frame files from disk")

    # Write filtered manifest
    valid_df.to_csv(MANIFEST, index=False)
    print(f"\nUpdated manifest written: {MANIFEST} ({len(valid_df)} rows)")

    # Also filter biodata manifest if it exists
    if os.path.exists(BIODATA_MANIFEST):
        bio_df = pd.read_csv(BIODATA_MANIFEST)
        bio_invalid = pd.Series(False, index=bio_df.index)
        for idx, row in bio_df.iterrows():
            if row['pool_name'].startswith('session_'):
                session = row['pool_name'].replace('session_', '')
                frame_num = int(row['frame_number'])
                if is_invalid(session, frame_num):
                    bio_invalid[idx] = True

        bio_valid = bio_df[~bio_invalid]
        bio_df.to_csv(BIODATA_MANIFEST + '.bak', index=False)  # backup
        bio_valid.to_csv(BIODATA_MANIFEST, index=False)
        print(f"Updated biodata manifest: {BIODATA_MANIFEST} ({len(bio_valid)} rows, was {len(bio_df)})")

    # Summary of new label distribution
    print(f"\n=== New emotion_label distribution ===")
    session_df = valid_df[valid_df['pool_name'].str.startswith('session_')]
    print(session_df['emotion_label'].value_counts().to_string())


if __name__ == '__main__':
    main()
