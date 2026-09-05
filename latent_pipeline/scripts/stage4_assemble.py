#!/usr/bin/env python3
"""Stage 4: Assemble biodata→W dataset.

1. Load trained encoder (best checkpoint)
2. Run inference on all biodata_mapping_manifest.csv frames
3. Look up corresponding continuous_features.csv rows
4. Output: biodata_w_dataset.csv (metadata + 512 W dims + 73 feature dims)

Usage:
    python latent_pipeline/scripts/stage4_assemble.py --config latent_pipeline/configs/default.yaml
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from models.encoder import EmotionEncoder


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_image(frame_path, repo_root):
    """Load and preprocess a single frame."""
    full_path = os.path.join(repo_root, frame_path)
    img = cv2.imread(full_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
    return img


def main():
    parser = argparse.ArgumentParser(description='Stage 4: Assemble biodata-W dataset')
    parser.add_argument('--config', default='latent_pipeline/configs/default.yaml')
    parser.add_argument('--checkpoint', default=None,
                        help='Encoder checkpoint (default: outputs/best.pt)')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = config['paths']['repo_root']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = os.path.join(repo_root, config['train_frames']['checkpoint_dir'])

    # Load encoder
    ec = config['encoder']
    encoder = EmotionEncoder(
        channels=tuple(ec['channels']),
        w_dim=ec['w_dim'],
        dropout=ec['dropout'],
    ).to(device)

    ckpt_path = args.checkpoint or os.path.join(output_dir, 'best.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()
    print(f"Loaded encoder from {ckpt_path} (epoch {ckpt['epoch']})")

    # Load biodata mapping manifest
    metadata_dir = os.path.join(repo_root, config['paths']['metadata_dir'])
    manifest_path = os.path.join(metadata_dir, 'biodata_mapping_manifest.csv')
    manifest = pd.read_csv(manifest_path)
    print(f"Biodata manifest: {len(manifest)} frames")

    # Load continuous features
    features_path = os.path.join(repo_root, config['paths']['continuous_features'])
    features_df = pd.read_csv(features_path)
    # Identify feature columns (exclude metadata columns)
    metadata_cols = {'timestamp', 'sample_idx', 'emotion', 'feeling_it', 'session_id'}
    feature_cols = [c for c in features_df.columns if c not in metadata_cols]
    print(f"Feature columns: {len(feature_cols)}")

    # Run inference in batches
    all_w = []
    all_indices = []
    batch_images = []
    batch_indices = []

    for i in tqdm(range(len(manifest)), desc="Encoding frames"):
        row = manifest.iloc[i]
        img = load_image(row['frame_path'], repo_root)
        if img is None:
            print(f"  WARNING: Could not load {row['frame_path']}")
            continue

        batch_images.append(img)
        batch_indices.append(i)

        if len(batch_images) == args.batch_size or i == len(manifest) - 1:
            images = torch.stack(batch_images).to(device)
            with torch.no_grad():
                w = encoder(images)
            all_w.append(w.cpu())
            all_indices.extend(batch_indices)
            batch_images = []
            batch_indices = []

    w_all = torch.cat(all_w, dim=0).numpy()  # (N, 512)
    print(f"Encoded {w_all.shape[0]} frames")

    # Assemble dataset
    rows = []
    for j, manifest_idx in enumerate(all_indices):
        mrow = manifest.iloc[manifest_idx]
        feature_idx = int(mrow['continuous_feature_idx'])

        if feature_idx >= len(features_df):
            print(f"  WARNING: feature_idx {feature_idx} out of range")
            continue

        frow = features_df.iloc[feature_idx]

        # Metadata
        entry = {
            'frame_path': mrow['frame_path'],
            'pool_name': mrow['pool_name'],
            'frame_number': mrow['frame_number'],
            'timestamp_s': mrow['timestamp_s'],
            'biodata_timestamp_s': mrow['biodata_timestamp_s'],
            'emotion_label': mrow['emotion_label'],
            'raw_emotion': mrow.get('raw_emotion'),
            'feeling_it': mrow['feeling_it'],
        }

        # W vector dimensions
        for d in range(w_all.shape[1]):
            entry[f'w_{d:03d}'] = w_all[j, d]

        # Feature dimensions
        for col in feature_cols:
            entry[col] = frow[col]

        rows.append(entry)

    result_df = pd.DataFrame(rows)

    # Validate
    n_nan = result_df.isna().sum().sum()
    print(f"\nAssembled dataset: {len(result_df)} rows, {len(result_df.columns)} columns")
    print(f"NaN values: {n_nan}")

    if n_nan > 0:
        nan_cols = result_df.columns[result_df.isna().any()].tolist()
        print(f"Columns with NaN: {nan_cols}")

    # W vector stats per session
    w_cols = [f'w_{d:03d}' for d in range(w_all.shape[1])]
    for pool_name in result_df['pool_name'].unique():
        pool_df = result_df[result_df['pool_name'] == pool_name]
        w_norms = np.linalg.norm(pool_df[w_cols].values, axis=1)
        print(f"  {pool_name}: {len(pool_df)} frames, "
              f"W norm: {w_norms.mean():.2f} +/- {w_norms.std():.2f}")

    # Emotion label consistency check
    for pool_name in result_df['pool_name'].unique():
        pool_df = result_df[result_df['pool_name'] == pool_name]
        unique_emotions = pool_df['emotion_label'].unique()
        print(f"  {pool_name} emotions: {sorted(unique_emotions)}")

    # Save
    output_path = os.path.join(repo_root, 'latent_pipeline', 'data', 'biodata_w_dataset.csv')
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  Metadata columns: 8")
    print(f"  W dimensions: {len(w_cols)}")
    print(f"  Feature dimensions: {len(feature_cols)}")
    print(f"  Total columns: {len(result_df.columns)}")


if __name__ == '__main__':
    main()
