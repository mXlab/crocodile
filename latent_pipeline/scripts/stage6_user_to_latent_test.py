#!/usr/bin/env python3
"""Stage 6: offline user-to-latent pipeline test -- a new subject's pre-recorded
biodata to generated faces.

This is the first end-to-end test of the full user-to-latent chain described
in PIPELINE.md:

    [raw user biodata] -> [feature extraction] -> [user-actress alignment]
        -> [actress-features-to-W regressor] -> StyleGAN2 -> face

"Offline" here is deliberate, not incidental: this reads a pre-recorded
subject CSV and processes it as a batch, same as every other latent_pipeline
stage. It is NOT the live "runtime pipeline" from PIPELINE.md (continuously
incoming sensor data, causal/online feature extraction, per-sample alignment
+ regression + render in a loop) -- that remains unbuilt. What this proves is
that the four offline pieces (extraction, alignment, regressor, StyleGAN2)
compose correctly end-to-end on a subject other than the actress; wiring the
same chain to live data is a separate, not-yet-started piece of work.

Unlike Stage 5's visual check (which pairs a real actress frame against its
regressor-predicted reconstruction), there is no ground-truth face for a new
subject -- only her own emotion labels for context. This renders generated
faces only, grouped by emotion, sampled evenly across the session's timeline
per emotion for variety.

Prerequisites (produced outside this script):
  1. biodata_pipeline/scripts/extract_continuous_features_batch.py --input <subject raw csv>
  2. biodata_pipeline/scripts/train_transformer.py (reference=actress, subject=new subject)
  3. biodata_pipeline/scripts/apply_transformer.py -> aligned features CSV
  4. latent_pipeline/scripts/stage5_train_regressor.py -> regressor.joblib

Usage:
    python latent_pipeline/scripts/stage6_user_to_latent_test.py \
        --config latent_pipeline/configs/default.yaml \
        --aligned-features biodata_pipeline/data/processed/erin_features_aligned.csv \
        --subject-name erin
"""

import argparse
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from models.stylegan import load_stylegan, generate


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def select_diverse_indices(df, emotion_col, n_per_emotion):
    """Evenly-spaced samples per emotion across the session timeline, so a
    single emotion's temporal variation (not just one frozen moment) shows
    in the grid."""
    indices = []
    for emotion in sorted(df[emotion_col].unique()):
        rows = df.index[df[emotion_col] == emotion].to_numpy()
        picks = np.linspace(0, len(rows) - 1, min(n_per_emotion, len(rows))).round().astype(int)
        indices.extend(rows[i] for i in sorted(set(picks)))
    return indices


@torch.no_grad()
def save_user_to_latent_grid(G, df, emotion_col, w_pred, device, output_path, indices, ncols=4):
    """Grid of generated-only faces (no ground truth exists for a new
    subject), each labeled with the subject's own emotion label."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(indices)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.8))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    for idx, i in enumerate(indices):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        wp = torch.tensor(w_pred[i:i + 1], dtype=torch.float32, device=device)
        img = generate(G, wp)
        img = F.interpolate(img, size=256, mode='bilinear', align_corners=False)
        arr = ((img[0].cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
        ax.imshow(arr)
        ax.axis('off')
        ax.set_title(df.iloc[i][emotion_col], fontsize=10, fontweight='bold')

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path} ({n} frames)")


def main():
    parser = argparse.ArgumentParser(description='Stage 6: offline user-to-latent pipeline test on a new subject')
    parser.add_argument('--config', default='latent_pipeline/configs/default.yaml')
    parser.add_argument('--aligned-features', required=True,
                        help='CSV of subject features already passed through the '
                             'user-actress alignment transformer (apply_transformer.py output)')
    parser.add_argument('--regressor', default=None,
                        help='Path to regressor.joblib (default: from config checkpoint_dir)')
    parser.add_argument('--subject-name', default='subject', help='Used in output filename')
    parser.add_argument('--n-per-emotion', type=int, default=4,
                        help='Frames to render per distinct emotion label')
    parser.add_argument('--ncols', type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = config['paths']['repo_root']
    rc = config['biodata_regressor']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    regressor_path = args.regressor or os.path.join(
        repo_root, rc['checkpoint_dir'], 'regressor.joblib')
    reg_data = joblib.load(regressor_path)
    model, scaler = reg_data['model'], reg_data['scaler']
    feature_cols, w_cols = reg_data['feature_cols'], reg_data['w_cols']
    print(f"Loaded {reg_data['model_type']} regressor from {regressor_path} "
          f"({len(feature_cols)} features -> {len(w_cols)} W dims)")

    aligned_path = args.aligned_features if os.path.isabs(args.aligned_features) \
        else os.path.join(repo_root, args.aligned_features)
    df = pd.read_csv(aligned_path)
    print(f"Loaded {len(df)} aligned rows from {aligned_path}")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Aligned features missing regressor's expected columns: {missing}")

    emotion_col = 'emotion' if 'emotion' in df.columns else 'emotion_label'
    print(f"Emotions present: {sorted(df[emotion_col].unique())}")

    X = df[feature_cols].values
    n_before = len(df)
    valid = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
    df, X = df[valid].reset_index(drop=True), X[valid]
    print(f"Dropped {n_before - len(df)} rows with NaN/inf features -> {len(df)} usable rows")

    X_scaled = scaler.transform(X)
    w_pred = model.predict(X_scaled)
    print(f"Predicted W vectors: {w_pred.shape}")

    print("\nLoading StyleGAN2 and rendering...")
    G = load_stylegan(config, device)

    indices = select_diverse_indices(df, emotion_col, args.n_per_emotion)
    output_dir = os.path.join(repo_root, rc['checkpoint_dir'])
    output_path = os.path.join(output_dir, f'user_to_latent_test_{args.subject_name}.png')
    save_user_to_latent_grid(G, df, emotion_col, w_pred, device, output_path, indices, ncols=args.ncols)


if __name__ == '__main__':
    main()
