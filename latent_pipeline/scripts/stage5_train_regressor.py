#!/usr/bin/env python3
"""Stage 5: Train a biodata -> W-space regressor.

Baseline: Ridge regression on all 73 standardized biodata features -> the
512-dim W vector, trained on biodata_w_dataset.csv (Stage 4 output). This
mirrors biodata_pipeline/scripts/train_transformer.py's existing Ridge +
StandardScaler approach rather than introducing a new convention.

Train/val split holds out one full session (default: session_4S) so the
split respects temporal correlation within a session, matching Stage 2B's
convention -- a random row-level split would leak near-duplicate consecutive
frames between train and val.

This is deliberately the full-feature baseline, not a feature-selected model:
establish the ceiling first, then decide what to prune based on *this* task's
own feature importance (see PIPELINE.md discussion) rather than reusing the
classification-task ANOVA ranking from biodata_pipeline's feature_analyzer.

Usage:
    python latent_pipeline/scripts/stage5_train_regressor.py --config latent_pipeline/configs/default.yaml
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from models.stylegan import load_stylegan, generate

METADATA_COLS = {
    'frame_path', 'pool_name', 'frame_number',
    'timestamp_s', 'biodata_timestamp_s', 'emotion_label', 'raw_emotion', 'feeling_it',
}

# emotion_label collapses war (warmup) / tra (transition) / coo (cooldown) /
# neu (neutral) all into 'none' -- raw_emotion (added in this session) keeps
# them distinct. war/tra/coo are frames between emotional states where the
# biodata is plausibly still settling; excluding them is scoped to Stage 5's
# regression task only (Stage 2B's encoder training keeps them -- see
# audit_dataset_labels.py, a valid image is a valid image for reconstruction
# regardless of emotional content). 'neu' is a genuine scripted rest state,
# not an in-between one, so it stays.
TRANSITIONAL_LABELS = {'war', 'tra', 'coo'}


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def clean_rows(X, Y):
    """Drop rows with NaN/inf in either features or targets."""
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return X[mask], Y[mask]


def winsorize_with_train_bounds(X_train, X_val, pct):
    """Clip both splits to [pct, 100-pct] percentile bounds computed from
    TRAINING data only.

    No outlier/glitch rejection exists anywhere upstream of this script --
    confirmed by audit: continuous_feature_extractor.py's trend/rate-of-change
    features only gate on a minimum sample COUNT (not elapsed time), which is
    too loose to catch cold-start blowups in the first few seconds of a
    session, and features like eda.scr_recent_mean_amplitude have no bounds
    checking at all. A single such glitch is finite (passes clean_rows) but
    can dominate a Ridge fit -- e.g. one t=928s amplitude spike (z=13) drove
    an entire LOSO fold's R^2 to -84.

    Clipping (not dropping) the row: earlier tests showed dropping rows
    (feeling_it filtering, transitional exclusion on the full feature set)
    unpredictably destabilized fits by shifting which extreme values survive
    to dominate. Clipping caps a glitch's influence while keeping the row's
    other ~72 feature values and its W target intact.

    IMPORTANT: bounds must come from training data only, NOT per-session
    (including the held-out one) -- tried that first and it made things much
    worse (all-73 LOSO mean R^2 went from -1.46 to -8.76). Clipping a
    validation session to its OWN percentile bounds doesn't correct the
    cross-session scale mismatch, it just creates a session-specific plateau
    of clipped values that a model trained on a *different* session's scale
    then mispredicts systematically across many rows, instead of being wrong
    on just the few genuinely bad ones. Bounds from training data only (this
    version) is standard practice and fixed it: cardiac-only LOSO mean R^2
    improved from -0.519 to -0.362.
    """
    lo = np.nanpercentile(X_train, pct, axis=0)
    hi = np.nanpercentile(X_train, 100 - pct, axis=0)
    return np.clip(X_train, lo, hi), np.clip(X_val, lo, hi)


@torch.no_grad()
def save_visual_grid(G, val_df, w_true, w_pred, device, output_path, n=8):
    """Side-by-side: encoder's ground-truth W vs. biodata-predicted W, both
    rendered through the frozen StyleGAN2. This is the check that actually
    matters -- a low regression MSE doesn't guarantee the rendered face looks
    right, since W-space isn't uniformly perceptual."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    step = max(1, len(val_df) // n)
    indices = list(range(0, len(val_df), step))[:n]

    fig, axes = plt.subplots(2, len(indices), figsize=(2.5 * len(indices), 5.5))
    if len(indices) == 1:
        axes = axes[:, None]

    for col, i in enumerate(indices):
        wt = torch.tensor(w_true[i:i + 1], dtype=torch.float32, device=device)
        wp = torch.tensor(w_pred[i:i + 1], dtype=torch.float32, device=device)

        gen_t = generate(G, wt)
        img_t = F.interpolate(gen_t, size=256, mode='bilinear', align_corners=False)
        gen_p = generate(G, wp)
        img_p = F.interpolate(gen_p, size=256, mode='bilinear', align_corners=False)

        emotion = val_df.iloc[i]['emotion_label']
        for row, (img, label) in enumerate([(img_t, 'Encoder W (GT)'),
                                            (img_p, 'Predicted from biodata')]):
            arr = ((img[0].cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
            axes[row, col].imshow(arr)
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(emotion, fontsize=7)
            if col == 0:
                axes[row, col].set_ylabel(label, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Stage 5: Train biodata -> W regressor')
    parser.add_argument('--config', default='latent_pipeline/configs/default.yaml')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Ridge regularization strength (default: from config)')
    parser.add_argument('--val-pool', default=None,
                        help='Pool held out for validation (default: from config)')
    parser.add_argument('--keep-transitional', action='store_true',
                        help='Keep war/tra/coo (warmup/transition/cooldown) frames '
                             'instead of excluding them (default: exclude)')
    parser.add_argument('--winsorize-pct', type=float, default=None,
                        help='Clip each feature to [pct, 100-pct] percentile bounds '
                             'computed from training data only (default: from config; '
                             '0 disables)')
    args = parser.parse_args()

    config = load_config(args.config)
    rc = config['biodata_regressor']
    repo_root = config['paths']['repo_root']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    alpha = args.alpha if args.alpha is not None else rc['alpha']
    val_pool = args.val_pool if args.val_pool is not None else rc['val_pool']
    winsorize_pct = args.winsorize_pct if args.winsorize_pct is not None else rc['winsorize_pct']

    dataset_path = os.path.join(repo_root, 'latent_pipeline', 'data', 'biodata_w_dataset.csv')
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} rows from {dataset_path}")

    if not args.keep_transitional:
        n_before = len(df)
        # isin() is False for NaN, so also explicitly drop unannotated rows
        # (frame_number outside any labeled range) -- unknown state, same
        # "don't train on ambiguous data" reasoning as excluding war/tra/coo.
        exclude = df['raw_emotion'].isin(TRANSITIONAL_LABELS) | df['raw_emotion'].isna()
        df = df[~exclude].reset_index(drop=True)
        print(f"Excluded {n_before - len(df)} warmup/transition/cooldown/unannotated frames "
              f"-> {len(df)} rows (pass --keep-transitional to disable)")

    w_cols = [c for c in df.columns if c.startswith('w_')]
    feature_cols = [c for c in df.columns if c not in w_cols and c not in METADATA_COLS]
    print(f"W dims: {len(w_cols)}, feature dims: {len(feature_cols)} (full baseline, no selection)")

    train_df = df[df['pool_name'] != val_pool].reset_index(drop=True)
    val_df = df[df['pool_name'] == val_pool].reset_index(drop=True)
    print(f"Train: {len(train_df)} rows {sorted(train_df['pool_name'].unique())}")
    print(f"Val:   {len(val_df)} rows ['{val_pool}']")

    X_train, Y_train = clean_rows(train_df[feature_cols].values, train_df[w_cols].values)
    X_val, Y_val = clean_rows(val_df[feature_cols].values, val_df[w_cols].values)
    if len(X_train) < len(train_df) or len(X_val) < len(val_df):
        print(f"  Dropped {len(train_df) - len(X_train)} train / "
              f"{len(val_df) - len(X_val)} val rows with NaN/inf")

    if winsorize_pct > 0:
        X_train, X_val = winsorize_with_train_bounds(X_train, X_val, winsorize_pct)
        print(f"Winsorized to training-only [{winsorize_pct}, {100-winsorize_pct}] percentile bounds")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    print(f"\nTraining Ridge (alpha={alpha}) on {X_train_s.shape[1]} features "
          f"-> {Y_train.shape[1]} W dims...")
    model = Ridge(alpha=alpha)
    model.fit(X_train_s, Y_train)

    train_pred = model.predict(X_train_s)
    val_pred = model.predict(X_val_s)

    train_mse = mean_squared_error(Y_train, train_pred)
    val_mse = mean_squared_error(Y_val, val_pred)
    train_r2 = r2_score(Y_train, train_pred)
    val_r2 = r2_score(Y_val, val_pred)
    val_r2_per_dim = r2_score(Y_val, val_pred, multioutput='raw_values')

    print(f"Train MSE: {train_mse:.4f}  R^2: {train_r2:.4f}")
    print(f"Val   MSE: {val_mse:.4f}  R^2: {val_r2:.4f}")
    print(f"Val R^2 per-W-dim: min={val_r2_per_dim.min():.3f} "
          f"median={np.median(val_r2_per_dim):.3f} max={val_r2_per_dim.max():.3f}")

    output_dir = os.path.join(repo_root, rc['checkpoint_dir'])
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'w_cols': w_cols,
        'alpha': alpha,
        'val_pool': val_pool,
    }, os.path.join(output_dir, 'ridge_baseline.joblib'))

    report = {
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'n_features': len(feature_cols),
        'alpha': alpha,
        'val_pool': val_pool,
        'winsorize_pct': winsorize_pct,
        'excluded_transitional': not args.keep_transitional,
        'train_mse': float(train_mse),
        'val_mse': float(val_mse),
        'train_r2': float(train_r2),
        'val_r2': float(val_r2),
        'val_r2_per_dim_min': float(val_r2_per_dim.min()),
        'val_r2_per_dim_median': float(np.median(val_r2_per_dim)),
        'val_r2_per_dim_max': float(val_r2_per_dim.max()),
    }
    report_path = os.path.join(output_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved model + report to {output_dir}")

    print("\nRendering qualitative comparison grid...")
    G = load_stylegan(config, device)
    save_visual_grid(G, val_df, Y_val, val_pred, device,
                     os.path.join(output_dir, 'visual_check.png'), n=rc['n_visual'])


if __name__ == '__main__':
    main()
