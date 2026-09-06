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
from sklearn.neural_network import MLPRegressor
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


def assign_time_blocks(df, block_size_s):
    """Group rows into contiguous ~block_size_s-second chunks within each
    pool_name, for block-shuffled cross-validation.

    Session-holdout (LOSO) evaluation measures cross-session extrapolation,
    which isn't actually this regressor's job at deployment time -- that's
    what biodata_pipeline's cross-subject alignment step is for (it runs
    BEFORE biodata reaches this regressor, specifically to absorb the
    different-session/different-subject baseline problem). This regressor's
    real job is closer to interpolating within the actress' own reference
    space. Session holdout also isn't the right bar for a small, single-
    subject art dataset in the first place. Block-shuffled k-fold (pooling
    data across all sessions, splitting on contiguous time chunks) measures
    that interpolation task instead.

    Blocks, not individual frames: at 1Hz, consecutive frames are highly
    autocorrelated (near-duplicate biodata and W), so a pure per-frame
    shuffle would leak near-identical neighbors across train/val and inflate
    R^2 without teaching us anything. block_size_s=10s is a deliberate
    middle ground -- real emotion segments range from 1.8s to 222.6s
    (median 56.6s), so 10s blocks give the median segment ~5-6 blocks
    (multiple blocks per emotion, not one), while still being long enough
    to meaningfully separate train/val neighbors. A handful of segments
    under ~10-15s unavoidably end up as a single block each.
    """
    block_idx = (df['timestamp_s'] // block_size_s).astype(int)
    return df['pool_name'].astype(str) + '_' + block_idx.astype(str)


def blocked_kfold_indices(block_ids, n_folds, seed=42):
    """Shuffle unique blocks (not rows) and split into n_folds groups."""
    rng = np.random.RandomState(seed)
    unique_blocks = np.array(block_ids.unique(), dtype=object)
    rng.shuffle(unique_blocks)
    fold_of_block = {b: i % n_folds for i, b in enumerate(unique_blocks)}
    return block_ids.map(fold_of_block).values


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


def make_model(model_type, alpha, mlp_hidden_layers):
    if model_type == 'ridge':
        return Ridge(alpha=alpha)
    if model_type == 'mlp':
        return MLPRegressor(hidden_layer_sizes=tuple(mlp_hidden_layers), alpha=alpha,
                            max_iter=500, early_stopping=True, n_iter_no_change=15,
                            random_state=0)
    raise ValueError(f"Unknown model_type: {model_type}")


def fit_eval_fold(train_df, val_df, feature_cols, w_cols, alpha, winsorize_pct,
                  model_type='ridge', mlp_hidden_layers=(256, 128)):
    """Fit a model on one train/val split and return (model, scaler, metrics, val_pred, Y_val).

    Shared by both CV modes (LOSO and blocked k-fold) so they apply the exact
    same pipeline (NaN drop -> winsorize -> scale -> fit) and stay comparable.

    model_type='mlp': under blocked CV (see assign_time_blocks), a small MLP
    clearly beats Ridge -- (256,128) hidden layers, alpha=3.0 gave mean val
    R^2=0.457 vs Ridge's 0.272 on the full 51-feature batch dataset. Ridge's
    alpha sweep was flat (0.271-0.274 across 5 orders of magnitude), meaning
    the ceiling wasn't regularization, it was the linear model's limited
    expressiveness. Note: MLP fold-to-fold variance is much higher than
    Ridge's (std ~0.04-0.09 vs ~0.02) -- less stable, even though the mean
    is clearly better; alpha here still matters a lot more for MLP than it
    did for Ridge (too little -> overfits and destabilizes, too much ->
    underfits) so don't assume the tuned defaults transfer to a different
    feature set without re-checking.
    """
    X_train, Y_train = clean_rows(train_df[feature_cols].values, train_df[w_cols].values)
    X_val, Y_val = clean_rows(val_df[feature_cols].values, val_df[w_cols].values)

    if winsorize_pct > 0:
        X_train, X_val = winsorize_with_train_bounds(X_train, X_val, winsorize_pct)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = make_model(model_type, alpha, mlp_hidden_layers)
    model.fit(X_train_s, Y_train)
    val_pred = model.predict(X_val_s)
    train_pred = model.predict(X_train_s)

    metrics = {
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'train_mse': float(mean_squared_error(Y_train, train_pred)),
        'val_mse': float(mean_squared_error(Y_val, val_pred)),
        'train_r2': float(r2_score(Y_train, train_pred)),
        'val_r2': float(r2_score(Y_val, val_pred)),
    }
    return model, scaler, metrics, val_pred, Y_val


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
    parser.add_argument('--model', choices=['ridge', 'mlp'], default=None,
                        help='ridge (default) or mlp -- under blocked CV a small MLP '
                             '(256,128) clearly beats Ridge (val R^2 0.457 vs 0.272 on the '
                             'batch feature set), at the cost of higher fold-to-fold '
                             'variance (default: from config)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Regularization strength -- Ridge alpha or MLP L2 alpha '
                             '(default: from config)')
    parser.add_argument('--mlp-hidden-layers', type=int, nargs='+', default=None,
                        help='[mlp] Hidden layer sizes, e.g. --mlp-hidden-layers 256 128 '
                             '(default: from config)')
    parser.add_argument('--cv-mode', choices=['blocked', 'loso'], default=None,
                        help='blocked = shuffled k-fold over time blocks pooled across all '
                             'sessions (default -- measures interpolation, the actual job '
                             'this regressor has at deployment, since cross-subject '
                             'alignment already handles cross-session baseline differences '
                             'upstream); loso = leave-one-session-out (measures cross-'
                             'session extrapolation instead -- kept for comparison)')
    parser.add_argument('--val-pool', default=None,
                        help='[loso mode] Pool held out for validation (default: from config)')
    parser.add_argument('--block-size-s', type=float, default=None,
                        help='[blocked mode] Time-block size in seconds (default: from config)')
    parser.add_argument('--n-folds', type=int, default=None,
                        help='[blocked mode] Number of folds (default: from config)')
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

    model_type = args.model if args.model is not None else rc.get('model_type', 'ridge')
    default_alpha = rc.get('mlp_alpha', 3.0) if model_type == 'mlp' else rc['alpha']
    alpha = args.alpha if args.alpha is not None else default_alpha
    mlp_hidden_layers = tuple(args.mlp_hidden_layers) if args.mlp_hidden_layers is not None \
        else tuple(rc.get('mlp_hidden_layers', [256, 128]))
    cv_mode = args.cv_mode if args.cv_mode is not None else rc.get('cv_mode', 'blocked')
    winsorize_pct = args.winsorize_pct if args.winsorize_pct is not None else rc['winsorize_pct']
    print(f"Model: {model_type}" + (f" hidden={mlp_hidden_layers}" if model_type == 'mlp' else '')
          + f" alpha={alpha}")

    dataset_path = os.path.join(repo_root, 'latent_pipeline', 'data', 'biodata_w_dataset.csv')
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} rows from {dataset_path}")

    if not args.keep_transitional:
        n_before = len(df)
        exclude = df['raw_emotion'].isin(TRANSITIONAL_LABELS) | df['raw_emotion'].isna()
        df = df[~exclude].reset_index(drop=True)
        print(f"Excluded {n_before - len(df)} warmup/transition/cooldown/unannotated frames "
              f"-> {len(df)} rows (pass --keep-transitional to disable)")

    w_cols = [c for c in df.columns if c.startswith('w_')]
    feature_cols = [c for c in df.columns if c not in w_cols and c not in METADATA_COLS]
    print(f"W dims: {len(w_cols)}, feature dims: {len(feature_cols)} (full baseline, no selection)")
    print(f"CV mode: {cv_mode}")

    output_dir = os.path.join(repo_root, rc['checkpoint_dir'])
    os.makedirs(output_dir, exist_ok=True)

    if cv_mode == 'loso':
        val_pool = args.val_pool if args.val_pool is not None else rc['val_pool']
        train_df = df[df['pool_name'] != val_pool].reset_index(drop=True)
        val_df = df[df['pool_name'] == val_pool].reset_index(drop=True)
        print(f"Train: {len(train_df)} rows {sorted(train_df['pool_name'].unique())}")
        print(f"Val:   {len(val_df)} rows ['{val_pool}']")

        model, scaler, metrics, val_pred, Y_val = fit_eval_fold(
            train_df, val_df, feature_cols, w_cols, alpha, winsorize_pct,
            model_type, mlp_hidden_layers)
        val_r2_per_dim = r2_score(Y_val, val_pred, multioutput='raw_values')

        print(f"Train MSE: {metrics['train_mse']:.4f}  R^2: {metrics['train_r2']:.4f}")
        print(f"Val   MSE: {metrics['val_mse']:.4f}  R^2: {metrics['val_r2']:.4f}")
        print(f"Val R^2 per-W-dim: min={val_r2_per_dim.min():.3f} "
              f"median={np.median(val_r2_per_dim):.3f} max={val_r2_per_dim.max():.3f}")

        report = {
            'cv_mode': 'loso', 'val_pool': val_pool, 'model_type': model_type, 'alpha': alpha,
            'n_features': len(feature_cols), 'winsorize_pct': winsorize_pct,
            'excluded_transitional': not args.keep_transitional,
            **metrics,
            'val_r2_per_dim_min': float(val_r2_per_dim.min()),
            'val_r2_per_dim_median': float(np.median(val_r2_per_dim)),
            'val_r2_per_dim_max': float(val_r2_per_dim.max()),
        }
        grid_val_df = val_df

    else:  # blocked
        block_size_s = args.block_size_s if args.block_size_s is not None else rc['block_size_s']
        n_folds = args.n_folds if args.n_folds is not None else rc['n_folds']
        blocks = assign_time_blocks(df, block_size_s)
        fold_ids = blocked_kfold_indices(blocks, n_folds)
        print(f"Block size: {block_size_s}s -> {blocks.nunique()} blocks, {n_folds} folds")

        fold_metrics = []
        best_fold_val_df, best_fold_val_pred, best_fold_Y_val = None, None, None
        for fold in range(n_folds):
            train_df = df[fold_ids != fold].reset_index(drop=True)
            val_df = df[fold_ids == fold].reset_index(drop=True)
            model, scaler, metrics, val_pred, Y_val = fit_eval_fold(
                train_df, val_df, feature_cols, w_cols, alpha, winsorize_pct,
                model_type, mlp_hidden_layers)
            fold_metrics.append(metrics)
            print(f"  Fold {fold}: n_train={metrics['n_train']} n_val={metrics['n_val']} "
                  f"train_r2={metrics['train_r2']:.3f} val_r2={metrics['val_r2']:.3f}")
            if fold == 0:
                best_fold_val_df, best_fold_val_pred, best_fold_Y_val = val_df, val_pred, Y_val

        val_r2s = np.array([m['val_r2'] for m in fold_metrics])
        train_r2s = np.array([m['train_r2'] for m in fold_metrics])
        print(f"\nTrain R^2: mean={train_r2s.mean():.4f} std={train_r2s.std():.4f}")
        print(f"Val   R^2: mean={val_r2s.mean():.4f} std={val_r2s.std():.4f} "
              f"| per-fold: {[f'{r:.3f}' for r in val_r2s]}")

        # Refit on ALL data for the saved model/checkpoint (folds above are
        # for evaluation only) -- deploy-time model should use every frame.
        full_train_df = df
        dummy_val_df = df.sample(min(len(df), rc['n_visual'] * 10), random_state=0)
        model, scaler, _, _, _ = fit_eval_fold(
            full_train_df, dummy_val_df, feature_cols, w_cols, alpha, winsorize_pct,
            model_type, mlp_hidden_layers)

        report = {
            'cv_mode': 'blocked', 'block_size_s': block_size_s, 'n_folds': n_folds,
            'model_type': model_type, 'alpha': alpha,
            'n_features': len(feature_cols), 'winsorize_pct': winsorize_pct,
            'excluded_transitional': not args.keep_transitional,
            'val_r2_mean': float(val_r2s.mean()), 'val_r2_std': float(val_r2s.std()),
            'val_r2_per_fold': [float(r) for r in val_r2s],
            'train_r2_mean': float(train_r2s.mean()), 'train_r2_std': float(train_r2s.std()),
        }
        grid_val_df, val_pred, val_true_for_grid = best_fold_val_df, best_fold_val_pred, best_fold_Y_val

    joblib.dump({
        'model': model, 'scaler': scaler, 'feature_cols': feature_cols, 'w_cols': w_cols,
        'model_type': model_type, 'alpha': alpha, 'cv_mode': cv_mode,
    }, os.path.join(output_dir, 'regressor.joblib'))

    report_path = os.path.join(output_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved model + report to {output_dir}")

    print("\nRendering qualitative comparison grid...")
    G = load_stylegan(config, device)
    grid_true = Y_val if cv_mode == 'loso' else val_true_for_grid
    save_visual_grid(G, grid_val_df, grid_true, val_pred, device,
                     os.path.join(output_dir, 'visual_check.png'), n=rc['n_visual'])


if __name__ == '__main__':
    main()
