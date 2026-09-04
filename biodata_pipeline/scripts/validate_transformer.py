"""
Validate a trained subject alignment transformer.

Two validation methods:
1. Separation ratio: transforms subject samples, measures distance to correct
   vs wrong reference prototypes.
2. RF classifier: trains a Random Forest on reference data, classifies
   transformed subject data, reports accuracy per emotion.

Usage (from biodata_pipeline/):
    python scripts/validate_transformer.py --reference data/processed/laurence_main_features.csv --subject data/processed/erin_2026-02-09_features.csv
    python scripts/validate_transformer.py --reference data/processed/continuous_features.csv --subject data/processed/erin_features.csv --model models/custom_transformer.pkl
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Allow importing PrototypeAlignmentTransformer when run from biodata_pipeline/
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_transformer import load_transformer, METADATA_COLS


def validate(transformer, subject_df):
    """Per-emotion separation ratio validation."""
    feature_cols = transformer.feature_cols
    emotions = transformer.common_emotions
    results = {}

    print("\n" + "=" * 60)
    print("PER-EMOTION VALIDATION")
    print("=" * 60)

    for emotion in emotions:
        rows = subject_df.loc[subject_df['emotion'] == emotion, feature_cols].values
        rows = rows[~np.isnan(rows).any(axis=1)]
        rows = rows[~np.isinf(rows).any(axis=1)]

        if len(rows) == 0:
            print(f"  {emotion}: no valid samples, skipping")
            continue

        transformed = transformer.transform(rows)
        correct_proto = transformer.reference_prototypes[emotion]

        dist_correct = np.mean(np.linalg.norm(transformed - correct_proto, axis=1))

        dists_wrong = []
        for other in emotions:
            if other != emotion:
                d = np.mean(np.linalg.norm(
                    transformed - transformer.reference_prototypes[other], axis=1
                ))
                dists_wrong.append(d)

        mean_wrong = np.mean(dists_wrong) if dists_wrong else float('inf')
        sep_ratio = float(dist_correct / mean_wrong) if mean_wrong > 0 else float('inf')

        results[emotion] = {
            'n_samples': int(len(rows)),
            'dist_to_correct': float(dist_correct),
            'mean_dist_to_wrong': float(mean_wrong),
            'separation_ratio': sep_ratio,
        }

        quality = 'EXCELLENT' if sep_ratio < 0.7 else ('GOOD' if sep_ratio < 1.0 else 'NEEDS WORK')
        print(f"  {emotion}: ratio={sep_ratio:.3f} ({quality})  "
              f"n={len(rows)}  correct={dist_correct:.2f}  wrong={mean_wrong:.2f}")

    mean_sep = float(np.mean([r['separation_ratio'] for r in results.values()]))
    results['_summary'] = {'mean_separation_ratio': mean_sep, 'n_emotions': len(results)}

    print(f"\n{'=' * 60}")
    print(f"Mean separation ratio: {mean_sep:.3f}")
    if mean_sep < 0.7:
        print("EXCELLENT - transformation works very well")
    elif mean_sep < 1.0:
        print("GOOD - transformation is usable")
    else:
        print("NEEDS IMPROVEMENT")
    print("=" * 60)

    return results


def validate_with_classifier(transformer, reference_df, subject_df):
    """Train RF on reference data, classify transformed subject data."""
    feature_cols = transformer.feature_cols
    emotions = transformer.common_emotions

    print("\n" + "=" * 60)
    print("RANDOM FOREST CLASSIFIER VALIDATION")
    print("=" * 60)

    # Prepare reference training data (filter to common emotions, drop NaN/Inf)
    ref_mask = reference_df['emotion'].isin(emotions)
    ref_sub = reference_df.loc[ref_mask].copy()
    X_ref = ref_sub[feature_cols].values
    y_ref = ref_sub['emotion'].values

    valid = ~(np.isnan(X_ref).any(axis=1) | np.isinf(X_ref).any(axis=1))
    X_ref, y_ref = X_ref[valid], y_ref[valid]

    print(f"  Training RF on {len(X_ref)} reference samples")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_ref, y_ref)

    # Reference self-accuracy (sanity check)
    ref_acc = accuracy_score(y_ref, rf.predict(X_ref))
    print(f"  Reference self-accuracy: {ref_acc:.1%}")

    # Prepare subject data, transform, classify
    sub_mask = subject_df['emotion'].isin(emotions)
    sub_sub = subject_df.loc[sub_mask].copy()
    X_sub = sub_sub[feature_cols].values
    y_true = sub_sub['emotion'].values

    valid = ~(np.isnan(X_sub).any(axis=1) | np.isinf(X_sub).any(axis=1))
    X_sub, y_true = X_sub[valid], y_true[valid]

    X_transformed = transformer.transform(X_sub)
    y_pred = rf.predict(X_transformed)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Transformed subject accuracy: {acc:.1%}")
    print()
    print(classification_report(y_true, y_pred, zero_division=0))

    # Also test without transformation as baseline
    y_pred_raw = rf.predict(X_sub)
    raw_acc = accuracy_score(y_true, y_pred_raw)
    print(f"  Baseline (no transform): {raw_acc:.1%}")
    print(f"  With transform:          {acc:.1%}")

    return {
        'reference_self_accuracy': float(ref_acc),
        'transformed_accuracy': float(acc),
        'baseline_accuracy': float(raw_acc),
        'n_reference_samples': int(len(X_ref)),
        'n_subject_samples': int(len(X_sub)),
        'per_class': {
            emotion: {
                'precision': float((y_pred[y_true == emotion] == emotion).sum() /
                                   max((y_pred == emotion).sum(), 1)),
                'recall': float((y_pred[y_true == emotion] == emotion).sum() /
                                max((y_true == emotion).sum(), 1)),
            }
            for emotion in emotions
        },
    }


def validate_alignment_quality(transformer, subject_df):
    """
    Compute two method-agnostic alignment quality metrics.

    Nearest-Prototype Accuracy (NPA)
        For each transformed subject window, check whether the nearest
        actress' prototype is the correct one.  No classifier needed.
        Random baseline = 1 / n_emotions.

    Normalized Prototype RMSE
        Root-mean-square distance between each transformed subject prototype
        and its reference target, divided by the mean pairwise distance
        between reference prototypes.
        0 = perfect alignment.  1 = error as large as the inter-class spread.

    Both metrics work for any alignment method that exposes transform().
    """
    feature_cols = transformer.feature_cols
    emotions = transformer.common_emotions
    n = len(emotions)

    # Reference prototypes in raw reference space
    ref_protos_scaled = np.array([transformer.reference_prototypes[e] for e in emotions])
    ref_protos_raw = transformer.ref_scaler.inverse_transform(ref_protos_scaled)

    # Transformed subject prototypes via the method-agnostic interface
    trans_protos_raw = transformer.transform_prototypes()

    # ── Normalized Prototype RMSE ─────────────────────────────────────────
    proto_errors = np.linalg.norm(trans_protos_raw - ref_protos_raw, axis=1)
    rmse = float(np.sqrt(np.mean(proto_errors ** 2)))

    pairwise_dists = [
        np.linalg.norm(ref_protos_raw[i] - ref_protos_raw[j])
        for i in range(n) for j in range(i + 1, n)
    ]
    mean_pairwise = float(np.mean(pairwise_dists)) if pairwise_dists else 1.0
    rmse_norm = rmse / mean_pairwise if mean_pairwise > 0 else float('inf')

    # ── Nearest-Prototype Accuracy (NPA) ──────────────────────────────────
    correct = 0
    total = 0
    per_emotion_npa = {}

    for i, emotion in enumerate(emotions):
        rows = subject_df.loc[subject_df['emotion'] == emotion, feature_cols].values
        rows = rows[~np.isnan(rows).any(axis=1)]
        rows = rows[~np.isinf(rows).any(axis=1)]

        if len(rows) == 0:
            per_emotion_npa[emotion] = None
            continue

        transformed = transformer.transform(rows)  # (n_windows, n_features), raw ref space

        # Distance from each window to each reference prototype
        dists = np.stack([
            np.linalg.norm(transformed - ref_protos_raw[j], axis=1)
            for j in range(n)
        ], axis=1)  # (n_windows, n_emotions)

        n_correct = int((np.argmin(dists, axis=1) == i).sum())
        per_emotion_npa[emotion] = float(n_correct / len(rows))
        correct += n_correct
        total += len(rows)

    npa = float(correct / total) if total > 0 else 0.0

    # ── Print ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ALIGNMENT QUALITY METRICS")
    print("=" * 60)

    print(f"\nNormalized Prototype RMSE: {rmse_norm:.3f}")
    print(f"  (0=perfect, 1=error as large as inter-class distance)")
    print(f"  Raw RMSE:            {rmse:.4f}")
    print(f"  Mean pairwise dist:  {mean_pairwise:.4f}")
    print(f"  Per-emotion prototype error:")
    for i, emotion in enumerate(emotions):
        print(f"    {emotion}: {proto_errors[i]:.4f}")

    print(f"\nNearest-Prototype Accuracy: {npa:.1%}  "
          f"(random baseline: {1/n:.1%})")
    print(f"  Per emotion:")
    for emotion, val in per_emotion_npa.items():
        marker = '' if val is None else f'  ({val:.1%})'
        label = 'no samples' if val is None else ''
        print(f"    {emotion}: {label}{marker}")

    print("=" * 60)

    return {
        'rmse_norm': rmse_norm,
        'rmse_raw': rmse,
        'mean_pairwise_dist': mean_pairwise,
        'per_emotion_proto_error': {e: float(proto_errors[i]) for i, e in enumerate(emotions)},
        'npa': npa,
        'npa_random_baseline': float(1 / n),
        'per_emotion_npa': per_emotion_npa,
    }


def visualize(transformer, output_path):
    """PCA plot: subject prototypes (red) -> transformed (green) vs reference (blue)."""
    emotions = transformer.common_emotions
    sub_protos = np.array([transformer.subject_prototypes[e] for e in emotions])
    ref_protos = np.array([transformer.reference_prototypes[e] for e in emotions])
    trans_protos = transformer.transform(sub_protos)

    all_data = np.vstack([sub_protos, ref_protos, trans_protos])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_data)

    n = len(emotions)
    sub_proj = proj[:n]
    ref_proj = proj[n:2*n]
    trans_proj = proj[2*n:]

    fig, ax = plt.subplots(figsize=(12, 9))

    for i, emotion in enumerate(emotions):
        # Arrow from subject to transformed
        ax.annotate(
            '', xy=(trans_proj[i, 0], trans_proj[i, 1]),
            xytext=(sub_proj[i, 0], sub_proj[i, 1]),
            arrowprops=dict(arrowstyle='->', color='orange', alpha=0.5, lw=2)
        )

    # Plot points after arrows so they sit on top
    ax.scatter(sub_proj[:, 0], sub_proj[:, 1],
               c='red', marker='o', s=250, alpha=0.7, edgecolors='black',
               linewidth=1.5, label='Subject original', zorder=3)
    ax.scatter(trans_proj[:, 0], trans_proj[:, 1],
               c='green', marker='^', s=250, alpha=0.7, edgecolors='black',
               linewidth=1.5, label='Transformed', zorder=3)
    ax.scatter(ref_proj[:, 0], ref_proj[:, 1],
               c='blue', marker='s', s=250, alpha=0.7, edgecolors='black',
               linewidth=1.5, label='Reference target', zorder=3)

    for i, emotion in enumerate(emotions):
        ax.annotate(f' {emotion.upper()}', (ref_proj[i, 0], ref_proj[i, 1]),
                     fontsize=13, fontweight='bold')

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', fontsize=12)
    ax.set_title('Prototype Alignment: Subject -> Reference Space', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate subject alignment transformer")
    parser.add_argument(
        '--reference', required=True,
        help='Reference subject features CSV (used to train RF classifier)'
    )
    parser.add_argument(
        '--subject', required=True,
        help='Subject features CSV to validate against'
    )
    parser.add_argument(
        '--model',
        default='models/subject_alignment_transformer.pkl',
        help='Path to trained transformer'
    )
    parser.add_argument(
        '--output-dir',
        default='reports/',
        help='Directory for validation outputs'
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Transformer Validation")
    print("=" * 60)

    transformer = load_transformer(args.model)
    print(f"Loaded transformer from {args.model} ({type(transformer).__name__})")
    print(f"  Emotions: {transformer.common_emotions}")
    print(f"  Features: {len(transformer.feature_cols)}")

    reference_df = pd.read_csv(args.reference)
    print(f"Reference data: {len(reference_df)} samples from {args.reference}")

    subject_df = pd.read_csv(args.subject)
    print(f"Subject data: {len(subject_df)} samples from {args.subject}")

    results = validate(transformer, subject_df)
    results['alignment_quality'] = validate_alignment_quality(transformer, subject_df)
    results['classifier'] = validate_with_classifier(transformer, reference_df, subject_df)

    # Save JSON (numpy types → Python floats)
    json_path = out_dir / 'validation_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    visualize(transformer, str(out_dir / 'transformation_validation.png'))

    print("\nDone.")


if __name__ == '__main__':
    main()
