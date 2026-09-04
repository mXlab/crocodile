"""
Cross-Validation Evaluation for Windowed Data

This script performs proper cross-validation on windowed physiological data,
accounting for temporal dependencies between overlapping windows.

Usage:
    python scripts/evaluate_with_cv.py --features data/processed/all_features_hybrid.csv
"""

import sys
from pathlib import Path
import argparse

# Fix Python path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_prepare_data(features_path, group_by=None):
    """Load features and separate X, y, groups."""

    print(f"Loading features from: {features_path}")
    features = pd.read_csv(features_path)

    print(f"✓ Loaded {len(features):,} samples")
    print(f"  Emotions: {features['emotion'].nunique()}")

    # Identify metadata columns
    metadata_cols = [
        'window_id', 'parent_segment_id', 'segment_id',
        'session_id', 'emotion', 'duration',
        'feeling_it', 'feeling_it_ratio',
        'window_mode', 'emotion_purity'
    ]

    # Feature columns
    feature_cols = [c for c in features.columns if c not in metadata_cols]

    print(f"  Features: {len(feature_cols)}")

    # Extract data
    X = features[feature_cols].values
    y = features['emotion'].values

    # Groups for CV
    if group_by:
        if group_by not in features.columns:
            raise ValueError(f"--group-by column '{group_by}' not found in dataset")
        groups = features[group_by].values
        print(f"  Using '{group_by}' for grouping (--group-by)")
    elif 'parent_segment_id' in features.columns:
        groups = features['parent_segment_id'].fillna(features['window_id']).values
        print(f"  Using parent_segment_id for grouping")
    else:
        groups = features['session_id'].values
        print(f"  Using session_id for grouping")

    n_groups = len(np.unique(groups))
    print(f"  Unique groups: {n_groups}")

    return X, y, groups, features


def evaluate_with_groupkfold(X, y, groups, n_splits=5, random_state=42):
    """
    Perform GroupKFold cross-validation.
    
    Returns detailed results per fold.
    """
    
    print("\n" + "="*80)
    print("GROUP K-FOLD CROSS-VALIDATION")
    print("="*80)
    print(f"Number of folds: {n_splits}")
    print(f"Classifier: Random Forest (100 trees)")
    
    # Initialize CV
    cv = GroupKFold(n_splits=n_splits)
    
    # Prepare results storage
    fold_results = []
    
    # Perform CV
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), 1):
        print(f"\n{'─'*80}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'─'*80}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=random_state
        )
        
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"  Train samples: {len(train_idx):4d}")
        print(f"  Test samples:  {len(test_idx):4d}")
        print(f"  Accuracy:      {accuracy:.2%}")
        print(f"  F1 (macro):    {f1_macro:.2%}")
        print(f"  F1 (weighted): {f1_weighted:.2%}")
        
        # Store results
        fold_results.append({
            'fold': fold,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'y_test': y_test,
            'y_pred': y_pred
        })
    
    return fold_results


def print_summary(fold_results):
    """Print summary statistics across folds."""
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    # Extract metrics
    accuracies = [r['accuracy'] for r in fold_results]
    f1_macros = [r['f1_macro'] for r in fold_results]
    f1_weighteds = [r['f1_weighted'] for r in fold_results]
    
    # Print table
    print(f"\n{'Fold':<8} {'Accuracy':<12} {'F1 (macro)':<12} {'F1 (weighted)':<12}")
    print("─" * 80)
    
    for result in fold_results:
        print(f"{result['fold']:<8} "
              f"{result['accuracy']:>10.2%}  "
              f"{result['f1_macro']:>10.2%}  "
              f"{result['f1_weighted']:>10.2%}")
    
    print("─" * 80)
    print(f"{'Mean':<8} "
          f"{np.mean(accuracies):>10.2%}  "
          f"{np.mean(f1_macros):>10.2%}  "
          f"{np.mean(f1_weighteds):>10.2%}")
    
    print(f"{'Std':<8} "
          f"{np.std(accuracies):>10.2%}  "
          f"{np.std(f1_macros):>10.2%}  "
          f"{np.std(f1_weighteds):>10.2%}")
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Mean Accuracy:      {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
    print(f"Mean F1 (macro):    {np.mean(f1_macros):.2%} ± {np.std(f1_macros):.2%}")
    print(f"Mean F1 (weighted): {np.mean(f1_weighteds):.2%} ± {np.std(f1_weighteds):.2%}")
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_f1_macro': np.mean(f1_macros),
        'std_f1_macro': np.std(f1_macros)
    }


def print_per_emotion_results(fold_results):
    """Print classification report aggregated across folds."""
    
    print("\n" + "="*80)
    print("PER-EMOTION PERFORMANCE (Aggregated Across Folds)")
    print("="*80)
    
    # Combine all predictions
    all_y_test = np.concatenate([r['y_test'] for r in fold_results])
    all_y_pred = np.concatenate([r['y_pred'] for r in fold_results])
    
    # Classification report
    print("\n" + classification_report(all_y_test, all_y_pred, zero_division=0))
    
    return all_y_test, all_y_pred


def plot_confusion_matrix(y_test, y_pred, output_path=None):
    """Plot confusion matrix."""
    
    print("\nGenerating confusion matrix...")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    emotions = np.unique(np.concatenate([y_test, y_pred]))
    
    # Normalize by row (true label)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=emotions,
        yticklabels=emotions,
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title('Confusion Matrix (Normalized by True Label)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Emotion', fontsize=12, fontweight='bold')
    plt.ylabel('True Emotion', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved confusion matrix: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main evaluation workflow."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Cross-validation evaluation for windowed data'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='data/processed/all_features_hybrid.csv',
        help='Path to features CSV file'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/cross_validation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--group-by',
        type=str,
        default=None,
        help='Column to use for CV grouping (e.g. session_id for leave-one-session-out, '
             'parent_segment_id for super-segment grouping). '
             'Default: parent_segment_id if present, else session_id.'
    )

    args = parser.parse_args()
    
    print("="*80)
    print("CROSS-VALIDATION EVALUATION")
    print("="*80)
    
    # Create output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y, groups, features = load_and_prepare_data(PROJECT_ROOT / args.features,
                                                    group_by=args.group_by)
    
    # Check if we have enough groups for n_splits
    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < args.n_splits:
        print(f"\n⚠ Warning: Only {n_unique_groups} unique groups, reducing n_splits")
        args.n_splits = n_unique_groups
    
    # Perform cross-validation
    fold_results = evaluate_with_groupkfold(
        X, y, groups,
        n_splits=args.n_splits
    )
    
    # Print summary
    summary = print_summary(fold_results)
    
    # Per-emotion results
    y_test_all, y_pred_all = print_per_emotion_results(fold_results)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_test_all,
        y_pred_all,
        output_path=output_dir / 'confusion_matrix.png'
    )
    
    # Save results
    results_path = output_dir / 'cv_results.csv'
    results_df = pd.DataFrame(fold_results)
    results_df = results_df.drop(['y_test', 'y_pred'], axis=1)  # Remove arrays
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved results: {results_path}")
    
    # Save summary
    summary_path = output_dir / 'cv_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("CROSS-VALIDATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {args.features}\n")
        f.write(f"Samples: {len(X)}\n")
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"Emotions: {len(np.unique(y))}\n")
        f.write(f"Groups: {len(np.unique(groups))}\n")
        f.write(f"CV Folds: {args.n_splits}\n\n")
        f.write(f"Mean Accuracy:      {summary['mean_accuracy']:.2%} ± {summary['std_accuracy']:.2%}\n")
        f.write(f"Mean F1 (macro):    {summary['mean_f1_macro']:.2%} ± {summary['std_f1_macro']:.2%}\n")
    
    print(f"✓ Saved summary: {summary_path}")
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - cv_results.csv         Per-fold metrics")
    print("  - cv_summary.txt         Summary statistics")
    print("  - confusion_matrix.png   Confusion matrix heatmap")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    mean_acc = summary['mean_accuracy']
    baseline = 1 / len(np.unique(y))
    
    print(f"\nYour classifier: {mean_acc:.2%}")
    print(f"Random baseline:   {baseline:.2%} (1/{len(np.unique(y))} emotions)")
    print(f"Improvement:       {mean_acc/baseline:.1f}x better than random")
    
    if mean_acc >= 0.70:
        print("✓ Good performance! Classifier generalizes well to unseen data")
    elif mean_acc >= 0.50:
        print("⚠ Moderate performance. Consider:")
        print("  - Feature selection (use top features from Module 4)")
        print("  - Emotion consolidation (merge similar emotions)")
        print("  - More training data")
    else:
        print("⚠ Low performance. Likely issues:")
        print("  - Too many emotions for dataset size")
        print("  - Features not discriminative enough")
        print("  - Need more/better quality data")
    
    # print(f"\nRandom baseline: {1/len(np.unique(y)):.2%} (1/{len(np.unique(y))} emotions)")


if __name__ == "__main__":
    main()
