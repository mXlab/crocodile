"""
Simple Cross-Validation for Independent Windows

For use with non-overlapping windows where regular K-Fold is appropriate.

Usage:
    python scripts/train_and_evaluate.py --features data/processed/all_features_independent.csv
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
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, make_scorer
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def load_features(features_path):
    """Load features and separate X, y."""
    
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
    
    return X, y, features


def evaluate_simple_kfold(X, y, n_splits=5, random_state=42):
    """
    Perform simple K-Fold cross-validation.
    
    Works for independent (non-overlapping) windows.
    """
    
    print("\n" + "="*80)
    print("K-FOLD CROSS-VALIDATION (Independent Windows)")
    print("="*80)
    print(f"Number of folds: {n_splits}")
    print(f"Classifier: Random Forest (100 trees)")
    print(f"Shuffle: True (windows are independent)")
    
    # Initialize classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=random_state
    )
    
    # Initialize CV
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Prepare results storage
    fold_results = []
    all_y_test = []
    all_y_pred = []
    
    # Perform CV
    for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
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
        
        # Train
        clf.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
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
            'f1_weighted': f1_weighted
        })
        
        # Store predictions
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
    
    return fold_results, np.array(all_y_test), np.array(all_y_pred)


def print_summary(fold_results):
    """Print summary statistics."""
    
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


def print_classification_report(y_test, y_pred):
    """Print per-emotion classification metrics."""
    
    print("\n" + "="*80)
    print("PER-EMOTION PERFORMANCE")
    print("="*80)
    
    print("\n" + classification_report(y_test, y_pred, zero_division=0))


def plot_confusion_matrix(y_test, y_pred, output_path=None):
    """Plot confusion matrix."""
    
    print("\nGenerating confusion matrix...")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    emotions = np.unique(np.concatenate([y_test, y_pred]))
    
    # Normalize by row (true label)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Plot
    fig_size = max(10, len(emotions) * 0.5)
    plt.figure(figsize=(fig_size, fig_size))
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=emotions,
        yticklabels=emotions,
        cbar_kws={'label': 'Proportion'},
        square=True
    )
    
    plt.title('Confusion Matrix (Normalized by True Label)', 
             fontsize=14, fontweight='bold')
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
        description='Simple K-Fold evaluation for independent windows'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='data/processed/all_features_independent.csv',
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
        default='reports/evaluation',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("SIMPLE CROSS-VALIDATION EVALUATION")
    print("="*80)
    print("\nFor independent (non-overlapping) windows")
    print("Uses regular K-Fold (shuffle=True)")
    
    # Create output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y, features = load_features(PROJECT_ROOT / args.features)
    
    # Check sample size
    if len(X) < args.n_splits:
        print(f"\n⚠ Warning: Only {len(X)} samples, reducing n_splits")
        args.n_splits = min(args.n_splits, len(X))
    
    # Perform cross-validation
    fold_results, y_test_all, y_pred_all = evaluate_simple_kfold(
        X, y,
        n_splits=args.n_splits
    )
    
    # Print summary
    summary = print_summary(fold_results)
    
    # Per-emotion results
    print_classification_report(y_test_all, y_pred_all)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_test_all,
        y_pred_all,
        output_path=output_dir / 'confusion_matrix.png'
    )
    
    # Save results
    results_path = output_dir / 'cv_results.csv'
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved results: {results_path}")
    
    # Save summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("CROSS-VALIDATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {args.features}\n")
        f.write(f"Samples: {len(X)}\n")
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"Emotions: {len(np.unique(y))}\n")
        f.write(f"CV Method: K-Fold (shuffle=True)\n")
        f.write(f"CV Folds: {args.n_splits}\n\n")
        f.write(f"Mean Accuracy:      {summary['mean_accuracy']:.2%} ± {summary['std_accuracy']:.2%}\n")
        f.write(f"Mean F1 (macro):    {summary['mean_f1_macro']:.2%} ± {summary['std_f1_macro']:.2%}\n")
        f.write(f"\nRandom Baseline: {1/len(np.unique(y)):.2%}\n")
    
    print(f"✓ Saved summary: {summary_path}")
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    mean_acc = summary['mean_accuracy']
    baseline = 1 / len(np.unique(y))
    
    print(f"\nYour classifier:   {mean_acc:.2%}")
    print(f"Random baseline:   {baseline:.2%} (1/{len(np.unique(y))} emotions)")
    print(f"Improvement:       {mean_acc/baseline:.1f}x better than random")
    
    if mean_acc >= 0.70:
        print("\n✓ Excellent performance!")
    elif mean_acc >= 0.60:
        print("\n✓ Good performance!")
    elif mean_acc >= 0.50:
        print("\n⚠ Moderate performance - consider optimization")
    else:
        print("\n⚠ Low performance - need improvements")


if __name__ == "__main__":
    main()
