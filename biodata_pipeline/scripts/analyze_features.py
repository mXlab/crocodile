"""
Analyze extracted features to validate discriminability

This script loads processed features and generates a comprehensive analysis report
including feature importance, statistical tests, and visualizations.

Usage:
    python scripts/analyze_features.py [--features PATH] [--output DIR]

Examples:
    # Use default paths
    python scripts/analyze_features.py
    
    # Custom paths
    python scripts/analyze_features.py --features data/processed/all_features.csv --output reports/my_analysis
"""

import sys
from pathlib import Path
import argparse

# Fix Python path to find modules
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from modules.feature_analyzer import FeatureAnalyzer


def main():
    """Main analysis workflow"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Analyze extracted features for emotion discrimination'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='data/processed/all_features.csv',
        help='Path to features CSV file (default: data/processed/all_features.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/feature_analysis',
        help='Output directory for analysis reports (default: reports/feature_analysis)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top features to analyze (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    features_path = PROJECT_ROOT / args.features
    output_dir = PROJECT_ROOT / args.output
    
    print("="*80)
    print("FEATURE ANALYSIS")
    print("="*80)
    
    # Check if features file exists
    if not features_path.exists():
        print(f"\n❌ Features file not found: {features_path}")
        print("\nPlease run feature extraction first:")
        print("  python scripts/process_all_sessions.py")
        sys.exit(1)
    
    # Load features
    print(f"\nLoading features from: {features_path}")
    try:
        features = pd.read_csv(features_path)
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        sys.exit(1)
    
    print(f"✓ Loaded {len(features):,} samples")
    print(f"  Columns: {len(features.columns)}")
    print(f"  Emotions: {features['emotion'].nunique()}")
    
    # Quick validation
    feature_cols = [c for c in features.columns 
                   if c not in ['segment_id', 'session_id', 'emotion', 'duration', 
                               'feeling_it', 'feeling_it_ratio']]
    print(f"  Features: {len(feature_cols)}")
    
    if len(features) < 10:
        print("\n⚠ Warning: Very few samples ({len(features)}). Results may not be reliable.")
        print("  Consider collecting more data or relaxing filters.")
    
    # Check for missing emotions
    emotion_counts = features['emotion'].value_counts()
    print(f"\nEmotion distribution:")
    for emotion, count in emotion_counts.items():
        warning = " ⚠ LOW" if count < 5 else ""
        print(f"  {emotion:15s}: {count:3d} segments{warning}")
    
    # Initialize analyzer
    print("\n" + "="*80)
    print("INITIALIZING ANALYZER")
    print("="*80)
    
    try:
        analyzer = FeatureAnalyzer(features)
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Generate comprehensive report
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    print(f"Output directory: {output_dir}")
    
    try:
        analyzer.generate_report(str(output_dir))
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print quick summary of results
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    
    # Load and display top features
    importance_path = output_dir / 'feature_importance.csv'
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        
        print(f"\nTOP {min(args.top_n, len(importance))} FEATURES:")
        print("-" * 80)
        top_n = importance.head(args.top_n)
        for _, row in top_n.iterrows():
            print(f"  {row['rank']:2.0f}. {row['feature']:45s} {row['importance']:.4f}")
        
        n_for_80 = (importance['cumulative_importance'] <= 0.8).sum()
        print(f"\n✓ Top {n_for_80} features explain 80% of discriminative power")
    
    # Show most/least separable emotion pairs
    disc_path = output_dir / 'discriminability_matrix.csv'
    if disc_path.exists():
        disc_matrix = pd.read_csv(disc_path, index_col=0)
        emotions = disc_matrix.index.tolist()
        
        # Extract pairs
        pairs = []
        for i, emo1 in enumerate(emotions):
            for j, emo2 in enumerate(emotions):
                if i < j:
                    pairs.append({
                        'pair': f"{emo1} <-> {emo2}",
                        'distance': disc_matrix.loc[emo1, emo2]
                    })
        
        pairs_df = pd.DataFrame(pairs).sort_values('distance', ascending=False)
        
        print(f"\nMOST SEPARABLE EMOTION PAIRS (easy to classify):")
        print("-" * 80)
        for _, row in pairs_df.head(3).iterrows():
            print(f"  {row['pair']:25s}: {row['distance']:.3f}")
        
        print(f"\nLEAST SEPARABLE EMOTION PAIRS (may be confused):")
        print("-" * 80)
        for _, row in pairs_df.tail(3).iterrows():
            print(f"  {row['pair']:25s}: {row['distance']:.3f}")
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - feature_importance.csv          Feature rankings")
    print("  - statistical_tests.csv           ANOVA results")
    print("  - discriminability_matrix.csv     Emotion separability")
    print("  - feature_distributions.png       Distribution plots")
    print("  - correlation_matrix.png          Feature correlations")
    print("  - pca_emotions.png                2D visualization")
    print("  - feature_importance_bar.png      Importance chart")
    print("  - analysis_summary.txt            Text report")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review feature_importance.csv to select top features")
    print("2. Check discriminability_matrix.csv for confusable emotions")
    print("3. Examine visualizations for insights")
    print("4. Use top features to train classifier (Module 5)")
    
    print("\nRecommended features for classifier:")
    if importance_path.exists():
        print(f"  → Use top {n_for_80} features (explains 80% of variance)")
        print(f"  → Or top 20-25 features for balanced performance")


if __name__ == "__main__":
    main()
