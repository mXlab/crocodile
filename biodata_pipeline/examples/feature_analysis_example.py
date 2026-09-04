"""
Example: Feature Analysis Workflow

This example demonstrates the complete feature analysis workflow,
from loading data to selecting features for the classifier.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from modules.feature_analyzer import FeatureAnalyzer


def example_basic_usage():
    """Example 1: Basic usage - load and analyze"""
    
    print("="*80)
    print("EXAMPLE 1: Basic Feature Analysis")
    print("="*80)
    
    # Load features
    features = pd.read_csv('data/processed/all_features.csv')
    
    print(f"\nLoaded {len(features)} samples")
    print(f"Emotions: {features['emotion'].nunique()}")
    
    # Create analyzer
    analyzer = FeatureAnalyzer(features)
    
    # Generate complete report
    analyzer.generate_report('reports/feature_analysis')
    
    print("\n✓ Report generated in reports/feature_analysis/")


def example_selective_analysis():
    """Example 2: Run specific analyses only"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Selective Analysis")
    print("="*80)
    
    features = pd.read_csv('data/processed/all_features.csv')
    analyzer = FeatureAnalyzer(features)
    
    # 1. Feature importance only
    print("\n1. Computing feature importance...")
    importance = analyzer.compute_feature_importance(method='random_forest')
    print(f"\nTop 10 features:")
    print(importance.head(10)[['rank', 'feature', 'importance']])
    
    # 2. Statistical tests only
    print("\n2. Running statistical tests...")
    stats = analyzer.statistical_tests()
    significant = stats[stats['significant']]
    print(f"\nFound {len(significant)} significant features (p < 0.05)")
    
    # 3. Discriminability matrix only
    print("\n3. Computing discriminability...")
    disc_matrix = analyzer.discriminability_matrix()
    print("\nDiscriminability matrix:")
    print(disc_matrix)


def example_feature_selection():
    """Example 3: Feature selection for classifier"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Feature Selection for Classifier")
    print("="*80)
    
    features = pd.read_csv('data/processed/all_features.csv')
    analyzer = FeatureAnalyzer(features)
    
    # Strategy 1: Top N by importance
    print("\n--- Strategy 1: Top N Features ---")
    importance = analyzer.compute_feature_importance()
    
    top_20 = importance.head(20)['feature'].tolist()
    print(f"\nTop 20 features:")
    for i, feat in enumerate(top_20, 1):
        print(f"  {i:2d}. {feat}")
    
    # Strategy 2: Cumulative importance threshold
    print("\n--- Strategy 2: 80% Cumulative Importance ---")
    n_for_80 = (importance['cumulative_importance'] <= 0.8).sum()
    top_80pct = importance.head(n_for_80)['feature'].tolist()
    print(f"\nTop {n_for_80} features explain 80% of importance")
    
    # Strategy 3: Significant + important
    print("\n--- Strategy 3: Significant AND Important ---")
    stats = analyzer.statistical_tests()
    
    top_30_features = importance.head(30)['feature'].tolist()
    significant_features = stats[stats['significant']]['feature'].tolist()
    
    selected = list(set(top_30_features) & set(significant_features))
    print(f"\nSelected {len(selected)} features that are:")
    print("  - In top 30 by importance")
    print("  - Statistically significant (p < 0.05)")
    
    # Save selected features
    print("\n--- Saving Selected Features ---")
    Path('configs').mkdir(exist_ok=True)
    
    with open('configs/selected_features.txt', 'w') as f:
        for feat in selected:
            f.write(f"{feat}\n")
    
    print(f"✓ Saved {len(selected)} features to configs/selected_features.txt")


def example_pairwise_analysis():
    """Example 4: Analyze specific emotion pairs"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Pairwise Emotion Analysis")
    print("="*80)
    
    features = pd.read_csv('data/processed/all_features.csv')
    analyzer = FeatureAnalyzer(features)
    
    # Find most confusable emotions
    disc_matrix = analyzer.discriminability_matrix()
    
    emotions = disc_matrix.index.tolist()
    pairs = []
    for i, emo1 in enumerate(emotions):
        for j, emo2 in enumerate(emotions):
            if i < j:
                pairs.append({
                    'emotion_1': emo1,
                    'emotion_2': emo2,
                    'distance': disc_matrix.loc[emo1, emo2]
                })
    
    pairs_df = pd.DataFrame(pairs).sort_values('distance')
    
    # Analyze most confusable pair
    most_confusable = pairs_df.iloc[0]
    emo1, emo2 = most_confusable['emotion_1'], most_confusable['emotion_2']
    
    print(f"\nMost confusable pair: {emo1} <-> {emo2} (distance: {most_confusable['distance']:.3f})")
    print(f"\nFinding features that best separate them...")
    
    pairwise = analyzer.pairwise_feature_importance(emo1, emo2, top_n=10)
    
    print(f"\nTop 10 discriminative features:")
    print(pairwise[['feature', 'cohens_d', 'p_value']])


def example_compare_methods():
    """Example 5: Compare feature importance methods"""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Compare Importance Methods")
    print("="*80)
    
    features = pd.read_csv('data/processed/all_features.csv')
    analyzer = FeatureAnalyzer(features)
    
    # Compute importance with different methods
    print("\nComputing importance with 3 methods...")
    
    rf_importance = analyzer.compute_feature_importance(method='random_forest')
    mi_importance = analyzer.compute_feature_importance(method='mutual_info')
    anova_importance = analyzer.compute_feature_importance(method='anova')
    
    # Get top 20 from each
    rf_top = set(rf_importance.head(20)['feature'])
    mi_top = set(mi_importance.head(20)['feature'])
    anova_top = set(anova_importance.head(20)['feature'])
    
    # Find consensus
    consensus = rf_top & mi_top & anova_top
    rf_only = rf_top - mi_top - anova_top
    mi_only = mi_top - rf_top - anova_top
    
    print(f"\nTop 20 features by method:")
    print(f"  Random Forest: {len(rf_top)}")
    print(f"  Mutual Info:   {len(mi_top)}")
    print(f"  ANOVA:         {len(anova_top)}")
    
    print(f"\nConsensus (all 3 methods): {len(consensus)} features")
    if consensus:
        print("Features that rank high in all methods:")
        for feat in sorted(consensus):
            print(f"  - {feat}")
    
    print(f"\nRandom Forest unique: {len(rf_only)}")
    print(f"Mutual Info unique:   {len(mi_only)}")


def example_visualizations():
    """Example 6: Generate specific visualizations"""
    
    print("\n" + "="*80)
    print("EXAMPLE 6: Custom Visualizations")
    print("="*80)
    
    features = pd.read_csv('data/processed/all_features.csv')
    analyzer = FeatureAnalyzer(features)
    
    output_dir = Path('reports/custom_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # 1. Feature distributions
    print("  1. Feature distributions...")
    analyzer.plot_feature_distributions(
        top_n=12,
        output_path=output_dir / 'distributions_top12.png'
    )
    
    # 2. Correlation matrix
    print("  2. Correlation matrix...")
    analyzer.plot_correlation_matrix(
        top_n=25,
        output_path=output_dir / 'correlations_top25.png'
    )
    
    # 3. PCA 2D
    print("  3. PCA 2D...")
    analyzer.plot_pca(
        n_components=2,
        output_path=output_dir / 'pca_2d.png'
    )
    
    # 4. PCA 3D
    print("  4. PCA 3D...")
    analyzer.plot_pca(
        n_components=3,
        output_path=output_dir / 'pca_3d.png'
    )
    
    # 5. Importance bar chart
    print("  5. Importance bar chart...")
    analyzer.plot_feature_importance_bar(
        top_n=25,
        output_path=output_dir / 'importance_top25.png'
    )
    
    print(f"\n✓ Visualizations saved to {output_dir}/")


def main():
    """Run all examples"""
    
    print("\n" + "="*80)
    print("FEATURE ANALYZER - EXAMPLE WORKFLOWS")
    print("="*80)
    
    # Check if data exists
    if not Path('data/processed/all_features.csv').exists():
        print("\n❌ Error: data/processed/all_features.csv not found")
        print("\nPlease run feature extraction first:")
        print("  python scripts/process_all_sessions.py")
        return
    
    try:
        # Run examples
        example_basic_usage()
        example_selective_analysis()
        example_feature_selection()
        example_pairwise_analysis()
        example_compare_methods()
        example_visualizations()
        
        print("\n" + "="*80)
        print("✓ ALL EXAMPLES COMPLETED")
        print("="*80)
        print("\nNext steps:")
        print("  1. Review reports/feature_analysis/")
        print("  2. Load configs/selected_features.txt for classifier")
        print("  3. Proceed to Module 5: Classifier training")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
