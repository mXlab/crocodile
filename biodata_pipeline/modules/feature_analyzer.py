"""
Crocodile Biodata Pipeline - Feature Analyzer Module

Analyzes feature discriminability for emotion classification.
Provides comprehensive analysis of which features best separate emotions.

Key Classes:
    FeatureAnalyzer: Main analyzer with statistical tests and visualizations

Key Methods:
    - compute_feature_importance(): Rank features by discriminative power
    - statistical_tests(): ANOVA, t-tests between emotions
    - discriminability_matrix(): Pairwise emotion separability
    - plot_*(): Various visualizations
    - generate_report(): Create comprehensive analysis report

Usage:
    >>> import pandas as pd
    >>> from modules.feature_analyzer import FeatureAnalyzer
    >>> 
    >>> features = pd.read_csv('data/processed/all_features.csv')
    >>> analyzer = FeatureAnalyzer(features)
    >>> analyzer.generate_report('reports/feature_analysis')

See docs/guides/FeatureAnalyzer_Usage_Guide.md for detailed examples.
See docs/implementation/Module4_Implementation_Summary.md for design decisions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, f_classif
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """
    Analyze physiological features for emotion discriminability.
    
    This class provides comprehensive analysis tools to validate which
    features effectively discriminate between different emotional states.
    
    Attributes
    ----------
    data : pd.DataFrame
        Complete feature dataset with metadata and features
    feature_cols : list
        Names of feature columns (excludes metadata)
    emotions : list
        Unique emotion labels in dataset
    metadata_cols : list
        Names of metadata columns
    
    Examples
    --------
    >>> features = pd.read_csv('all_features.csv')
    >>> analyzer = FeatureAnalyzer(features)
    >>> 
    >>> # Compute feature importance
    >>> importance = analyzer.compute_feature_importance()
    >>> print(importance.head(10))
    >>> 
    >>> # Generate full report
    >>> analyzer.generate_report('reports/feature_analysis')
    """
    
    def __init__(self, features_df: pd.DataFrame):
        """
        Initialize Feature Analyzer.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            Features extracted from segments.
            Required columns: 'emotion', plus feature columns
            Optional: 'segment_id', 'session_id', 'duration', 'feeling_it_ratio'
        """
        self.data = features_df.copy()
        
        # Identify metadata vs feature columns
        self.metadata_cols = [
            'segment_id', 'session_id', 'emotion', 
            'duration', 'feeling_it', 'feeling_it_ratio',
            'window_id', 'parent_segment_id', 'window_mode', 'emotion_purity'
        ]
        
        self.feature_cols = [col for col in features_df.columns 
                            if col not in self.metadata_cols]
        
        self.emotions = sorted(features_df['emotion'].unique())
        
        print(f"FeatureAnalyzer initialized:")
        print(f"  Samples: {len(features_df)}")
        print(f"  Emotions: {len(self.emotions)} - {self.emotions}")
        print(f"  Features: {len(self.feature_cols)}")
    
    # ========================================================================
    # 1. FEATURE IMPORTANCE
    # ========================================================================
    
    def compute_feature_importance(self, method='random_forest', 
                                   n_estimators=100) -> pd.DataFrame:
        """
        Rank features by discriminative power.
        
        Parameters
        ----------
        method : str, default='random_forest'
            Method for computing importance:
            - 'random_forest': Random Forest feature importance
            - 'mutual_info': Mutual information with emotion labels
            - 'anova': ANOVA F-statistic (univariate)
        n_estimators : int, default=100
            Number of trees for random forest
            
        Returns
        -------
        importance_df : pd.DataFrame
            Features ranked by importance with columns:
            - feature: Feature name
            - importance: Importance score
            - rank: Rank (1 = most important)
            - cumulative_importance: Cumulative sum of importance
        
        Examples
        --------
        >>> importance = analyzer.compute_feature_importance()
        >>> top_10 = importance.head(10)
        >>> print(top_10[['rank', 'feature', 'importance']])
        """
        print(f"\nComputing feature importance ({method})...")
        
        X = self.data[self.feature_cols].values
        y = self.data['emotion'].values
        
        if method == 'random_forest':
            rf = RandomForestClassifier(
                n_estimators=n_estimators, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            rf.fit(X, y)
            importances = rf.feature_importances_
            
        elif method == 'mutual_info':
            importances = mutual_info_classif(X, y, random_state=42)
            
        elif method == 'anova':
            f_scores, _ = f_classif(X, y)
            importances = f_scores / f_scores.sum()  # Normalize
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        importance_df['rank'] = range(1, len(importance_df) + 1)
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        
        # Normalize cumulative to 0-1 range
        importance_df['cumulative_importance'] /= importance_df['cumulative_importance'].max()
        
        print(f"✓ Feature importance computed")
        print(f"\nTop 10 features:")
        print(importance_df[['rank', 'feature', 'importance']].head(10).to_string(index=False))
        
        # How many features for 80% importance?
        n_for_80 = (importance_df['cumulative_importance'] <= 0.8).sum()
        print(f"\nTop {n_for_80} features explain 80% of importance")
        
        return importance_df
    
    # ========================================================================
    # 2. STATISTICAL TESTS
    # ========================================================================
    
    def statistical_tests(self, test='anova', alpha=0.05) -> pd.DataFrame:
        """
        Test which features differ significantly across emotions.
        
        Parameters
        ----------
        test : str, default='anova'
            Statistical test to use:
            - 'anova': One-way ANOVA (for multiple groups)
        alpha : float, default=0.05
            Significance level for hypothesis testing
            
        Returns
        -------
        results_df : pd.DataFrame
            Statistical test results per feature with columns:
            - feature: Feature name
            - f_statistic: F-statistic (ANOVA)
            - p_value: P-value
            - significant: Boolean (p < alpha)
            - effect_size: Eta-squared (proportion of variance explained)
        
        Examples
        --------
        >>> stats_results = analyzer.statistical_tests()
        >>> significant = stats_results[stats_results['significant']]
        >>> print(f"Found {len(significant)} significant features")
        """
        print(f"\nPerforming {test.upper()} tests...")
        
        results = []
        
        for feature in self.feature_cols:
            # Group feature values by emotion
            groups = [self.data[self.data['emotion'] == emotion][feature].values
                     for emotion in self.emotions]
            
            if test == 'anova':
                f_stat, p_value = stats.f_oneway(*groups)
                
                # Calculate effect size (eta-squared)
                # eta^2 = SS_between / SS_total
                grand_mean = self.data[feature].mean()
                ss_total = np.sum((self.data[feature] - grand_mean) ** 2)
                ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 
                               for g in groups)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                results.append({
                    'feature': feature,
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'effect_size': eta_squared
                })
        
        results_df = pd.DataFrame(results).sort_values('p_value')
        
        n_significant = results_df['significant'].sum()
        print(f"✓ {n_significant}/{len(results_df)} features significantly different (α={alpha})")
        
        # Show top significant features
        if n_significant > 0:
            print(f"\nTop 10 significant features (by effect size):")
            top_sig = results_df[results_df['significant']].nlargest(10, 'effect_size')
            print(top_sig[['feature', 'f_statistic', 'p_value', 'effect_size']].to_string(index=False))
        
        return results_df
    
    # ========================================================================
    # 3. PAIRWISE EMOTION ANALYSIS
    # ========================================================================
    
    def discriminability_matrix(self, metric='euclidean') -> pd.DataFrame:
        """
        Compute pairwise emotion separability in feature space.
        
        Calculates distances between emotion centroids (mean feature vectors).
        Higher distances indicate emotions that are easier to discriminate.
        
        Parameters
        ----------
        metric : str, default='euclidean'
            Distance metric to use:
            - 'euclidean': Euclidean distance
            - 'cosine': Cosine distance
            - 'manhattan': Manhattan distance
            
        Returns
        -------
        matrix_df : pd.DataFrame
            Symmetric matrix of pairwise distances between emotions.
            Rows and columns are emotion labels.
        
        Examples
        --------
        >>> disc_matrix = analyzer.discriminability_matrix()
        >>> print(disc_matrix)
        >>> # Find most confusable emotions (lowest distances)
        """
        print(f"\nComputing discriminability matrix ({metric})...")
        
        # Compute emotion centroids (mean feature vector per emotion)
        centroids = []
        for emotion in self.emotions:
            emotion_data = self.data[self.data['emotion'] == emotion][self.feature_cols]
            centroid = emotion_data.mean().values
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Compute pairwise distances
        distances = pairwise_distances(centroids, metric=metric)
        
        # Create DataFrame
        matrix_df = pd.DataFrame(
            distances,
            index=self.emotions,
            columns=self.emotions
        )
        
        print(f"✓ Discriminability matrix computed")
        
        # Find most and least separable pairs
        pairs = []
        for i, emo1 in enumerate(self.emotions):
            for j, emo2 in enumerate(self.emotions):
                if i < j:  # Upper triangle only
                    pairs.append({
                        'emotion_1': emo1,
                        'emotion_2': emo2,
                        'distance': distances[i, j]
                    })
        
        pairs_df = pd.DataFrame(pairs)
        
        print(f"\nMost separable pairs (largest distances):")
        print(pairs_df.nlargest(5, 'distance')[['emotion_1', 'emotion_2', 'distance']].to_string(index=False))
        
        print(f"\nLeast separable pairs (most confusable):")
        print(pairs_df.nsmallest(5, 'distance')[['emotion_1', 'emotion_2', 'distance']].to_string(index=False))
        
        return matrix_df
    
    def pairwise_feature_importance(self, emotion1: str, emotion2: str, 
                                    top_n=10) -> pd.DataFrame:
        """
        Find features that best discriminate between two specific emotions.
        
        Parameters
        ----------
        emotion1, emotion2 : str
            Emotion labels to compare
        top_n : int, default=10
            Number of top features to return
            
        Returns
        -------
        pairwise_importance : pd.DataFrame
            Top features for discriminating these two emotions
        """
        print(f"\nComputing pairwise importance: {emotion1} vs {emotion2}")
        
        # Filter data for these two emotions
        mask = self.data['emotion'].isin([emotion1, emotion2])
        data_pair = self.data[mask].copy()
        
        # Compute t-statistic for each feature
        results = []
        for feature in self.feature_cols:
            group1 = data_pair[data_pair['emotion'] == emotion1][feature].values
            group2 = data_pair[data_pair['emotion'] == emotion2][feature].values
            
            if len(group1) > 1 and len(group2) > 1:
                t_stat, p_value = stats.ttest_ind(group1, group2)
                
                # Cohen's d effect size
                mean_diff = np.mean(group1) - np.mean(group2)
                pooled_std = np.sqrt(
                    ((len(group1)-1)*np.var(group1) + (len(group2)-1)*np.var(group2)) /
                    (len(group1) + len(group2) - 2)
                )
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                results.append({
                    'feature': feature,
                    't_statistic': abs(t_stat),
                    'p_value': p_value,
                    'cohens_d': abs(cohens_d),
                    'mean_diff': mean_diff
                })
        
        results_df = pd.DataFrame(results).sort_values('cohens_d', ascending=False)
        
        print(f"✓ Top {top_n} discriminative features:")
        print(results_df.head(top_n)[['feature', 'cohens_d', 'p_value']].to_string(index=False))
        
        return results_df.head(top_n)
    
    # ========================================================================
    # 4. VISUALIZATIONS
    # ========================================================================
    
    def plot_feature_distributions(self, top_n=15, output_path=None):
        """
        Plot distributions of top features by emotion.
        
        Creates violin plots showing feature value distributions for each emotion.
        
        Parameters
        ----------
        top_n : int, default=15
            Number of top features to plot
        output_path : str or Path, optional
            Path to save figure. If None, displays interactively.
        """
        print(f"\nPlotting feature distributions (top {top_n})...")
        
        # Get top features
        importance = self.compute_feature_importance(method='random_forest')
        top_features = importance.head(top_n)['feature'].tolist()
        
        # Create subplots
        n_cols = 3
        n_rows = (top_n + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, feature in enumerate(top_features):
            ax = axes[idx]
            
            # Prepare data for violin plot
            data_to_plot = [self.data[self.data['emotion'] == emotion][feature].values
                           for emotion in self.emotions]
            
            # Violin plot
            parts = ax.violinplot(data_to_plot, positions=range(len(self.emotions)), 
                                 showmeans=True, showmedians=True)
            
            # Color by emotion
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.emotions)))
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(self.emotions)))
            ax.set_xticklabels(self.emotions, rotation=45, ha='right')
            ax.set_ylabel('Value')
            ax.set_title(feature, fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove empty subplots
        for idx in range(top_n, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'Top {top_n} Feature Distributions by Emotion', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_correlation_matrix(self, top_n=20, output_path=None):
        """
        Plot correlation matrix of top features.
        
        Identifies redundant features (high correlation) that could be removed.
        
        Parameters
        ----------
        top_n : int, default=20
            Number of top features to include
        output_path : str or Path, optional
            Path to save figure
        """
        print(f"\nPlotting correlation matrix (top {top_n})...")
        
        # Get top features
        importance = self.compute_feature_importance(method='random_forest')
        top_features = importance.head(top_n)['feature'].tolist()
        
        # Compute correlation
        corr_matrix = self.data[top_features].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8:  # Threshold for high correlation
                    high_corr_pairs.append({
                        'feature_1': top_features[i],
                        'feature_2': top_features[j],
                        'correlation': corr_val
                    })
        
        if high_corr_pairs:
            print(f"\n⚠ Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8):")
            for pair in high_corr_pairs[:5]:
                print(f"  {pair['feature_1']:40s} <-> {pair['feature_2']:40s} (r={pair['correlation']:.3f})")
        else:
            print(f"\n✓ No highly correlated features found (good!)")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, 
                   cbar_kws={"shrink": 0.8, "label": "Correlation"})
        
        plt.title(f'Feature Correlation Matrix (Top {top_n})', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_pca(self, n_components=2, output_path=None):
        """
        Plot PCA visualization of emotions in feature space.
        
        Projects high-dimensional feature space to 2D for visualization.
        
        Parameters
        ----------
        n_components : int, default=2
            Number of principal components (2 or 3)
        output_path : str or Path, optional
            Path to save figure
        """
        print(f"\nPlotting PCA ({n_components}D)...")
        
        # Standardize features
        X = self.data[self.feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot
        if n_components == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.emotions)))
            for emotion, color in zip(self.emotions, colors):
                mask = self.data['emotion'] == emotion
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          label=emotion, alpha=0.6, s=100, color=color,
                          edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                         fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                         fontweight='bold')
            ax.set_title('PCA: Emotions in Feature Space', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
        elif n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.emotions)))
            for emotion, color in zip(self.emotions, colors):
                mask = self.data['emotion'] == emotion
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                          label=emotion, alpha=0.6, s=100, color=color,
                          edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontweight='bold')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontweight='bold')
            ax.set_title('PCA: Emotions in Feature Space (3D)', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
        
        # Print explained variance
        total_var = pca.explained_variance_ratio_.sum()
        print(f"✓ {n_components} components explain {total_var:.1%} of variance")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance_bar(self, top_n=20, output_path=None):
        """
        Plot horizontal bar chart of feature importance.
        
        Parameters
        ----------
        top_n : int, default=20
            Number of top features to plot
        output_path : str or Path, optional
            Path to save figure
        """
        print(f"\nPlotting feature importance bar chart (top {top_n})...")
        
        importance = self.compute_feature_importance(method='random_forest')
        top_importance = importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        # Horizontal bar chart
        y_pos = np.arange(len(top_importance))
        ax.barh(y_pos, top_importance['importance'].values, color='steelblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_importance['feature'].values)
        ax.invert_yaxis()  # Top feature at top
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title(f'Top {top_n} Features by Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    # ========================================================================
    # 5. COMPREHENSIVE REPORT GENERATION
    # ========================================================================
    
    def generate_report(self, output_dir='reports/feature_analysis'):
        """
        Generate comprehensive analysis report with all visualizations and statistics.
        
        Creates:
        - feature_importance.csv: Ranked features
        - statistical_tests.csv: ANOVA results
        - discriminability_matrix.csv: Pairwise emotion distances
        - feature_distributions.png: Violin plots
        - correlation_matrix.png: Feature correlations
        - pca_emotions.png: 2D PCA visualization
        - feature_importance_bar.png: Bar chart
        - analysis_summary.txt: Text summary
        
        Parameters
        ----------
        output_dir : str or Path, default='reports/feature_analysis'
            Directory to save all outputs
        
        Examples
        --------
        >>> analyzer.generate_report('reports/feature_analysis')
        >>> # Check reports/feature_analysis/ for all outputs
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE FEATURE ANALYSIS REPORT")
        print("="*80)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature importance
        print("\n[1/8] Computing feature importance...")
        importance = self.compute_feature_importance(method='random_forest')
        importance.to_csv(output_path / 'feature_importance.csv', index=False)
        
        # 2. Statistical tests
        print("\n[2/8] Running statistical tests...")
        stats_results = self.statistical_tests(test='anova')
        stats_results.to_csv(output_path / 'statistical_tests.csv', index=False)
        
        # 3. Discriminability matrix
        print("\n[3/8] Computing discriminability matrix...")
        disc_matrix = self.discriminability_matrix(metric='euclidean')
        disc_matrix.to_csv(output_path / 'discriminability_matrix.csv')
        
        # 4. Visualizations
        print("\n[4/8] Generating feature distributions plot...")
        self.plot_feature_distributions(
            top_n=15,
            output_path=output_path / 'feature_distributions.png'
        )
        
        print("\n[5/8] Generating correlation matrix...")
        self.plot_correlation_matrix(
            top_n=20,
            output_path=output_path / 'correlation_matrix.png'
        )
        
        print("\n[6/8] Generating PCA visualization...")
        self.plot_pca(
            n_components=2,
            output_path=output_path / 'pca_emotions.png'
        )
        
        print("\n[7/8] Generating importance bar chart...")
        self.plot_feature_importance_bar(
            top_n=20,
            output_path=output_path / 'feature_importance_bar.png'
        )
        
        # 5. Generate text summary
        print("\n[8/8] Writing analysis summary...")
        self._write_summary_report(output_path, importance, stats_results, disc_matrix)
        
        print("\n" + "="*80)
        print("✓ REPORT GENERATION COMPLETE")
        print("="*80)
        print(f"\nOutput directory: {output_path.absolute()}")
        print(f"\nGenerated files:")
        for file in sorted(output_path.glob('*')):
            size_kb = file.stat().st_size / 1024
            print(f"  - {file.name:40s} ({size_kb:6.1f} KB)")
    
    def _write_summary_report(self, output_path: Path, 
                            importance: pd.DataFrame,
                            stats_results: pd.DataFrame,
                            disc_matrix: pd.DataFrame):
        """Write text summary of analysis."""
        
        with open(output_path / 'analysis_summary.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("FEATURE ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Dataset summary
            f.write("DATASET SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total samples: {len(self.data)}\n")
            f.write(f"Total features: {len(self.feature_cols)}\n")
            f.write(f"Emotions: {len(self.emotions)}\n\n")
            
            f.write("Emotion distribution:\n")
            for emotion, count in self.data['emotion'].value_counts().items():
                f.write(f"  {emotion:15s}: {count:3d} segments\n")
            
            # Feature importance
            f.write("\n" + "="*80 + "\n")
            f.write("TOP 20 FEATURES (Random Forest Importance)\n")
            f.write("="*80 + "\n")
            top_20 = importance.head(20)
            f.write(top_20[['rank', 'feature', 'importance']].to_string(index=False))
            f.write("\n\n")
            
            n_for_80 = (importance['cumulative_importance'] <= 0.8).sum()
            f.write(f"Note: Top {n_for_80} features explain 80% of importance\n")
            
            # Statistical significance
            f.write("\n" + "="*80 + "\n")
            f.write("STATISTICAL SIGNIFICANCE (ANOVA)\n")
            f.write("="*80 + "\n")
            n_sig = stats_results['significant'].sum()
            f.write(f"Significant features (p < 0.05): {n_sig}/{len(stats_results)}\n\n")
            
            f.write("Top 10 by effect size:\n")
            top_sig = stats_results[stats_results['significant']].nlargest(10, 'effect_size')
            f.write(top_sig[['feature', 'f_statistic', 'p_value', 'effect_size']].to_string(index=False))
            
            # Discriminability
            f.write("\n\n" + "="*80 + "\n")
            f.write("EMOTION DISCRIMINABILITY\n")
            f.write("="*80 + "\n")
            
            # Most/least separable pairs
            pairs = []
            for i, emo1 in enumerate(self.emotions):
                for j, emo2 in enumerate(self.emotions):
                    if i < j:
                        pairs.append({
                            'emotion_1': emo1,
                            'emotion_2': emo2,
                            'distance': disc_matrix.loc[emo1, emo2]
                        })
            pairs_df = pd.DataFrame(pairs)
            
            f.write("\nMost separable emotion pairs (easy to discriminate):\n")
            f.write(pairs_df.nlargest(5, 'distance').to_string(index=False))
            
            f.write("\n\nLeast separable emotion pairs (confusable):\n")
            f.write(pairs_df.nsmallest(5, 'distance').to_string(index=False))
            
            # Recommendations
            f.write("\n\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            # Recommend feature subset
            top_features_80 = importance[importance['cumulative_importance'] <= 0.8]['feature'].tolist()
            f.write(f"1. Use top {len(top_features_80)} features for classifier (80% importance)\n")
            
            # Check for underrepresented emotions
            emotion_counts = self.data['emotion'].value_counts()
            low_count_emotions = emotion_counts[emotion_counts < 5]
            if len(low_count_emotions) > 0:
                f.write(f"\n2. Collect more data for emotions with < 5 segments:\n")
                for emotion, count in low_count_emotions.items():
                    f.write(f"   - {emotion}: currently {count} segments\n")
            
            # Check for confusable pairs
            confusable = pairs_df.nsmallest(3, 'distance')
            f.write(f"\n3. Watch for confusion between:\n")
            for _, pair in confusable.iterrows():
                f.write(f"   - {pair['emotion_1']} <-> {pair['emotion_2']} (distance: {pair['distance']:.3f})\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Saved: {output_path / 'analysis_summary.txt'}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_and_analyze(features_path: str, output_dir: str = 'reports/feature_analysis'):
    """
    Convenience function to load features and generate full analysis report.
    
    Parameters
    ----------
    features_path : str
        Path to features CSV file
    output_dir : str, default='reports/feature_analysis'
        Directory for output files
    
    Examples
    --------
    >>> from modules.feature_analyzer import load_and_analyze
    >>> load_and_analyze('data/processed/all_features.csv')
    """
    import pandas as pd
    
    print(f"Loading features from: {features_path}")
    features = pd.read_csv(features_path)
    
    analyzer = FeatureAnalyzer(features)
    analyzer.generate_report(output_dir)
    
    return analyzer


if __name__ == "__main__":
    """
    Example usage when run as script.
    """
    print("Feature Analyzer Module")
    print("=" * 80)
    print("\nUsage:")
    print("  from modules.feature_analyzer import FeatureAnalyzer")
    print("  analyzer = FeatureAnalyzer(features_df)")
    print("  analyzer.generate_report('reports/feature_analysis')")
    print("\nOr use convenience function:")
    print("  from modules.feature_analyzer import load_and_analyze")
    print("  load_and_analyze('data/processed/all_features.csv')")
