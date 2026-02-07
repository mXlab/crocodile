# Feature Analyzer - Usage Guide

## Overview

The Feature Analyzer validates which of your extracted features effectively discriminate between emotions. This is a critical step before building a classifier.

**Key outputs:**
- Feature importance rankings
- Statistical significance tests
- Emotion discriminability analysis
- Comprehensive visualizations

---

## Quick Start

```python
import pandas as pd
from modules.feature_analyzer import FeatureAnalyzer

# Load features
features = pd.read_csv('data/processed/all_features.csv')

# Create analyzer
analyzer = FeatureAnalyzer(features)

# Generate complete report
analyzer.generate_report('reports/feature_analysis')
```

**Or use the script:**
```bash
python scripts/analyze_features.py
```

---

## Core Functionality

### 1. Feature Importance

Ranks features by discriminative power using Random Forest importance.

```python
# Compute importance
importance = analyzer.compute_feature_importance(method='random_forest')

# View top features
print(importance.head(20))

# How many features for 80% importance?
n_for_80 = (importance['cumulative_importance'] <= 0.8).sum()
print(f"Top {n_for_80} features explain 80% of importance")
```

**Methods available:**
- `'random_forest'`: Random Forest feature importance (recommended)
- `'mutual_info'`: Mutual information with labels
- `'anova'`: ANOVA F-statistic (univariate)

**Output:**
```
rank  feature                          importance  cumulative_importance
   1  respiratory.resp_sigh_frequency      0.0876                 0.0876
   2  respiratory.resp_variability_cv      0.0654                 0.1530
   3  eda.scr_event_clustering             0.0589                 0.2119
  ...
```

---

### 2. Statistical Tests

Tests which features differ significantly across emotions.

```python
# Run ANOVA tests
stats_results = analyzer.statistical_tests(test='anova', alpha=0.05)

# View significant features
significant = stats_results[stats_results['significant']]
print(f"Found {len(significant)} significant features")

# Top features by effect size
top_sig = significant.nlargest(10, 'effect_size')
print(top_sig[['feature', 'p_value', 'effect_size']])
```

**Outputs:**
- `f_statistic`: ANOVA F-statistic
- `p_value`: Statistical significance
- `effect_size`: Eta-squared (proportion of variance explained)
- `significant`: Boolean (p < α)

**Interpretation:**
- **p < 0.05**: Feature differs significantly across emotions
- **Effect size > 0.14**: Large effect (Cohen's guidelines)
- **Effect size 0.06-0.14**: Medium effect
- **Effect size < 0.06**: Small effect

---

### 3. Discriminability Matrix

Measures how well emotion pairs can be separated in feature space.

```python
# Compute pairwise distances
disc_matrix = analyzer.discriminability_matrix(metric='euclidean')

# View matrix
print(disc_matrix)

# Find confusable emotions
# (Emotions with low distance are hard to separate)
```

**Example output:**
```
         joy    sad   fear   calm
joy     0.00   3.45   2.87   3.12
sad     3.45   0.00   2.21   1.98
fear    2.87   2.21   0.00   3.54
calm    3.12   1.98   3.54   0.00
```

**Interpretation:**
- **High distance (>3.0)**: Emotions well-separated, easy to classify
- **Medium distance (2.0-3.0)**: Moderate separation
- **Low distance (<2.0)**: Confusable emotions, may need more/better features

**Most separable:** `fear <-> calm` (3.54)  
**Least separable:** `sad <-> calm` (1.98) ← May be confused!

---

### 4. Pairwise Feature Importance

Find features that best discriminate between two specific emotions.

```python
# Compare two emotions
pairwise = analyzer.pairwise_feature_importance('joy', 'sad', top_n=10)

# Shows which features differ most between joy and sad
print(pairwise[['feature', 'cohens_d', 'p_value']])
```

**Use case:** If classifier confuses two emotions, identify discriminating features.

---

## Visualizations

### Feature Distributions

Violin plots showing feature value distributions per emotion.

```python
analyzer.plot_feature_distributions(
    top_n=15,  # Number of features to plot
    output_path='reports/feature_distributions.png'
)
```

**What to look for:**
- **Well-separated distributions** = good discriminator
- **Overlapping distributions** = poor discriminator
- **Outliers** = potential data quality issues

---

### Correlation Matrix

Identifies redundant features (high correlation).

```python
analyzer.plot_correlation_matrix(
    top_n=20,
    output_path='reports/correlation_matrix.png'
)
```

**What to look for:**
- **High correlation (|r| > 0.8)**: Features are redundant, pick one
- **Low correlation**: Features provide independent information

**Example:** If `eda.scl_mean` and `eda.scl_median` correlate at r=0.95, you only need one.

---

### PCA Visualization

Projects high-dimensional features to 2D for visualization.

```python
# 2D projection
analyzer.plot_pca(n_components=2, output_path='reports/pca_2d.png')

# 3D projection
analyzer.plot_pca(n_components=3, output_path='reports/pca_3d.png')
```

**What to look for:**
- **Distinct clusters per emotion** = features work well
- **Overlapping clouds** = emotions not well-separated
- **Explained variance**: Higher is better (look at axis labels)

---

### Feature Importance Bar Chart

Horizontal bar chart of top features.

```python
analyzer.plot_feature_importance_bar(
    top_n=20,
    output_path='reports/importance_bar.png'
)
```

---

## Complete Report Generation

Generate all analyses and visualizations at once:

```python
analyzer.generate_report('reports/feature_analysis')
```

**Creates:**
```
reports/feature_analysis/
├── feature_importance.csv          # Ranked features
├── statistical_tests.csv           # ANOVA results
├── discriminability_matrix.csv     # Pairwise distances
├── feature_distributions.png       # Violin plots
├── correlation_matrix.png          # Correlations
├── pca_emotions.png                # 2D PCA
├── feature_importance_bar.png      # Bar chart
└── analysis_summary.txt            # Text report
```

---

## Interpreting Results

### Example Analysis Workflow

**1. Load and check data**
```python
features = pd.read_csv('data/processed/all_features.csv')
print(f"Samples: {len(features)}")
print(features['emotion'].value_counts())
```

**2. Generate report**
```python
analyzer = FeatureAnalyzer(features)
analyzer.generate_report('reports/feature_analysis')
```

**3. Review feature importance**
```python
importance = pd.read_csv('reports/feature_analysis/feature_importance.csv')

# Top 20 features
top_20 = importance.head(20)['feature'].tolist()
print(top_20)

# How many for 80% importance?
n_80 = (importance['cumulative_importance'] <= 0.8).sum()
print(f"Use top {n_80} features")
```

**4. Check for confusable emotions**
```python
disc = pd.read_csv('reports/feature_analysis/discriminability_matrix.csv', index_col=0)

# Find pairs with distance < 2.0
for i, emo1 in enumerate(disc.index):
    for j, emo2 in enumerate(disc.columns):
        if i < j and disc.loc[emo1, emo2] < 2.0:
            print(f"⚠ {emo1} <-> {emo2}: {disc.loc[emo1, emo2]:.3f}")
```

**5. Review statistical significance**
```python
stats = pd.read_csv('reports/feature_analysis/statistical_tests.csv')

# How many significant features?
n_sig = stats['significant'].sum()
print(f"{n_sig}/{len(stats)} features are significant")

# Top by effect size
top_effect = stats[stats['significant']].nlargest(10, 'effect_size')
print(top_effect[['feature', 'effect_size', 'p_value']])
```

**6. Make decisions for classifier**
```python
# Select features
selected_features = importance.head(25)['feature'].tolist()

# Save for classifier
with open('configs/selected_features.txt', 'w') as f:
    for feat in selected_features:
        f.write(f"{feat}\n")
```

---

## Common Patterns

### Pattern 1: Too Few Samples per Emotion

**Symptom:** Some emotions have < 5 segments

**Solution:**
- Collect more data for underrepresented emotions
- Merge similar emotions (e.g., combine 'tir' and 'rlx')
- Exclude rare emotions from initial classifier

### Pattern 2: No Significant Features

**Symptom:** Statistical tests show p > 0.05 for most features

**Possible causes:**
- Too few samples (need more data)
- Features not discriminative (wrong features)
- Emotions too similar (need different emotions)

**Solutions:**
- Increase sample size
- Check feature extraction for errors
- Review emotion definitions

### Pattern 3: High Feature Correlation

**Symptom:** Correlation matrix shows many r > 0.8

**Impact:** Redundant features (not harmful, just inefficient)

**Solution:**
- Remove highly correlated features
- Keep the one with highest importance
- Or use PCA to reduce dimensions

### Pattern 4: Emotions Not Separable

**Symptom:** PCA shows overlapping clusters

**Possible causes:**
- Features don't capture emotion differences
- Emotions are physiologically similar
- Data quality issues

**Solutions:**
- Check feature extraction parameters
- Collect longer/better quality segments
- Consider different emotion categories

---

## Advanced Usage

### Custom Feature Selection

```python
# Combine multiple criteria
importance = analyzer.compute_feature_importance()
stats = analyzer.statistical_tests()

# Select features that are:
# 1. In top 30 by importance
# 2. Statistically significant
# 3. Not highly correlated

top_30_features = importance.head(30)['feature'].tolist()
significant_features = stats[stats['significant']]['feature'].tolist()

# Intersection
selected = list(set(top_30_features) & set(significant_features))
print(f"Selected {len(selected)} features")
```

### Compare Feature Selection Methods

```python
# Compare different methods
rf_importance = analyzer.compute_feature_importance(method='random_forest')
mi_importance = analyzer.compute_feature_importance(method='mutual_info')
anova_importance = analyzer.compute_feature_importance(method='anova')

# Features that rank high in all methods
rf_top = set(rf_importance.head(20)['feature'])
mi_top = set(mi_importance.head(20)['feature'])
anova_top = set(anova_importance.head(20)['feature'])

consensus = rf_top & mi_top & anova_top
print(f"Consensus features: {consensus}")
```

### Analyze Specific Emotion Pairs

```python
# Focus on confusable emotions
confusable_pairs = [('tir', 'rlx'), ('joy', 'aro')]

for emo1, emo2 in confusable_pairs:
    print(f"\nAnalyzing {emo1} vs {emo2}:")
    pairwise = analyzer.pairwise_feature_importance(emo1, emo2, top_n=5)
    print(pairwise[['feature', 'cohens_d']])
```

---

## Troubleshooting

### Issue: "FeatureAnalyzer initialized: Samples: 0"

**Cause:** Empty features DataFrame

**Fix:** Check that `all_features.csv` has data

### Issue: "ValueError: Found array with 0 sample(s)"

**Cause:** Emotion category with no samples after filtering

**Fix:** Check emotion distribution, exclude rare emotions

### Issue: Plots look wrong / overlapping labels

**Cause:** Too many emotions or features for plot size

**Fix:** Reduce `top_n` parameter or increase figure size

### Issue: "All features have importance 0"

**Cause:** Random Forest couldn't train (all samples same class, or other issue)

**Fix:**
- Check you have multiple emotion classes
- Check features have variation (not all constant)

---

## Next Steps

After feature analysis:

1. **Select top features** (top 20-30 or 80% cumulative importance)
2. **Review confusable emotions** (consider collecting more data)
3. **Proceed to Module 5: Classifier** (train with selected features)

---

## API Reference

See `modules/feature_analyzer.py` for complete API documentation.

**Key methods:**
- `compute_feature_importance(method, n_estimators)`
- `statistical_tests(test, alpha)`
- `discriminability_matrix(metric)`
- `pairwise_feature_importance(emotion1, emotion2, top_n)`
- `plot_feature_distributions(top_n, output_path)`
- `plot_correlation_matrix(top_n, output_path)`
- `plot_pca(n_components, output_path)`
- `generate_report(output_dir)`

---

## Examples

See `examples/feature_analysis_example.py` for complete workflow examples.
