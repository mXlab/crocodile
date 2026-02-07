# Module 4: Feature Analyzer - Implementation Summary

## Overview

**Module 4: Feature Analyzer** validates which physiological features effectively discriminate between emotions. This analysis is critical for selecting features to use in the classifier (Module 5).

**Status:** ✅ Complete and tested  
**Lines of Code:** ~850  
**Dependencies:** numpy, pandas, scipy, sklearn, matplotlib, seaborn

---

## Design Goals

### Primary Objectives

1. **Identify discriminative features** - Rank features by importance
2. **Validate statistical significance** - Test which features differ across emotions
3. **Measure emotion separability** - Quantify how well emotion pairs can be distinguished
4. **Provide actionable insights** - Generate visualizations and recommendations

### Non-Goals

- ❌ Build a classifier (that's Module 5)
- ❌ Optimize features (use existing 54 features from Module 3)
- ❌ Real-time analysis (offline batch processing only)

---

## Architecture

### Class Design

```python
class FeatureAnalyzer:
    """
    Main analyzer class with five functional groups:
    1. Feature Importance (Random Forest, Mutual Info, ANOVA)
    2. Statistical Tests (ANOVA, effect sizes)
    3. Pairwise Analysis (Discriminability, pairwise importance)
    4. Visualizations (6 different plots)
    5. Report Generation (Complete analysis)
    """
```

### Input/Output

**Input:**
- `features_df`: DataFrame with emotion-labeled feature vectors
  - Required columns: `emotion`, plus 54 feature columns
  - Optional: `segment_id`, `session_id`, `duration`, `feeling_it_ratio`

**Output:**
- CSV files: Rankings, statistics, matrices
- PNG files: Visualizations
- TXT file: Human-readable summary

---

## Core Algorithms

### 1. Feature Importance

**Method: Random Forest (Recommended)**

```python
rf = RandomForestClassifier(
    n_estimators=100,    # 100 trees
    max_depth=10,        # Prevent overfitting
    min_samples_split=5, # Require minimum samples
    random_state=42      # Reproducible
)
rf.fit(X, y)
importances = rf.feature_importances_
```

**Why Random Forest:**
- ✅ Handles non-linear relationships
- ✅ Captures feature interactions
- ✅ Robust to outliers
- ✅ Provides normalized importance scores
- ✅ Industry-standard for feature selection

**Alternative Methods:**
- **Mutual Information**: Measures statistical dependence (non-linear)
- **ANOVA F-statistic**: Univariate test (linear only)

**Output Format:**
```
rank  feature                          importance  cumulative
   1  respiratory.resp_sigh_frequency      0.0876         0.09
   2  respiratory.resp_variability_cv      0.0654         0.15
   3  eda.scr_event_clustering             0.0589         0.21
  ...
```

---

### 2. Statistical Significance Testing

**Method: One-Way ANOVA**

```python
# For each feature, test H0: all emotion means are equal
groups = [data[data['emotion'] == emo][feature] for emo in emotions]
f_stat, p_value = stats.f_oneway(*groups)

# Calculate effect size (eta-squared)
eta_squared = SS_between / SS_total
```

**Interpretation:**
- **p < 0.05**: Feature differs significantly across emotions (reject H0)
- **Effect size (η²)**:
  - < 0.06: Small effect
  - 0.06-0.14: Medium effect
  - \> 0.14: Large effect

**Why ANOVA:**
- ✅ Tests multiple groups simultaneously
- ✅ Provides effect size (practical significance)
- ✅ Well-understood in psychophysiology literature
- ✅ Complements Random Forest (different perspective)

---

### 3. Discriminability Matrix

**Method: Pairwise Centroid Distances**

```python
# 1. Compute emotion centroids (mean feature vector)
centroids = [features[emotion==e].mean() for e in emotions]

# 2. Compute pairwise Euclidean distances
distances = pairwise_distances(centroids, metric='euclidean')
```

**Interpretation:**
- **High distance**: Emotions well-separated in feature space → easy to classify
- **Low distance**: Emotions similar in feature space → may be confused

**Example:**
```
Distance(joy, sad) = 3.45    ← Well separated
Distance(sad, calm) = 1.98   ← Confusable
```

**Use Case:** Identify which emotion pairs need:
- More training data
- Different/additional features
- Potential merging (if too similar)

---

### 4. Pairwise Feature Importance

**Method: Independent t-tests with effect sizes**

For two specific emotions (e.g., joy vs sad):

```python
# For each feature:
t_stat, p_value = stats.ttest_ind(joy_values, sad_values)

# Cohen's d effect size
d = (mean_joy - mean_sad) / pooled_std
```

**Use Case:** When classifier confuses two emotions, identify which features best separate them.

---

## Visualizations

### 1. Feature Distributions (Violin Plots)

**Purpose:** See how feature values distribute across emotions

**Implementation:**
- Violin plot = kernel density + box plot
- Shows: median, quartiles, full distribution
- One subplot per feature (top 15)

**What to look for:**
- **Distinct distributions**: Feature discriminates well
- **Overlapping distributions**: Feature not useful
- **Outliers**: Potential data quality issues

---

### 2. Correlation Matrix (Heatmap)

**Purpose:** Identify redundant features

**Implementation:**
- Pearson correlation coefficient
- Threshold: |r| > 0.8 flagged as "high correlation"

**What to do:**
- If two features correlate > 0.8, keep only the one with higher importance
- Reduces redundancy, speeds up classifier training

---

### 3. PCA Visualization

**Purpose:** Visualize high-dimensional data in 2D/3D

**Implementation:**
```python
# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot with colors by emotion
```

**Interpretation:**
- **Distinct clusters**: Emotions separable
- **Overlapping clouds**: Emotions not well-separated
- **Explained variance**: How much info retained (higher is better)

---

### 4. Feature Importance Bar Chart

**Purpose:** Quick visual ranking of top features

**Implementation:** Horizontal bar chart, sorted by importance

---

## Key Design Decisions

### Decision 1: Random Forest Over Deep Learning

**Choice:** Use Random Forest for feature importance

**Rationale:**
- ✅ Small dataset (76 samples) → RF more appropriate than neural networks
- ✅ Interpretable feature importance scores
- ✅ Fast to compute
- ✅ No need for hyperparameter tuning
- ✅ Proven effective in emotion recognition literature

**Trade-off:** RF may miss complex non-linear interactions that deep learning could capture, but with 76 samples, deep learning would overfit.

---

### Decision 2: Multiple Importance Methods

**Choice:** Provide Random Forest, Mutual Info, and ANOVA

**Rationale:**
- Different methods capture different aspects
- **Random Forest**: Captures interactions
- **Mutual Info**: Non-linear dependencies
- **ANOVA**: Linear differences in means
- Consensus across methods = robust features

---

### Decision 3: Effect Size Over p-values Alone

**Choice:** Report both p-values and effect sizes (η², Cohen's d)

**Rationale:**
- p-values tell you **if** difference exists
- Effect sizes tell you **how large** the difference is
- Small datasets: statistical significance may be hard to achieve
- Large effect sizes with p > 0.05 still informative

**Example:** Feature with p=0.08 (not significant) but η²=0.15 (large effect) → still worth including

---

### Decision 4: Comprehensive Report Generation

**Choice:** Single function generates all analyses + visualizations

**Rationale:**
- ✅ Convenience for users (one command)
- ✅ Ensures consistency across outputs
- ✅ Complete record for documentation
- ✅ Reproducible analysis

**Trade-off:** Takes longer to run, but only run once per dataset

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Time (76 samples, 54 features) |
|-----------|------------|-------------------------------|
| Random Forest | O(n × m × trees) | ~2 seconds |
| ANOVA (all features) | O(n × m × e) | ~0.1 seconds |
| Discriminability Matrix | O(e² × m) | ~0.01 seconds |
| PCA | O(n × m²) | ~0.1 seconds |
| **Total Report** | - | **~15-20 seconds** |

Where:
- n = number of samples (76)
- m = number of features (54)
- e = number of emotions (7)
- trees = 100

**Bottleneck:** Plotting visualizations (~10-15 seconds)

---

## Output Files

### Generated by `generate_report()`

```
reports/feature_analysis/
├── feature_importance.csv          # Ranked features with scores
├── statistical_tests.csv           # ANOVA results per feature
├── discriminability_matrix.csv     # Pairwise emotion distances
├── feature_distributions.png       # Violin plots (top 15)
├── correlation_matrix.png          # Heatmap (top 20)
├── pca_emotions.png                # 2D projection
├── feature_importance_bar.png      # Bar chart (top 20)
└── analysis_summary.txt            # Human-readable summary
```

### File Sizes (Typical)

- CSV files: 10-50 KB each
- PNG files: 100-500 KB each
- Total: ~1-2 MB

---

## Integration Points

### ← From Module 3 (Feature Extractor)

**Input:** `all_features.csv`
- 76 rows (segments)
- 60 columns (6 metadata + 54 features)

**Expected format:**
```csv
segment_id,session_id,emotion,duration,feeling_it,feeling_it_ratio,eda.scl_mean,...
seg_001,session1,joy,34.5,True,0.85,0.523,...
```

---

### → To Module 5 (Classifier)

**Output:** Selected feature list

**Usage in Module 5:**
```python
# Load selected features
importance = pd.read_csv('reports/feature_analysis/feature_importance.csv')
top_features = importance.head(25)['feature'].tolist()

# Use in classifier
classifier.train(features_df, feature_subset=top_features)
```

---

## Validation Strategy

### Unit Tests

```python
# Test basic functionality
def test_feature_importance():
    analyzer = FeatureAnalyzer(test_data)
    importance = analyzer.compute_feature_importance()
    assert len(importance) == 54
    assert importance['importance'].sum() > 0
    assert importance['rank'].iloc[0] == 1

def test_discriminability_matrix():
    analyzer = FeatureAnalyzer(test_data)
    matrix = analyzer.discriminability_matrix()
    assert matrix.shape == (7, 7)  # 7 emotions
    assert np.allclose(matrix.values.T, matrix.values)  # Symmetric
```

### Integration Test

```python
# Test complete workflow
def test_generate_report():
    analyzer = FeatureAnalyzer(test_data)
    analyzer.generate_report('test_output')
    
    # Check all files created
    assert os.path.exists('test_output/feature_importance.csv')
    assert os.path.exists('test_output/pca_emotions.png')
    # ... etc
```

---

## Known Limitations

### 1. Small Sample Size

**Issue:** 76 samples across 7 emotions = ~11 samples/emotion

**Impact:**
- Statistical tests may lack power
- Random Forest may overfit
- PCA may not generalize well

**Mitigation:**
- Use cross-validation in Module 5
- Focus on effect sizes, not just p-values
- Be cautious with interpretations

---

### 2. Imbalanced Emotions

**Issue:** Some emotions have many samples, others few

**Impact:**
- Feature importance biased toward majority emotions
- Discriminability matrix unreliable for rare emotions

**Mitigation:**
- Report emotion counts
- Flag emotions with < 5 samples
- Consider class-balanced Random Forest

---

### 3. No Temporal Information

**Issue:** Features are aggregated over entire segment (5-140s)

**Impact:**
- Misses temporal dynamics of emotions
- Can't detect emotion transitions

**Future Work:** Temporal feature importance analysis

---

## Best Practices

### 1. Always Check Emotion Distribution

```python
print(features['emotion'].value_counts())
# Flag if any emotion < 5 samples
```

### 2. Use Multiple Importance Methods

```python
rf_importance = analyzer.compute_feature_importance('random_forest')
mi_importance = analyzer.compute_feature_importance('mutual_info')

# Features that rank high in both
consensus = set(rf_importance.head(20)['feature']) & \
           set(mi_importance.head(20)['feature'])
```

### 3. Visualize Before Deciding

Always review plots:
- Feature distributions → understand why features work
- PCA → get intuition for emotion separability
- Correlation matrix → avoid redundancy

### 4. Document Decisions

Save selected features:
```python
# Save for Module 5
selected = importance.head(25)['feature'].tolist()
with open('configs/selected_features.txt', 'w') as f:
    for feat in selected:
        f.write(f"{feat}\n")
```

---

## Future Enhancements

### Potential Improvements

1. **Class-Balanced Random Forest** - Handle imbalanced emotions
2. **Temporal Feature Importance** - Analyze features over time
3. **Interactive Visualizations** - Plotly for exploration
4. **Feature Engineering** - Automated feature combinations
5. **Ensemble Feature Selection** - Combine multiple methods robustly

---

## References

### Psychophysiology Literature

- Kreibig (2010): Autonomic nervous system activity in emotion
- Jerritta et al. (2011): Emotion recognition using respiratory features
- Scherer (2009): Emotion theories and concepts

### Machine Learning

- Guyon & Elisseeff (2003): Feature selection
- Breiman (2001): Random Forests
- Cohen (1988): Statistical power and effect sizes

---

## Summary

**Module 4: Feature Analyzer** provides comprehensive validation of features for emotion classification:

✅ **Feature Importance** - Random Forest, Mutual Info, ANOVA  
✅ **Statistical Tests** - ANOVA with effect sizes  
✅ **Discriminability** - Pairwise emotion separability  
✅ **Visualizations** - 6 different plots  
✅ **Complete Report** - Automated generation  

**Output:** Evidence-based feature selection for Module 5 (Classifier)

**Next:** Use top 20-25 features to train emotion classifier
