# Module 4: Feature Analyzer

Validates which physiological features effectively discriminate between emotions.

## Quick Start

```python
# Option 1: Python script
python scripts/analyze_features.py

# Option 2: Python API
from modules.feature_analyzer import FeatureAnalyzer
import pandas as pd

features = pd.read_csv('data/processed/all_features.csv')
analyzer = FeatureAnalyzer(features)
analyzer.generate_report('reports/feature_analysis')
```

## What It Does

1. **Ranks features by importance** - Identifies top discriminative features
2. **Tests statistical significance** - ANOVA across emotions
3. **Measures emotion separability** - Which emotion pairs are confusable?
4. **Generates visualizations** - Distribution plots, PCA, correlations
5. **Creates comprehensive report** - Complete analysis in one command

## Output Files

```
reports/feature_analysis/
├── feature_importance.csv          # Top features ranked
├── statistical_tests.csv           # Significance tests
├── discriminability_matrix.csv     # Emotion pair separability
├── feature_distributions.png       # Violin plots
├── correlation_matrix.png          # Feature correlations
├── pca_emotions.png                # 2D visualization
├── feature_importance_bar.png      # Importance chart
└── analysis_summary.txt            # Human-readable summary
```

## Key Insights

After running, you'll know:
- ✅ **Which 20-25 features to use** for classifier
- ✅ **Which emotions are well-separated** (easy to classify)
- ✅ **Which emotions are confusable** (need more data/features)
- ✅ **Which features are redundant** (high correlation)

## Example Results

```
TOP 10 FEATURES:
  1. respiratory.resp_sigh_frequency       0.0876
  2. respiratory.resp_variability_cv       0.0654
  3. eda.scr_event_clustering              0.0589
  4. cardiac.hrv_rmssd                     0.0512
  5. multimodal.arousal_index              0.0487
  ...

Top 22 features explain 80% of discriminative power

MOST SEPARABLE EMOTIONS:
  joy <-> sad:   3.45  (easy to classify)
  fear <-> calm: 3.54  (easy to classify)

LEAST SEPARABLE EMOTIONS:
  tired <-> relaxed: 1.98  (may be confused)
```

## Dependencies

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn

## Documentation

- **Usage Guide:** `docs/guides/FeatureAnalyzer_Usage_Guide.md`
- **Implementation:** `docs/implementation/Module4_Implementation_Summary.md`
- **API Reference:** See docstrings in `modules/feature_analyzer.py`

## Next Steps

After feature analysis:
1. Review `feature_importance.csv` → select top 20-25 features
2. Check `discriminability_matrix.csv` → identify confusable emotions
3. Proceed to **Module 5: Classifier** with selected features

## Status

✅ Complete and tested  
✅ Validated on 76-segment dataset  
✅ Generates all visualizations  
✅ Comprehensive documentation
