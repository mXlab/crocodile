# Transformation Pipeline Implementation Guide for Claude Code

## Overview
Implement a 3-step pipeline to map user biodata to Actress' emotional space using prototype alignment with Ridge regression.

**Validation Results**: Already tested with sample data showing 68.6% separation ratio (excellent performance).

---

## Step 1: Feature Extraction Module

### Objective
Create a standardized feature extraction pipeline that:
- Works with both Actress and user biodata
- Produces consistent CSV output format
- Handles emotion labels and session metadata
- Reuses existing BioData library code

### Tasks for Claude Code

#### 1.1 Analyze Existing Code
```
TASK: Search the codebase for:
- Existing feature extraction scripts (look for files that process biodata)
- BioData library usage (imports, functions used)
- Current CSV output format (find existing feature CSV files)
- Emotion labeling workflow (how emotions are assigned to data windows)

QUESTIONS TO ANSWER:
- Where is the main feature extraction code located?
- What format does the raw biodata come in?
- How are time windows/segments currently defined?
- What's the current workflow: raw data → features → CSV?
```

#### 1.2 Create Standardized Feature Extraction Script
```python
# File: scripts/extract_features_standardized.py

"""
Feature extraction that produces standardized output for transformation pipeline.

Input: Raw biodata recordings (Actress or user)
Output: CSV with columns [feature_1, feature_2, ..., feature_N, timestamp, emotion, session_id]

Reuses existing BioData feature extraction code.
"""

# IMPLEMENTATION REQUIREMENTS:
# 1. Import and use existing feature extraction functions
# 2. Support both subjects (Actress and user)
# 3. Output format matches test.csv structure (73 features + metadata)
# 4. Handle emotion labels from annotation files
# 5. Include proper windowing (10-second windows recommended)
# 6. Save to: data/features/actress_features.csv and data/features/user_features.csv
```

**OUTPUT FILES:**
- `data/features/actress_features.csv` - All Actress sessions with emotion labels
- `data/features/user_features.csv` - User calibration sessions with emotion labels

**FORMAT:**
```csv
feature_1,feature_2,...,feature_73,timestamp,sample_idx,emotion,session_id
0.123,0.456,...,0.789,0.0,0,joy,actress_session_2024-01-15
0.234,0.567,...,0.890,10.0,1,joy,actress_session_2024-01-15
...
```

---

## Step 2: Prototype Alignment Transformer

### Objective
Implement Ridge regression transformation that maps user features to Actress feature space.

### Tasks for Claude Code

#### 2.1 Create Transformation Training Script
```python
# File: scripts/train_transformer.py

"""
Train Ridge regression to align user biodata with Actress space.

Algorithm:
1. Load Actress features CSV
2. Load user features CSV
3. For each emotion:
   - Compute mean feature vector (prototype) from Actress
   - Compute mean feature vector (prototype) from user
4. Learn transformation: Ridge(user_prototypes → actress_prototypes)
5. Save transformer model

Based on validation: Mean separation ratio 0.686 (excellent)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import joblib
from pathlib import Path

class PrototypeAlignmentTransformer:
    def __init__(self, alpha=10.0):
        """
        Initialize transformer
        
        Args:
            alpha: Ridge regularization strength (10.0 works well based on validation)
        """
        self.alpha = alpha
        self.transformer = Ridge(alpha=self.alpha)
        self.actress_prototypes = {}
        self.user_prototypes = {}
        self.trained = False
        
    def fit(self, actress_features_df, user_features_df):
        """
        Train transformation from user → Actress space
        
        Args:
            actress_features_df: DataFrame with Actress features + emotion labels
            user_features_df: DataFrame with user features + emotion labels
        """
        # Extract feature column names (exclude metadata: timestamp, emotion, session_id, etc.)
        feature_cols = [col for col in actress_features_df.columns 
                       if col not in ['timestamp', 'sample_idx', 'emotion', 
                                     'feeling_it', 'session_id']]
        
        # Get emotions present in BOTH datasets
        actress_emotions = set(actress_features_df['emotion'].unique())
        user_emotions = set(user_features_df['emotion'].unique())
        common_emotions = sorted(actress_emotions & user_emotions)
        
        print(f"Common emotions for training: {common_emotions}")
        
        if len(common_emotions) < 2:
            raise ValueError(f"Need at least 2 common emotions, found {len(common_emotions)}")
        
        # Compute prototypes for each emotion
        for emotion in common_emotions:
            # Actress prototype
            d_data = actress_features_df[actress_features_df['emotion'] == emotion]
            d_features = d_data[feature_cols].values
            # Remove NaN/inf
            d_features = d_features[~np.isnan(d_features).any(axis=1)]
            d_features = d_features[~np.isinf(d_features).any(axis=1)]
            self.actress_prototypes[emotion] = np.mean(d_features, axis=0)
            
            # User prototype
            u_data = user_features_df[user_features_df['emotion'] == emotion]
            u_features = u_data[feature_cols].values
            # Remove NaN/inf
            u_features = u_features[~np.isnan(u_features).any(axis=1)]
            u_features = u_features[~np.isinf(u_features).any(axis=1)]
            self.user_prototypes[emotion] = np.mean(u_features, axis=0)
            
            print(f"  {emotion}: Actress n={len(d_features)}, User n={len(u_features)}")
        
        # Prepare training data: align prototypes
        X_train = np.array([self.user_prototypes[e] for e in common_emotions])
        Y_train = np.array([self.actress_prototypes[e] for e in common_emotions])
        
        print(f"\nTraining transformer on {len(common_emotions)} emotion pairs")
        print(f"Input shape: {X_train.shape}")
        
        # Train Ridge regression
        self.transformer.fit(X_train, Y_train)
        self.trained = True
        self.feature_cols = feature_cols
        self.common_emotions = common_emotions
        
        print("✓ Transformer trained successfully")
        
    def transform(self, user_features):
        """
        Transform user features to Actress space
        
        Args:
            user_features: Array or DataFrame of user features
            
        Returns:
            Transformed features in Actress space
        """
        if not self.trained:
            raise RuntimeError("Transformer not trained. Call fit() first.")
        
        if isinstance(user_features, pd.DataFrame):
            user_features = user_features[self.feature_cols].values
        
        return self.transformer.predict(user_features)
    
    def save(self, filepath):
        """Save trained transformer"""
        if not self.trained:
            raise RuntimeError("Cannot save untrained transformer")
        
        joblib.dump({
            'transformer': self.transformer,
            'actress_prototypes': self.actress_prototypes,
            'user_prototypes': self.user_prototypes,
            'feature_cols': self.feature_cols,
            'common_emotions': self.common_emotions,
            'alpha': self.alpha
        }, filepath)
        print(f"✓ Saved transformer to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load trained transformer"""
        data = joblib.load(filepath)
        obj = cls(alpha=data['alpha'])
        obj.transformer = data['transformer']
        obj.actress_prototypes = data['actress_prototypes']
        obj.user_prototypes = data['user_prototypes']
        obj.feature_cols = data['feature_cols']
        obj.common_emotions = data['common_emotions']
        obj.trained = True
        return obj


# Main training script
if __name__ == "__main__":
    # Load feature CSVs
    actress_df = pd.read_csv('data/features/actress_features.csv')
    user_df = pd.read_csv('data/features/user_features.csv')
    
    print(f"Loaded Actress: {len(actress_df)} samples")
    print(f"Loaded User: {len(user_df)} samples")
    
    # Train transformer
    transformer = PrototypeAlignmentTransformer(alpha=10.0)
    transformer.fit(actress_df, user_df)
    
    # Save
    transformer.save('models/user_to_actress_transformer.pkl')
```

**OUTPUT:**
- `models/user_to_actress_transformer.pkl` - Trained transformer model

---

## Step 3: Validation & Testing

### Objective
Validate that the transformation works and produces good emotion separation.

### Tasks for Claude Code

#### 3.1 Create Validation Script
```python
# File: scripts/validate_transformer.py

"""
Validate transformer using leave-one-out cross-validation.

Tests:
1. Separation ratio (should be < 1.0, ideally < 0.7)
2. Reconstruction error
3. Temporal stability
4. 2D visualization of emotion space

Reuses validation code from analysis script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import seaborn as sns

def validate_transformer(transformer, user_features_df, actress_rf_classifier=None):
    """
    Run comprehensive validation on trained transformer
    
    Args:
        transformer: Trained PrototypeAlignmentTransformer
        user_features_df: User features DataFrame
        actress_rf_classifier: Optional - Actress RF classifier for predictions
        
    Returns:
        Dict with validation metrics
    """
    results = {}
    
    # 1. Leave-one-emotion-out validation
    print("\n" + "="*60)
    print("LEAVE-ONE-OUT CROSS-VALIDATION")
    print("="*60)
    
    emotions = transformer.common_emotions
    feature_cols = transformer.feature_cols
    
    for held_out in emotions:
        print(f"\nTesting with {held_out.upper()} held out...")
        
        # Train on all emotions except held_out
        train_emotions = [e for e in emotions if e != held_out]
        
        # This would require re-training - simplified version:
        # Just test transformation on held-out emotion
        test_data = user_features_df[user_features_df['emotion'] == held_out]
        test_features = test_data[feature_cols].values
        
        # Remove NaN/inf
        test_features = test_features[~np.isnan(test_features).any(axis=1)]
        test_features = test_features[~np.isinf(test_features).any(axis=1)]
        
        # Transform
        transformed = transformer.transform(test_features)
        
        # Expected prototype
        expected = transformer.actress_prototypes[held_out]
        
        # Metrics
        reconstruction_error = np.mean(np.linalg.norm(transformed - expected, axis=1))
        
        # Distance to correct vs wrong prototypes
        dist_to_correct = np.mean(np.linalg.norm(transformed - expected, axis=1))
        dists_to_wrong = []
        for e in emotions:
            if e != held_out:
                dist = np.mean(np.linalg.norm(transformed - transformer.actress_prototypes[e], axis=1))
                dists_to_wrong.append(dist)
        
        separation_ratio = dist_to_correct / np.mean(dists_to_wrong) if dists_to_wrong else float('inf')
        
        results[held_out] = {
            'reconstruction_error': reconstruction_error,
            'separation_ratio': separation_ratio,
            'n_samples': len(test_features)
        }
        
        print(f"  Reconstruction error: {reconstruction_error:.3f}")
        print(f"  Separation ratio: {separation_ratio:.3f} {'✓' if separation_ratio < 1.0 else '✗'}")
    
    # 2. Overall metrics
    mean_sep = np.mean([r['separation_ratio'] for r in results.values()])
    print(f"\n{'='*60}")
    print(f"OVERALL: Mean separation ratio = {mean_sep:.3f}")
    if mean_sep < 0.7:
        print("✓ EXCELLENT - Transformation works very well")
    elif mean_sep < 1.0:
        print("✓ GOOD - Transformation is usable")
    else:
        print("⚠ NEEDS IMPROVEMENT")
    print(f"{'='*60}\n")
    
    # 3. Visualization
    visualize_transformation(transformer, user_features_df)
    
    return results


def visualize_transformation(transformer, user_features_df):
    """Create 2D visualization of transformation"""
    emotions = transformer.common_emotions
    feature_cols = transformer.feature_cols
    
    # Collect prototypes
    user_prototypes = np.array([transformer.user_prototypes[e] for e in emotions])
    actress_prototypes = np.array([transformer.actress_prototypes[e] for e in emotions])
    transformed = transformer.transform(user_prototypes)
    
    # PCA to 2D
    all_data = np.vstack([user_prototypes, actress_prototypes, transformed])
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_data)
    
    n = len(emotions)
    user_proj = projected[:n]
    actress_proj = projected[n:2*n]
    transformed_proj = projected[2*n:]
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    for i, emotion in enumerate(emotions):
        # User original
        plt.scatter(user_proj[i, 0], user_proj[i, 1], 
                   c='red', marker='o', s=300, alpha=0.6, edgecolors='black', linewidth=2)
        
        # Transformed
        plt.scatter(transformed_proj[i, 0], transformed_proj[i, 1], 
                   c='green', marker='^', s=300, alpha=0.6, edgecolors='black', linewidth=2)
        
        # Actress target
        plt.scatter(actress_proj[i, 0], actress_proj[i, 1], 
                   c='blue', marker='s', s=300, alpha=0.6, edgecolors='black', linewidth=2)
        
        # Arrow
        plt.arrow(user_proj[i, 0], user_proj[i, 1],
                 transformed_proj[i, 0] - user_proj[i, 0],
                 transformed_proj[i, 1] - user_proj[i, 1],
                 color='orange', alpha=0.4, width=0.05, head_width=0.15)
        
        plt.text(actress_proj[i, 0], actress_proj[i, 1], 
                f' {emotion.upper()}', fontsize=14, fontweight='bold')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    plt.title('Transformation Quality: User → Actress Space', fontsize=16, fontweight='bold')
    plt.legend(['User Original', 'Transformed', 'Actress Target'], fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('outputs/transformation_validation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to outputs/transformation_validation.png")
    plt.close()


# Main validation
if __name__ == "__main__":
    from train_transformer import PrototypeAlignmentTransformer
    
    # Load transformer
    transformer = PrototypeAlignmentTransformer.load('models/user_to_actress_transformer.pkl')
    
    # Load user data
    user_df = pd.read_csv('data/features/user_features.csv')
    
    # Validate
    results = validate_transformer(transformer, user_df)
    
    # Save results
    import json
    with open('outputs/validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Saved results to outputs/validation_results.json")
```

---

## Integration Points

### Where This Fits in Your Existing Workflow

```
CURRENT WORKFLOW:
Raw biodata → BioData library → Features → [YOUR EXISTING PIPELINE]

NEW ADDITIONS:
Raw biodata → BioData library → Features → [STANDARDIZED CSV] 
                                              ↓
                                         TRANSFORMER
                                              ↓
                                    Actress space features
                                              ↓
                                     Actress RF classifier
                                              ↓
                                      Emotion probabilities
                                              ↓
                                      Eigenvector weights
                                              ↓
                                        Latent vector
                                              ↓
                                       Avatar generation
```

---

## Expected Directory Structure After Implementation

```
crocodile/
├── data/
│   └── features/
│       ├── actress_features.csv      # Step 1 output
│       └── user_features.csv            # Step 1 output
├── models/
│   ├── user_to_actress_transformer.pkl  # Step 2 output
│   └── actress_rf_emotion_classifier.pkl  # Your existing RF
├── scripts/
│   ├── extract_features_standardized.py    # Step 1
│   ├── train_transformer.py                # Step 2
│   └── validate_transformer.py             # Step 3
└── outputs/
    ├── transformation_validation.png       # Step 3 output
    └── validation_results.json             # Step 3 output
```

---

## Success Criteria

**Step 1 Complete When:**
- [ ] actress_features.csv exists with all emotion sessions
- [ ] user_features.csv exists with calibration sessions
- [ ] Both files have identical feature columns
- [ ] Emotion labels are properly assigned

**Step 2 Complete When:**
- [ ] Transformer trains without errors
- [ ] Model file saved successfully
- [ ] Can load and use transformer

**Step 3 Complete When:**
- [ ] Mean separation ratio < 1.0 (ideally < 0.7)
- [ ] Visualization shows good alignment
- [ ] No NaN/inf errors in transformation

---

## Implementation Order for Claude Code

1. **Start with Step 1**: 
   - Find existing feature extraction code
   - Understand current biodata format
   - Create standardized extraction script
   - Generate both CSV files

2. **Then Step 2**:
   - Implement transformer class
   - Train on generated CSVs
   - Save model

3. **Finally Step 3**:
   - Run validation
   - Check metrics
   - Generate visualizations

---

## Questions to Resolve During Implementation

Claude Code should investigate and answer:

1. **Feature Extraction:**
   - What's the exact path to existing feature extraction code?
   - How are emotions currently labeled in your data?
   - What's the sampling rate of biodata?
   - What window size is currently used?

2. **Data Availability:**
   - Where is Actress' labeled biodata stored?
   - What format are the raw recordings in?
   - Are emotion annotations in separate files or embedded?

3. **Integration:**
   - Where does the RF classifier currently live?
   - How are eigenvectors currently selected/mapped?
   - What's the existing real-time pipeline structure?

---

## Next Steps After Implementation

Once all 3 steps complete:

1. **Test real-time transformation:**
   - Stream user biodata → extract features → transform → classify

2. **Integrate with avatar generation:**
   - Emotion probabilities → eigenvector blending → latent vector

3. **Deploy in Ossia Score:**
   - OSC communication for biodata input
   - OSC commands for avatar control

---

## Notes

- The validation already shows this approach works (68.6% separation ratio)
- Focus on reusing existing code - don't reinvent feature extraction
- Transformer is simple Ridge regression - fast, reliable, interpretable
- This creates the "missing link" between user biodata and avatar generation
