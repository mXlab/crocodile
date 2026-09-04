# Module 4: Feature Analyzer - Installation Checklist

## Files to Copy

Copy these files to your `biodata_pipeline` directory:

### 1. Module File
```bash
cp modules/feature_analyzer.py biodata_pipeline/modules/
```

### 2. Script File
```bash
cp scripts/analyze_features.py biodata_pipeline/scripts/
```

### 3. Documentation Files
```bash
cp docs/guides/FeatureAnalyzer_Usage_Guide.md biodata_pipeline/docs/guides/
cp docs/implementation/Module4_Implementation_Summary.md biodata_pipeline/docs/implementation/
cp docs/README_Module4.md biodata_pipeline/docs/
```

### 4. Example File (Optional)
```bash
mkdir -p biodata_pipeline/examples
cp examples/feature_analysis_example.py biodata_pipeline/examples/
```

---

## Quick Setup Commands

```bash
# Navigate to your biodata_pipeline directory
cd /path/to/crocodile/biodata_pipeline

# Create directories if needed
mkdir -p modules scripts docs/guides docs/implementation examples

# Copy module
cp /path/to/downloads/modules/feature_analyzer.py modules/

# Copy script
cp /path/to/downloads/scripts/analyze_features.py scripts/

# Copy docs
cp /path/to/downloads/docs/guides/FeatureAnalyzer_Usage_Guide.md docs/guides/
cp /path/to/downloads/docs/implementation/Module4_Implementation_Summary.md docs/implementation/
cp /path/to/downloads/docs/README_Module4.md docs/

# Copy example
cp /path/to/downloads/examples/feature_analysis_example.py examples/
```

---

## Verify Installation

```bash
# Check files are in place
ls modules/feature_analyzer.py
ls scripts/analyze_features.py

# Test import
python -c "from modules.feature_analyzer import FeatureAnalyzer; print('✓ Module imported successfully')"
```

---

## Run Feature Analysis

```bash
# Make sure you're in biodata_pipeline directory
cd biodata_pipeline

# Activate virtual environment
source venv/bin/activate

# Run analysis
python scripts/analyze_features.py
```

**Expected output:**
```
================================================================================
FEATURE ANALYSIS
================================================================================

Loading features from: data/processed/all_features.csv
✓ Loaded 76 samples
...

[Generates complete analysis report]

✓ ANALYSIS COMPLETE
Results saved to: reports/feature_analysis/
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'modules'"

**Solution:**
```bash
# Make sure you're running from biodata_pipeline directory
cd biodata_pipeline
python scripts/analyze_features.py
```

### Error: "FileNotFoundError: all_features.csv"

**Solution:**
```bash
# Run feature extraction first
python scripts/process_all_sessions.py
```

### Error: Import errors (seaborn, sklearn, etc.)

**Solution:**
```bash
# Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

---

## File Structure After Installation

```
biodata_pipeline/
├── modules/
│   ├── __init__.py
│   ├── data_slicer.py          ✅
│   ├── feature_extractor.py    ✅
│   └── feature_analyzer.py     ✅ NEW
│
├── scripts/
│   ├── process_all_sessions.py ✅
│   └── analyze_features.py     ✅ NEW
│
├── docs/
│   ├── guides/
│   │   ├── DataSlicer_Usage_Guide.md        ✅
│   │   └── FeatureAnalyzer_Usage_Guide.md   ✅ NEW
│   │
│   ├── implementation/
│   │   ├── Feature_Extraction_Validation_Report.md  ✅
│   │   ├── Module2_Implementation_Summary.md        ✅
│   │   └── Module4_Implementation_Summary.md        ✅ NEW
│   │
│   └── README_Module4.md        ✅ NEW
│
└── examples/
    └── feature_analysis_example.py  ✅ NEW
```

---

## Next Steps

1. ✅ Install Module 4 files
2. ✅ Run feature analysis: `python scripts/analyze_features.py`
3. ✅ Review results in `reports/feature_analysis/`
4. 📋 Select top features for classifier
5. 📋 Proceed to Module 5: Classifier training

---

## Quick Test

```bash
# After installation, run this to verify everything works:

cd biodata_pipeline
source venv/bin/activate

# Test 1: Import module
python << 'EOF'
from modules.feature_analyzer import FeatureAnalyzer
print("✓ Module imports successfully")
EOF

# Test 2: Run analysis (if you have data)
python scripts/analyze_features.py

# Test 3: Check outputs
ls reports/feature_analysis/
```

**Expected files in `reports/feature_analysis/`:**
- feature_importance.csv
- statistical_tests.csv
- discriminability_matrix.csv
- feature_distributions.png
- correlation_matrix.png
- pca_emotions.png
- feature_importance_bar.png
- analysis_summary.txt

---

## Ready to Use!

Your Feature Analyzer is now installed and ready to validate your features! 🚀
