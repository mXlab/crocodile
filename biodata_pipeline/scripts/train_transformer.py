"""
Train Ridge regression to align a new subject's biodata features
to the reference subject's emotional feature space.

Algorithm:
1. Load reference and subject feature CSVs
2. Find emotions present in both datasets
3. Compute per-emotion mean feature vectors (prototypes)
4. Train Ridge regression: subject prototypes → reference prototypes
5. Save transformer model

Usage (from biodata_pipeline/):
    python scripts/train_transformer.py --reference data/processed/continuous_features.csv --subject data/processed/erin_features.csv
    python scripts/train_transformer.py --reference data/processed/continuous_features.csv --subject data/processed/new_subject.csv --alpha 5.0
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
import joblib
import argparse

METADATA_COLS = ['timestamp', 'sample_idx', 'emotion', 'feeling_it', 'session_id']


class PrototypeAlignmentTransformer:
    def __init__(self, alpha=10.0, n_features=None):
        self.alpha = alpha
        self.n_features = n_features
        self.transformer = Ridge(alpha=self.alpha)
        self.reference_prototypes = {}
        self.subject_prototypes = {}
        self.feature_cols = []
        self.common_emotions = []
        self.ref_scaler = StandardScaler()
        self.sub_scaler = StandardScaler()
        self.trained = False

    def fit(self, reference_df, subject_df):
        """Train transformation from subject → reference feature space."""
        ref_features = set(c for c in reference_df.columns if c not in METADATA_COLS)
        sub_features = set(c for c in subject_df.columns if c not in METADATA_COLS)
        self.feature_cols = sorted(ref_features & sub_features)

        if ref_features != sub_features:
            print(f"  Note: using {len(self.feature_cols)} common features "
                  f"(reference has {len(ref_features)}, subject has {len(sub_features)})")

        ref_emotions = set(reference_df['emotion'].dropna().unique())
        sub_emotions = set(subject_df['emotion'].dropna().unique())
        self.common_emotions = sorted(ref_emotions & sub_emotions)

        print(f"Common emotions for training: {self.common_emotions}")

        if len(self.common_emotions) < 2:
            raise ValueError(
                f"Need at least 2 common emotions, found {len(self.common_emotions)}: "
                f"reference has {sorted(ref_emotions)}, subject has {sorted(sub_emotions)}"
            )

        def clean(arr):
            mask = ~(np.isnan(arr).any(axis=1) | np.isinf(arr).any(axis=1))
            return arr[mask]

        # Feature selection: rank by ANOVA F-test on reference data
        if self.n_features and self.n_features < len(self.feature_cols):
            ref_sel = reference_df.loc[
                reference_df['emotion'].isin(self.common_emotions)
            ].copy()
            X_sel = ref_sel[self.feature_cols].values
            y_sel = ref_sel['emotion'].values
            valid = ~(np.isnan(X_sel).any(axis=1) | np.isinf(X_sel).any(axis=1))
            X_sel, y_sel = X_sel[valid], y_sel[valid]

            f_scores, _ = f_classif(X_sel, y_sel)
            top_idx = np.argsort(f_scores)[::-1][:self.n_features]
            selected = sorted([self.feature_cols[i] for i in top_idx])

            print(f"  Feature selection: {len(self.feature_cols)} -> {len(selected)} "
                  f"(top {self.n_features} by ANOVA F-test)")
            self.feature_cols = selected

        # Collect all valid rows to fit scalers
        ref_all = reference_df.loc[
            reference_df['emotion'].isin(self.common_emotions), self.feature_cols
        ].values
        sub_all = subject_df.loc[
            subject_df['emotion'].isin(self.common_emotions), self.feature_cols
        ].values

        ref_all = clean(ref_all)
        sub_all = clean(sub_all)

        self.ref_scaler.fit(ref_all)
        self.sub_scaler.fit(sub_all)

        for emotion in self.common_emotions:
            r_feat = reference_df.loc[reference_df['emotion'] == emotion, self.feature_cols].values
            r_feat = clean(r_feat)
            r_scaled = self.ref_scaler.transform(r_feat)
            self.reference_prototypes[emotion] = np.mean(r_scaled, axis=0)

            s_feat = subject_df.loc[subject_df['emotion'] == emotion, self.feature_cols].values
            s_feat = clean(s_feat)
            s_scaled = self.sub_scaler.transform(s_feat)
            self.subject_prototypes[emotion] = np.mean(s_scaled, axis=0)

            print(f"  {emotion}: reference n={len(r_feat)}, subject n={len(s_feat)}")

        X_train = np.array([self.subject_prototypes[e] for e in self.common_emotions])
        Y_train = np.array([self.reference_prototypes[e] for e in self.common_emotions])

        print(f"\nTraining Ridge(alpha={self.alpha}) on {len(self.common_emotions)} prototype pairs")
        print(f"  Feature dimensionality: {X_train.shape[1]}")

        self.transformer.fit(X_train, Y_train)
        self.trained = True
        print("Transformer trained successfully")

    def transform(self, features):
        """Transform subject features into reference feature space."""
        if not self.trained:
            raise RuntimeError("Transformer not trained. Call fit() first.")
        if isinstance(features, pd.DataFrame):
            features = features[self.feature_cols].values
        scaled = self.sub_scaler.transform(features)
        transformed_scaled = self.transformer.predict(scaled)
        return self.ref_scaler.inverse_transform(transformed_scaled)

    def save(self, filepath):
        if not self.trained:
            raise RuntimeError("Cannot save untrained transformer")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'transformer': self.transformer,
            'reference_prototypes': self.reference_prototypes,
            'subject_prototypes': self.subject_prototypes,
            'feature_cols': self.feature_cols,
            'common_emotions': self.common_emotions,
            'alpha': self.alpha,
            'n_features': self.n_features,
            'ref_scaler': self.ref_scaler,
            'sub_scaler': self.sub_scaler,
        }, filepath)
        print(f"Saved transformer to {filepath}")

    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        obj = cls(alpha=data['alpha'], n_features=data.get('n_features'))
        obj.transformer = data['transformer']
        obj.reference_prototypes = data['reference_prototypes']
        obj.subject_prototypes = data['subject_prototypes']
        obj.feature_cols = data['feature_cols']
        obj.common_emotions = data['common_emotions']
        obj.ref_scaler = data['ref_scaler']
        obj.sub_scaler = data['sub_scaler']
        obj.trained = True
        return obj


def main():
    parser = argparse.ArgumentParser(
        description="Train subject-to-reference prototype alignment transformer"
    )
    parser.add_argument(
        '--reference', required=True,
        help='Reference subject features CSV'
    )
    parser.add_argument(
        '--subject', required=True,
        help='New subject features CSV'
    )
    parser.add_argument(
        '--alpha', type=float, default=10.0,
        help='Ridge regularization strength (default: 10.0)'
    )
    parser.add_argument(
        '--n-features', type=int, default=None,
        help='Keep only top N features by ANOVA F-test (default: use all)'
    )
    parser.add_argument(
        '--output',
        default='models/subject_alignment_transformer.pkl',
        help='Output path for trained transformer'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Subject-to-Reference Prototype Alignment")
    print("=" * 60)

    ref_df = pd.read_csv(args.reference)
    sub_df = pd.read_csv(args.subject)
    print(f"Reference: {len(ref_df)} samples from {args.reference}")
    print(f"Subject:   {len(sub_df)} samples from {args.subject}")

    transformer = PrototypeAlignmentTransformer(alpha=args.alpha, n_features=args.n_features)
    transformer.fit(ref_df, sub_df)
    transformer.save(args.output)

    print("\nDone.")


if __name__ == '__main__':
    main()
