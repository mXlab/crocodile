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

    def transform_prototypes(self):
        """Return transformed subject prototypes in raw reference space."""
        sub_protos_scaled = np.array([self.subject_prototypes[e] for e in self.common_emotions])
        trans_scaled = self.transformer.predict(sub_protos_scaled)
        return self.ref_scaler.inverse_transform(trans_scaled)

    def save(self, filepath):
        if not self.trained:
            raise RuntimeError("Cannot save untrained transformer")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'class': 'Ridge',
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
    def load(cls, data):
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


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def _clean(arr):
    mask = ~(np.isnan(arr).any(axis=1) | np.isinf(arr).any(axis=1))
    return arr[mask]


def _common_setup(reference_df, subject_df):
    """Return feature_cols, common_emotions shared by both dataframes."""
    ref_features = set(c for c in reference_df.columns if c not in METADATA_COLS)
    sub_features = set(c for c in subject_df.columns if c not in METADATA_COLS)
    feature_cols = sorted(ref_features & sub_features)

    if ref_features != sub_features:
        print(f"  Note: using {len(feature_cols)} common features "
              f"(reference has {len(ref_features)}, subject has {len(sub_features)})")

    ref_emotions = set(reference_df['emotion'].dropna().unique())
    sub_emotions = set(subject_df['emotion'].dropna().unique())
    common_emotions = sorted(ref_emotions & sub_emotions)

    print(f"Common emotions: {common_emotions}")

    if len(common_emotions) < 2:
        raise ValueError(
            f"Need at least 2 common emotions, found {len(common_emotions)}: "
            f"reference has {sorted(ref_emotions)}, subject has {sorted(sub_emotions)}"
        )

    return feature_cols, common_emotions


def _sym_sqrt(M):
    """Symmetric PSD matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _sym_inv_sqrt(M):
    """Symmetric PSD matrix inverse square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


# ─────────────────────────────────────────────────────────────────────────────
# Linear OT transformer (global Gaussian Monge map)
# ─────────────────────────────────────────────────────────────────────────────

class LinearOTTransformer:
    """
    Global linear optimal transport (Gaussian Monge map).

    Fits a single linear map from the subject distribution to the reference
    distribution using all samples from common emotions.  Unlike Ridge, this
    uses the full covariance structure of both distributions, not just means.
    """

    def __init__(self, reg=1e-5):
        self.reg = reg
        self.ot = None
        self.sub_scaler = StandardScaler()
        self.ref_scaler = StandardScaler()
        self.feature_cols = []
        self.common_emotions = []
        self.subject_prototypes = {}
        self.reference_prototypes = {}
        self.trained = False

    def fit(self, reference_df, subject_df):
        import ot as pot

        self.feature_cols, self.common_emotions = _common_setup(reference_df, subject_df)

        ref_all = _clean(reference_df.loc[
            reference_df['emotion'].isin(self.common_emotions), self.feature_cols
        ].values)
        sub_all = _clean(subject_df.loc[
            subject_df['emotion'].isin(self.common_emotions), self.feature_cols
        ].values)

        self.ref_scaler.fit(ref_all)
        self.sub_scaler.fit(sub_all)

        for emotion in self.common_emotions:
            r = _clean(reference_df.loc[reference_df['emotion'] == emotion, self.feature_cols].values)
            s = _clean(subject_df.loc[subject_df['emotion'] == emotion, self.feature_cols].values)
            self.reference_prototypes[emotion] = np.mean(self.ref_scaler.transform(r), axis=0)
            self.subject_prototypes[emotion] = np.mean(self.sub_scaler.transform(s), axis=0)
            print(f"  {emotion}: reference n={len(r)}, subject n={len(s)}")

        X_ref_sc = self.ref_scaler.transform(ref_all)
        X_sub_sc = self.sub_scaler.transform(sub_all)

        print(f"\nFitting Linear OT on {len(X_sub_sc)} subject → {len(X_ref_sc)} reference samples")
        self.ot = pot.da.LinearTransport(reg=self.reg)
        self.ot.fit(Xs=X_sub_sc, Xt=X_ref_sc)
        self.trained = True
        print("Linear OT transformer trained successfully")

    def transform(self, features):
        if not self.trained:
            raise RuntimeError("Transformer not trained. Call fit() first.")
        if isinstance(features, pd.DataFrame):
            features = features[self.feature_cols].values
        scaled = self.sub_scaler.transform(features)
        transformed_scaled = self.ot.transform(Xs=scaled)
        return self.ref_scaler.inverse_transform(transformed_scaled)

    def transform_prototypes(self):
        """Return transformed subject prototypes in raw reference space."""
        sub_protos_scaled = np.array([self.subject_prototypes[e] for e in self.common_emotions])
        trans_scaled = self.ot.transform(Xs=sub_protos_scaled)
        return self.ref_scaler.inverse_transform(trans_scaled)

    def save(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'class': 'LinearOT',
            'reg': self.reg,
            'ot': self.ot,
            'sub_scaler': self.sub_scaler,
            'ref_scaler': self.ref_scaler,
            'feature_cols': self.feature_cols,
            'common_emotions': self.common_emotions,
            'subject_prototypes': self.subject_prototypes,
            'reference_prototypes': self.reference_prototypes,
        }, filepath)
        print(f"Saved LinearOT transformer to {filepath}")

    @classmethod
    def load(cls, data):
        obj = cls(reg=data['reg'])
        obj.ot = data['ot']
        obj.sub_scaler = data['sub_scaler']
        obj.ref_scaler = data['ref_scaler']
        obj.feature_cols = data['feature_cols']
        obj.common_emotions = data['common_emotions']
        obj.subject_prototypes = data['subject_prototypes']
        obj.reference_prototypes = data['reference_prototypes']
        obj.trained = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Class-conditional OT transformer (per-emotion Gaussian Monge maps)
# ─────────────────────────────────────────────────────────────────────────────

class ClassConditionalOTTransformer:
    """
    Per-emotion linear optimal transport (Gaussian Monge map).

    Fits a separate OT map for each emotion class, aligning
    subject_features|emotion → reference_features|emotion.

    At inference (unknown emotion), assigns each sample to its nearest
    subject prototype and applies the corresponding emotion map.
    """

    def __init__(self, reg=1e-5):
        self.reg = reg
        self.ot_maps = {}
        self.sub_scaler = StandardScaler()
        self.ref_scaler = StandardScaler()
        self.feature_cols = []
        self.common_emotions = []
        self.subject_prototypes = {}
        self.reference_prototypes = {}
        self.trained = False

    def fit(self, reference_df, subject_df):
        import ot as pot

        self.feature_cols, self.common_emotions = _common_setup(reference_df, subject_df)

        ref_all = _clean(reference_df.loc[
            reference_df['emotion'].isin(self.common_emotions), self.feature_cols
        ].values)
        sub_all = _clean(subject_df.loc[
            subject_df['emotion'].isin(self.common_emotions), self.feature_cols
        ].values)

        self.ref_scaler.fit(ref_all)
        self.sub_scaler.fit(sub_all)

        for emotion in self.common_emotions:
            r = _clean(reference_df.loc[reference_df['emotion'] == emotion, self.feature_cols].values)
            s = _clean(subject_df.loc[subject_df['emotion'] == emotion, self.feature_cols].values)

            r_sc = self.ref_scaler.transform(r)
            s_sc = self.sub_scaler.transform(s)

            self.reference_prototypes[emotion] = np.mean(r_sc, axis=0)
            self.subject_prototypes[emotion] = np.mean(s_sc, axis=0)

            print(f"  {emotion}: reference n={len(r)}, subject n={len(s)} — fitting OT map")
            ot_map = pot.da.LinearTransport(reg=self.reg)
            ot_map.fit(Xs=s_sc, Xt=r_sc)
            self.ot_maps[emotion] = ot_map

        self.trained = True
        print("Class-conditional OT transformer trained successfully")

    def _assign_emotions(self, features_scaled):
        """Assign each sample to its nearest subject prototype (scaled space)."""
        protos = np.array([self.subject_prototypes[e] for e in self.common_emotions])
        dists = np.stack([
            np.linalg.norm(features_scaled - protos[i], axis=1)
            for i in range(len(self.common_emotions))
        ], axis=1)
        return np.argmin(dists, axis=1)

    def transform(self, features):
        if not self.trained:
            raise RuntimeError("Transformer not trained. Call fit() first.")
        if isinstance(features, pd.DataFrame):
            features = features[self.feature_cols].values
        scaled = self.sub_scaler.transform(features)
        assignments = self._assign_emotions(scaled)
        result_scaled = np.zeros_like(scaled)
        for i, emotion in enumerate(self.common_emotions):
            mask = assignments == i
            if mask.any():
                result_scaled[mask] = self.ot_maps[emotion].transform(Xs=scaled[mask])
        return self.ref_scaler.inverse_transform(result_scaled)

    def transform_prototypes(self):
        """Return transformed subject prototypes in raw reference space."""
        trans_raw = []
        for emotion in self.common_emotions:
            proto_sc = self.subject_prototypes[emotion].reshape(1, -1)
            trans_sc = self.ot_maps[emotion].transform(Xs=proto_sc)
            trans_raw.append(self.ref_scaler.inverse_transform(trans_sc)[0])
        return np.array(trans_raw)

    def save(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'class': 'ClassConditionalOT',
            'reg': self.reg,
            'ot_maps': self.ot_maps,
            'sub_scaler': self.sub_scaler,
            'ref_scaler': self.ref_scaler,
            'feature_cols': self.feature_cols,
            'common_emotions': self.common_emotions,
            'subject_prototypes': self.subject_prototypes,
            'reference_prototypes': self.reference_prototypes,
        }, filepath)
        print(f"Saved ClassConditionalOT transformer to {filepath}")

    @classmethod
    def load(cls, data):
        obj = cls(reg=data['reg'])
        obj.ot_maps = data['ot_maps']
        obj.sub_scaler = data['sub_scaler']
        obj.ref_scaler = data['ref_scaler']
        obj.feature_cols = data['feature_cols']
        obj.common_emotions = data['common_emotions']
        obj.subject_prototypes = data['subject_prototypes']
        obj.reference_prototypes = data['reference_prototypes']
        obj.trained = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# CORAL / Euclidean alignment transformer (global covariance whitening+recoloring)
# ─────────────────────────────────────────────────────────────────────────────

class CORALTransformer:
    """
    Unsupervised global feature alignment via CORAL (Correlation Alignment,
    Sun & Saenko 2016) -- the closed-form linear-algebra cousin of Euclidean
    Alignment used for cross-subject transfer in BCI.

    Whitens the subject distribution to identity covariance, then re-colors
    it with the reference distribution's covariance (both scaled to zero
    mean by sub_scaler/ref_scaler beforehand, so no separate mean-centering
    step is needed -- ref_scaler's inverse_transform re-adds the reference
    mean at the end).

    Like `ot_global`, this never looks at the emotion label when computing
    the map -- common_emotions only selects which pooled samples to estimate
    the covariance from, so no per-class calibration data is strictly
    required. Unlike `ot_global` (which solves for the optimal-transport
    Gaussian map), this uses the classic non-optimal whitening/recoloring
    closed form -- in practice the two behave similarly, since both are
    class-blind, moment-matching linear maps (see `ClassConditionalOT` for
    the class-aware alternative).
    """

    def __init__(self, reg=1e-5):
        self.reg = reg
        self.A = None  # Cs^{-1/2} @ Ct^{1/2}, applied to scaled subject features
        self.sub_scaler = StandardScaler()
        self.ref_scaler = StandardScaler()
        self.feature_cols = []
        self.common_emotions = []
        self.subject_prototypes = {}
        self.reference_prototypes = {}
        self.trained = False

    def fit(self, reference_df, subject_df):
        self.feature_cols, self.common_emotions = _common_setup(reference_df, subject_df)

        ref_all = _clean(reference_df.loc[
            reference_df['emotion'].isin(self.common_emotions), self.feature_cols
        ].values)
        sub_all = _clean(subject_df.loc[
            subject_df['emotion'].isin(self.common_emotions), self.feature_cols
        ].values)

        self.ref_scaler.fit(ref_all)
        self.sub_scaler.fit(sub_all)

        for emotion in self.common_emotions:
            r = _clean(reference_df.loc[reference_df['emotion'] == emotion, self.feature_cols].values)
            s = _clean(subject_df.loc[subject_df['emotion'] == emotion, self.feature_cols].values)
            self.reference_prototypes[emotion] = np.mean(self.ref_scaler.transform(r), axis=0)
            self.subject_prototypes[emotion] = np.mean(self.sub_scaler.transform(s), axis=0)
            print(f"  {emotion}: reference n={len(r)}, subject n={len(s)}")

        X_ref_sc = self.ref_scaler.transform(ref_all)
        X_sub_sc = self.sub_scaler.transform(sub_all)
        n_features = X_sub_sc.shape[1]

        print(f"\nFitting CORAL on {len(X_sub_sc)} subject -> {len(X_ref_sc)} reference samples "
              f"(class-blind, {n_features} features)")

        Cs = np.cov(X_sub_sc, rowvar=False) + self.reg * np.eye(n_features)
        Ct = np.cov(X_ref_sc, rowvar=False) + self.reg * np.eye(n_features)

        self.A = _sym_inv_sqrt(Cs) @ _sym_sqrt(Ct)
        self.trained = True
        print("CORAL transformer trained successfully")

    def transform(self, features):
        if not self.trained:
            raise RuntimeError("Transformer not trained. Call fit() first.")
        if isinstance(features, pd.DataFrame):
            features = features[self.feature_cols].values
        scaled = self.sub_scaler.transform(features)
        aligned_scaled = scaled @ self.A
        return self.ref_scaler.inverse_transform(aligned_scaled)

    def transform_prototypes(self):
        """Return transformed subject prototypes in raw reference space."""
        sub_protos_scaled = np.array([self.subject_prototypes[e] for e in self.common_emotions])
        aligned_scaled = sub_protos_scaled @ self.A
        return self.ref_scaler.inverse_transform(aligned_scaled)

    def save(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'class': 'CORAL',
            'reg': self.reg,
            'A': self.A,
            'sub_scaler': self.sub_scaler,
            'ref_scaler': self.ref_scaler,
            'feature_cols': self.feature_cols,
            'common_emotions': self.common_emotions,
            'subject_prototypes': self.subject_prototypes,
            'reference_prototypes': self.reference_prototypes,
        }, filepath)
        print(f"Saved CORAL transformer to {filepath}")

    @classmethod
    def load(cls, data):
        obj = cls(reg=data['reg'])
        obj.A = data['A']
        obj.sub_scaler = data['sub_scaler']
        obj.ref_scaler = data['ref_scaler']
        obj.feature_cols = data['feature_cols']
        obj.common_emotions = data['common_emotions']
        obj.subject_prototypes = data['subject_prototypes']
        obj.reference_prototypes = data['reference_prototypes']
        obj.trained = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def load_transformer(filepath):
    """Load any transformer type from a .pkl file."""
    data = joblib.load(filepath)
    class_name = data.get('class', 'Ridge')
    if class_name == 'LinearOT':
        return LinearOTTransformer.load(data)
    elif class_name == 'ClassConditionalOT':
        return ClassConditionalOTTransformer.load(data)
    elif class_name == 'CORAL':
        return CORALTransformer.load(data)
    else:
        return PrototypeAlignmentTransformer.load(data)


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
        '--method',
        choices=['ridge', 'ot_global', 'ot_classconditional', 'coral'],
        default='ridge',
        help='Alignment method: ridge (default), ot_global, ot_classconditional, coral'
    )
    parser.add_argument(
        '--alpha', type=float, default=10.0,
        help='Ridge regularization strength (default: 10.0, Ridge only)'
    )
    parser.add_argument(
        '--reg', type=float, default=1e-5,
        help='OT/CORAL regularization strength (default: 1e-5, OT and CORAL methods only)'
    )
    parser.add_argument(
        '--n-features', type=int, default=None,
        help='Keep only top N features by ANOVA F-test (default: use all, Ridge only)'
    )
    parser.add_argument(
        '--output',
        default='models/subject_alignment_transformer.pkl',
        help='Output path for trained transformer'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Subject-to-Reference Prototype Alignment")
    print(f"Method: {args.method}")
    print("=" * 60)

    ref_df = pd.read_csv(args.reference)
    sub_df = pd.read_csv(args.subject)
    print(f"Reference: {len(ref_df)} samples from {args.reference}")
    print(f"Subject:   {len(sub_df)} samples from {args.subject}")

    if args.method == 'ridge':
        transformer = PrototypeAlignmentTransformer(alpha=args.alpha, n_features=args.n_features)
    elif args.method == 'ot_global':
        transformer = LinearOTTransformer(reg=args.reg)
    elif args.method == 'ot_classconditional':
        transformer = ClassConditionalOTTransformer(reg=args.reg)
    elif args.method == 'coral':
        transformer = CORALTransformer(reg=args.reg)

    transformer.fit(ref_df, sub_df)
    transformer.save(args.output)

    print("\nDone.")


if __name__ == '__main__':
    main()
