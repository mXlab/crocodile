"""
Held-out-emotion generalization test for cross-subject alignment transformers.

The closed-set benchmarks in train_transformer.py / validate_transformer.py
calibrate and evaluate on the *same* emotions, which can't tell us how a
transformer behaves once a live participant leaves the calibrated set --
the actual deployment scenario (calibrate on a handful of elicited
emotions, then observe the participant living freely afterward).

For each emotion h shared between subject and reference:
  1. Fit each transformer on all shared emotions EXCEPT h. Only the
     subject dataframe is filtered -- the reference dataframe is left
     intact, since its real h-labeled rows are the ground truth we score
     against.
  2. Transform the subject's held-out h rows anyway (transform() doesn't
     care whether a label was present during fit).
  3. Score against:
     - the actress' TRUE h prototype (never seen by the transformer)
       -> held-out RMSE_norm, normalized by the mean pairwise distance
          among the FULL set of known reference prototypes, so it's on
          the same scale as the closed-set RMSE_norm benchmark and the
          two numbers can be compared directly.
     - a Random Forest trained on the reference's full known-emotion data
       -> whether transformed held-out samples still land in h's correct
          decision region despite the alignment map never having seen h.
     - (class-conditional OT only) which known emotion's map each
       held-out sample got routed to, to make the "wrong bucket" failure
       mode visible and countable rather than theoretical.

Usage (from biodata_pipeline/):
    python scripts/validate_heldout_emotion.py \\
        --reference data/processed/laurence_main_features.csv \\
        --subject data/processed/erin_2026-02-09_features.csv
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_transformer import (
    PrototypeAlignmentTransformer,
    LinearOTTransformer,
    ClassConditionalOTTransformer,
    CORALTransformer,
    METADATA_COLS,
    _clean,
)

METHOD_FACTORIES = {
    'ridge': lambda alpha, reg: PrototypeAlignmentTransformer(alpha=alpha),
    'ot_global': lambda alpha, reg: LinearOTTransformer(reg=reg),
    'ot_classconditional': lambda alpha, reg: ClassConditionalOTTransformer(reg=reg),
    'coral': lambda alpha, reg: CORALTransformer(reg=reg),
}


def _feature_cols(reference_df, subject_df):
    ref_features = set(c for c in reference_df.columns if c not in METADATA_COLS)
    sub_features = set(c for c in subject_df.columns if c not in METADATA_COLS)
    return sorted(ref_features & sub_features)


def _prototype(df, emotion, feature_cols):
    rows = _clean(df.loc[df['emotion'] == emotion, feature_cols].values)
    return np.mean(rows, axis=0), len(rows)


def _anova_select(reference_df, calib_emotions, feature_cols, n_features):
    """Top-n_features by ANOVA F-test on the reference's calibration-emotion
    data only (never the held-out emotion) -- same convention as Ridge's
    existing --n-features option in train_transformer.py."""
    ref_sel = reference_df.loc[reference_df['emotion'].isin(calib_emotions)]
    X_sel = ref_sel[feature_cols].values
    y_sel = ref_sel['emotion'].values
    valid = ~(np.isnan(X_sel).any(axis=1) | np.isinf(X_sel).any(axis=1))
    X_sel, y_sel = X_sel[valid], y_sel[valid]
    f_scores, _ = f_classif(X_sel, y_sel)
    # NaN (constant-feature) scores sort to the end ascending, which `[::-1]`
    # would otherwise put first -- rank them lowest instead.
    f_scores = np.nan_to_num(f_scores, nan=-np.inf)
    top_idx = np.argsort(f_scores)[::-1][:n_features]
    return sorted(feature_cols[i] for i in top_idx)


def run_heldout_test(reference_df, subject_df, methods, alpha, reg, n_features=None):
    all_feature_cols = _feature_cols(reference_df, subject_df)
    ref_emotions = set(reference_df['emotion'].dropna().unique())
    sub_emotions = set(subject_df['emotion'].dropna().unique())
    full_common = sorted(ref_emotions & sub_emotions)

    if len(full_common) < 3:
        raise ValueError(
            f"Need >= 3 shared emotions to hold one out and still calibrate "
            f"on >= 2, found {len(full_common)}: {full_common}"
        )

    print(f"Shared emotions: {full_common}")
    if n_features:
        print(f"ANOVA feature selection: top {n_features} of {len(all_feature_cols)} "
              f"features (re-selected per held-out split, from calibration emotions only)\n")
    else:
        print(f"Using all {len(all_feature_cols)} features (no selection)\n")

    results = {}

    for h in full_common:
        print("=" * 70)
        print(f"HOLDING OUT: {h}")
        print("=" * 70)

        calib_emotions = [e for e in full_common if e != h]

        # Feature selection (if requested) is re-done per split, using only
        # the calibration emotions' reference data -- the held-out emotion
        # never influences which features get picked.
        if n_features:
            feature_cols = _anova_select(reference_df, calib_emotions, all_feature_cols, n_features)
        else:
            feature_cols = all_feature_cols

        # Prototypes/scale recomputed in this split's feature space so
        # RMSE_norm stays an apples-to-apples ratio within the split, even
        # though the selected columns can differ split to split.
        full_ref_protos = {e: _prototype(reference_df, e, feature_cols)[0] for e in full_common}
        pairwise = [
            np.linalg.norm(full_ref_protos[a] - full_ref_protos[b])
            for i, a in enumerate(full_common) for b in full_common[i + 1:]
        ]
        scale = float(np.mean(pairwise))

        # RF trained on the reference's full known-emotion data in this
        # split's feature space.
        ref_mask = reference_df['emotion'].isin(full_common)
        X_ref_all = reference_df.loc[ref_mask, feature_cols].values
        y_ref_all = reference_df.loc[ref_mask, 'emotion'].values
        valid = ~(np.isnan(X_ref_all).any(axis=1) | np.isinf(X_ref_all).any(axis=1))
        X_ref_all, y_ref_all = X_ref_all[valid], y_ref_all[valid]
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_ref_all, y_ref_all)

        # Restrict what fit() sees to this split's selected columns, so
        # transformer.feature_cols comes out equal to `feature_cols` above.
        meta_cols = [c for c in reference_df.columns if c in METADATA_COLS]
        ref_reduced = reference_df[meta_cols + feature_cols]
        subject_calib = subject_df.loc[subject_df['emotion'] != h, meta_cols + feature_cols]

        held_out_rows = _clean(
            subject_df.loc[subject_df['emotion'] == h, feature_cols].values
        )
        if len(held_out_rows) == 0:
            print(f"  No valid held-out samples for {h}, skipping")
            continue

        true_proto = full_ref_protos[h]
        results[h] = {'_split_scale': scale, '_split_feature_cols': feature_cols}

        for method in methods:
            transformer = METHOD_FACTORIES[method](alpha, reg)
            try:
                transformer.fit(ref_reduced, subject_calib)
            except ValueError as e:
                print(f"  [{method}] fit failed: {e}")
                continue

            transformed = transformer.transform(held_out_rows)

            # Held-out RMSE, normalized on the fixed full-set scale.
            dists = np.linalg.norm(transformed - true_proto, axis=1)
            rmse = float(np.sqrt(np.mean(dists ** 2)))
            rmse_norm = rmse / scale if scale > 0 else float('inf')

            # RF check: does it still land in h's correct decision region?
            y_pred = rf.predict(transformed)
            recall_h = float((y_pred == h).mean())

            entry = {
                'n_held_out': int(len(held_out_rows)),
                'calibrated_on': transformer.common_emotions,
                'held_out_rmse_norm': rmse_norm,
                'rf_recall_on_held_out': recall_h,
                'rf_prediction_distribution': {
                    str(lbl): float((y_pred == lbl).mean())
                    for lbl in sorted(set(y_pred.tolist()) | {h})
                },
            }

            line = (f"  [{method:20s}] calibrated on {transformer.common_emotions} | "
                    f"held-out RMSE_norm={rmse_norm:.3f}  RF recall({h})={recall_h:.1%}")

            if method == 'ot_classconditional':
                assignments = transformer._assign_emotions(
                    transformer.sub_scaler.transform(held_out_rows)
                )
                assigned_emotions = [transformer.common_emotions[i] for i in assignments]
                routed_to = dict(Counter(assigned_emotions))
                entry['routed_to'] = routed_to
                line += f"  routed_to={routed_to}"

            results[h][method] = entry
            print(line)

        print()

    return results, full_common


def main():
    parser = argparse.ArgumentParser(
        description="Held-out-emotion generalization test for alignment transformers"
    )
    parser.add_argument('--reference', required=True, help='Reference (actress) features CSV')
    parser.add_argument('--subject', required=True, help='New subject features CSV')
    parser.add_argument(
        '--methods', nargs='+',
        default=['ridge', 'ot_global', 'ot_classconditional', 'coral'],
        choices=list(METHOD_FACTORIES.keys()),
        help='Which methods to test (default: all four)'
    )
    parser.add_argument('--alpha', type=float, default=10.0, help='Ridge regularization strength')
    parser.add_argument('--reg', type=float, default=1e-5, help='OT/CORAL regularization strength')
    parser.add_argument(
        '--n-features', type=int, default=None,
        help='Keep only top N features by ANOVA F-test, re-selected per held-out split from '
             'calibration-emotion reference data only (default: use all features)'
    )
    parser.add_argument(
        '--output-dir', default='reports/heldout_emotion',
        help='Directory for the results JSON'
    )
    args = parser.parse_args()

    reference_df = pd.read_csv(args.reference)
    subject_df = pd.read_csv(args.subject)
    print(f"Reference: {len(reference_df)} samples from {args.reference}")
    print(f"Subject:   {len(subject_df)} samples from {args.subject}\n")

    results, full_common = run_heldout_test(
        reference_df, subject_df, args.methods, args.alpha, args.reg, args.n_features
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'heldout_emotion_results.json'
    with open(out_path, 'w') as f:
        json.dump({
            'n_features': args.n_features,
            'full_common_emotions': full_common,
            'results': results,
        }, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()
