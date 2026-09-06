"""Compare OnlineFeatureExtractor against BatchFeatureExtractor feature-by-
feature on the same raw sessions.

Both extractors produce the identical 51-feature schema (same names, same
window sizes -- see online_feature_extractor.py's module docstring), so
this measures purely how well the online-safe signal derivation
(causal filters + causal peak/onset detection) approximates NeuroKit2's
offline algorithms, feature by feature.

Usage (from biodata_pipeline/):
    python scripts/compare_extractors.py
    python scripts/compare_extractors.py --input emotion_biodata_laurence_main_1.csv --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.batch_feature_extractor import BatchFeatureExtractor
from modules.online_feature_extractor import OnlineFeatureExtractor


def compare_session(session_path, warmup_s=30):
    """Run both extractors on one session, return a per-feature metrics
    DataFrame. `warmup_s` excludes the startup period from metrics (both
    extractors are least reliable there -- NeuroKit2 for different reasons
    than the online one, see PIPELINE.md), so the comparison reflects
    steady-state behavior, not the cold-start region both already
    acknowledge as unreliable."""
    session_df = pd.read_csv(session_path)

    batch_out = BatchFeatureExtractor(sampling_rate=100).process_session(session_df)
    online_out = OnlineFeatureExtractor(sampling_rate=100).process_session(session_df)

    assert len(batch_out) == len(online_out), \
        f"Row count mismatch: batch={len(batch_out)} online={len(online_out)}"
    assert np.allclose(batch_out['timestamp'].values, online_out['timestamp'].values), \
        "Timestamp misalignment between extractors"

    feature_cols = [c for c in batch_out.columns if '.' in c]
    mask = batch_out['timestamp'].values >= warmup_s

    rows = []
    for col in feature_cols:
        b = batch_out[col].values[mask].astype(float)
        o = online_out[col].values[mask].astype(float)
        valid = ~(np.isnan(b) | np.isnan(o))
        n_valid = valid.sum()
        if n_valid < 10:
            rows.append({'feature': col, 'n_valid': n_valid, 'corr': np.nan,
                         'mae': np.nan, 'nmae': np.nan, 'batch_std': np.nan})
            continue
        bv, ov = b[valid], o[valid]
        corr = np.corrcoef(bv, ov)[0, 1] if bv.std() > 0 and ov.std() > 0 else np.nan
        mae = np.mean(np.abs(bv - ov))
        batch_std = bv.std()
        nmae = mae / batch_std if batch_std > 0 else np.nan
        rows.append({'feature': col, 'n_valid': int(n_valid), 'corr': corr,
                     'mae': mae, 'nmae': nmae, 'batch_std': batch_std})

    return pd.DataFrame(rows), batch_out, online_out


def main():
    parser = argparse.ArgumentParser(
        description='Feature-by-feature comparison: OnlineFeatureExtractor vs BatchFeatureExtractor')
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--input', nargs='+', default=None,
                        help='File patterns relative to --data-dir (default: laurence sessions)')
    parser.add_argument('--warmup-s', type=float, default=30,
                        help='Exclude this many seconds from the start of each session')
    parser.add_argument('--output', default='data/processed/extractor_comparison.csv')
    parser.add_argument('--plot', action='store_true',
                        help='Save a time-series comparison plot for a few representative features')
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    if args.input:
        csv_files = []
        for pattern in args.input:
            csv_files.extend(sorted(data_dir.glob(pattern)))
        csv_files = sorted(set(csv_files))
    else:
        csv_files = sorted(data_dir.glob('emotion_biodata_laurence_main_*.csv'))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        sys.exit(1)

    all_metrics = []
    plot_data = {}
    for session_path in csv_files:
        print(f"Processing {session_path.name}...")
        metrics, batch_out, online_out = compare_session(session_path, warmup_s=args.warmup_s)
        metrics['session'] = session_path.stem
        all_metrics.append(metrics)
        if args.plot and session_path == csv_files[0]:
            plot_data = {'batch': batch_out, 'online': online_out, 'name': session_path.stem}

    combined = pd.concat(all_metrics, ignore_index=True)

    print("\n" + "=" * 80)
    print("PER-FEATURE AGREEMENT (pooled mean across sessions, sorted worst-to-best correlation)")
    print("=" * 80)
    pooled = combined.groupby('feature').agg(
        corr_mean=('corr', 'mean'), nmae_mean=('nmae', 'mean'), n_valid=('n_valid', 'sum')
    ).reset_index().sort_values('corr_mean')
    with pd.option_context('display.max_rows', None, 'display.width', 120):
        print(pooled.to_string(index=False))

    print(f"\nOverall mean correlation: {pooled['corr_mean'].mean():.3f}")
    print(f"Overall mean normalized MAE: {pooled['nmae_mean'].mean():.3f}")

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    pooled_path = output_path.with_name(output_path.stem + '_pooled.csv')
    pooled.to_csv(pooled_path, index=False)
    print(f"\nSaved per-session metrics: {output_path}")
    print(f"Saved pooled summary: {pooled_path}")

    if args.plot and plot_data:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        show_features = ['eda.tonic_level', 'eda.phasic_std_10s', 'cardiac.hr_mean_10s',
                         'cardiac.hrv_rmssd_60s', 'respiratory.rate_mean_10s',
                         'respiratory.amplitude_mean_10s']
        batch_out, online_out = plot_data['batch'], plot_data['online']
        fig, axes = plt.subplots(len(show_features), 1, figsize=(12, 2.2 * len(show_features)), sharex=True)
        for ax, col in zip(axes, show_features):
            ax.plot(batch_out['timestamp'], batch_out[col], label='batch (NeuroKit2)', alpha=0.8)
            ax.plot(online_out['timestamp'], online_out[col], label='online', alpha=0.8)
            ax.set_ylabel(col, fontsize=8)
            ax.legend(fontsize=7, loc='upper right')
        axes[-1].set_xlabel('time (s)')
        fig.suptitle(f"Batch vs. online extractor: {plot_data['name']}")
        plt.tight_layout()
        plot_path = output_path.with_name('extractor_comparison_timeseries.png')
        plt.savefig(plot_path, dpi=130)
        plt.close()
        print(f"Saved comparison plot: {plot_path}")


if __name__ == '__main__':
    main()
