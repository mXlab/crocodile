"""Generate a synthetic raw biodata CSV (heart/gsr/respiration) for testing
the live pipeline without a real sensor recording.

Real biodata recordings can't be shared on GitHub for privacy reasons (see
INSTALL.md), so a fresh checkout has nothing for replay_biodata_as_osc.py to
replay out of the box. This script uses NeuroKit2's own physiological
signal simulators (ppg_simulate, eda_simulate, rsp_simulate) to produce a
CSV with the same columns replay_biodata_as_osc.py expects (heart, gsr,
respiration) -- enough to exercise the full OSC plumbing, calibration, and
feature-extraction path end-to-end. The resulting W vectors are NOT
physiologically meaningful (no real subject/emotion signal underlies them)
-- this is for testing that data flows correctly, not for producing
plausible expressions.

The three signals are simulated in NeuroKit2's own physiological units and
then linearly rescaled to an arbitrary raw-integer range (--heart-range /
--gsr-range / --respiration-range, default 0-4095, a common 12-bit-ADC-like
span) -- NOT calibrated to any specific sensor hardware, since
OnlineFeatureExtractor self-calibrates its filters/thresholds from
whatever range the data it's given happens to be in (see
biodata_pipeline/modules/online_feature_extractor.py).

Usage (from the repo root, biodata_pipeline/venv):
    # Two independent, non-overlapping segments (different --seed), mirroring
    # live_pipeline/data/erin_calibration_segment.csv / erin_live_segment.csv:
    python live_pipeline/generate_synthetic_biodata.py \\
        --duration 60 --seed 1 --output live_pipeline/data/synthetic_calibration.csv
    python live_pipeline/generate_synthetic_biodata.py \\
        --duration 180 --seed 2 --output live_pipeline/data/synthetic_live.csv
"""

import argparse
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd


def _rescale(signal, lo, hi):
    signal = np.asarray(signal, dtype=float)
    smin, smax = signal.min(), signal.max()
    if smax - smin < 1e-9:
        return np.full_like(signal, (lo + hi) / 2)
    return lo + (signal - smin) / (smax - smin) * (hi - lo)


def main():
    parser = argparse.ArgumentParser(
        description='Generate a synthetic raw biodata CSV (NeuroKit2-based) for testing '
                    'the live pipeline without a real sensor recording',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--duration', type=int, default=60,
                        help='Signal duration in seconds (keep at 10+ -- NeuroKit2\'s respiration '
                             'simulator behaves oddly and can under-run this on very short durations)')
    parser.add_argument('--sampling-rate', type=int, default=100,
                        help="Hz -- must match live_pipeline.py's --sampling-rate")
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed -- use a different seed per generated segment so '
                             'calibration and live data never overlap')
    parser.add_argument('--heart-rate', type=float, default=75.0, help='Simulated heart rate (bpm)')
    parser.add_argument('--respiratory-rate', type=float, default=15.0, help='Simulated respiratory rate (breaths/min)')
    parser.add_argument('--scr-number', type=int, default=None,
                        help='Number of simulated EDA skin-conductance responses '
                             '(default: roughly one every 20s of --duration)')
    parser.add_argument('--heart-range', type=float, nargs=2, default=(0, 4095), metavar=('MIN', 'MAX'),
                        help='Raw output range for the heart/PPG column (arbitrary, not hardware-calibrated)')
    parser.add_argument('--gsr-range', type=float, nargs=2, default=(0, 4095), metavar=('MIN', 'MAX'),
                        help='Raw output range for the gsr/EDA column')
    parser.add_argument('--respiration-range', type=float, nargs=2, default=(0, 4095), metavar=('MIN', 'MAX'),
                        help='Raw output range for the respiration column')
    args = parser.parse_args()

    scr_number = args.scr_number if args.scr_number is not None else max(1, int(args.duration // 20))

    ppg = nk.ppg_simulate(duration=args.duration, sampling_rate=args.sampling_rate,
                           heart_rate=args.heart_rate, random_state=args.seed)
    eda = nk.eda_simulate(duration=args.duration, sampling_rate=args.sampling_rate,
                           scr_number=scr_number, random_state=args.seed)
    rsp = nk.rsp_simulate(duration=args.duration, sampling_rate=args.sampling_rate,
                           respiratory_rate=args.respiratory_rate, random_state=args.seed)

    n = min(len(ppg), len(eda), len(rsp))
    df = pd.DataFrame({
        'heart': _rescale(ppg[:n], *args.heart_range).round().astype(int),
        'gsr': _rescale(eda[:n], *args.gsr_range).round().astype(int),
        'respiration': _rescale(rsp[:n], *args.respiration_range).round().astype(int),
    })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} samples ({n / args.sampling_rate:.1f}s at {args.sampling_rate}Hz) "
          f"to {output_path} (seed={args.seed})")


if __name__ == '__main__':
    main()
