"""One-time offline preparation of the synthetic test fixtures committed
under live_pipeline/data/ (see the .gitignore exceptions there for why
these two specific generated files are safe to publish unlike everything
else *.csv/*.pkl normally excludes). Regenerate only if the online feature
schema (biodata_pipeline/modules/online_feature_extractor.py) or
generate_synthetic_biodata.py's output format change.

Produces two files, both derived entirely from NeuroKit2-simulated data --
no real biometric recording is touched at any point:
  - live_pipeline/data/synthetic_test_transformer.pkl -- a ZScoreTransformer
    pre-fit on two independently generated synthetic biodata segments (a
    "reference"/actress stand-in and a "visitor" calibration), the same way
    live_pipeline.py's own live calibration fit would, just done once
    offline. Passing this via --transformer lets a session skip the
    calibration phase entirely (straight from --start-session to
    --start-live) -- see README.md's "Simplest possible test" section.
  - live_pipeline/data/synthetic_test_live.csv -- a third synthetic segment,
    meant to be replayed as the "visitor" during that same simple test.

Reuses generate_synthetic_biodata.py (as a subprocess, so its own
NeuroKit2/rescaling logic isn't duplicated here) and
OnlineFeatureExtractor.process_session() (the exact extraction path
live_pipeline.py's calibration uses) for the reference/calibration ->
features step, then fits and saves a ZScoreTransformer directly --
the simplest of the five alignment methods (see PIPELINE.md's "Live
per-visitor alignment fit" section), chosen because it needs no emotion
labels and is what an un-cued single-segment calibration falls back to
anyway.

RAW_RANGES below matter a lot: generate_synthetic_biodata.py's own
defaults (0-4095 on every channel) don't resemble the real sensor rig's
actual ADC output at all, which used to make the extracted features land
1.5-8x outside real per-feature scales (e.g. eda.tonic_level ~1130 vs a
real recording's ~320, respiratory.amplitude_mean_10s ~3590 vs ~430) --
enough to push the transformed/regressed output into visibly out-of-domain
StyleGAN2 latents. RAW_RANGES were picked by inspecting the aggregate
min/max/mean/std of one real recording's raw heart/gsr/respiration
columns -- never the recording itself, never committed, and only used
here, once, to calibrate these six numbers -- then hand-tuned so the
*extracted feature* scales land close to that recording's. Only re-derive
these if the sensor hardware changes.

Usage:
    live_pipeline/run_prepare_synthetic_test_fixtures.sh
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from biodata_pipeline.modules.online_feature_extractor import OnlineFeatureExtractor
from biodata_pipeline.modules.alignment_transformer import ZScoreTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATE_SCRIPT = REPO_ROOT / 'live_pipeline' / 'generate_synthetic_biodata.py'

# Realistic ADC ranges (see module docstring) -- applied to every generated
# segment so extracted features land in real-world scale, not the generator's
# arbitrary 0-4095 default.
RAW_RANGES = {
    'heart_range': (0, 1024),
    'gsr_range': (100, 800),
    'respiration_range': (14450, 14950),
}


def generate_raw(output_path, duration, seed, **kwargs):
    cmd = [sys.executable, str(GENERATE_SCRIPT), '--output', str(output_path),
           '--duration', str(duration), '--seed', str(seed)]
    for key, value in {**RAW_RANGES, **kwargs}.items():
        if isinstance(value, tuple):
            cmd += [f'--{key.replace("_", "-")}'] + [str(v) for v in value]
        else:
            cmd += [f'--{key.replace("_", "-")}', str(value)]
    subprocess.run(cmd, check=True)


def extract_features(raw_csv_path):
    df = pd.read_csv(raw_csv_path)
    return OnlineFeatureExtractor(sampling_rate=100).process_session(df)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare the synthetic transformer + replay CSV used by the no-calibration simple test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output-dir', default=str(REPO_ROOT / 'live_pipeline' / 'data'))
    parser.add_argument('--reference-duration', type=int, default=90,
                         help='Seconds of synthetic "actress" reference data')
    parser.add_argument('--calibration-duration', type=int, default=60,
                         help='Seconds of synthetic "visitor" calibration data')
    parser.add_argument('--live-duration', type=int, default=60,
                         help='Seconds of synthetic "visitor" live data (the file actually replayed)')
    parser.add_argument('--reference-seed', type=int, default=1001)
    parser.add_argument('--calibration-seed', type=int, default=1002)
    parser.add_argument('--live-seed', type=int, default=1003)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transformer_path = output_dir / 'synthetic_test_transformer.pkl'
    live_path = output_dir / 'synthetic_test_live.csv'

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        reference_raw = tmp / 'reference_raw.csv'
        calibration_raw = tmp / 'calibration_raw.csv'

        print("Generating synthetic reference (actress stand-in) segment...")
        generate_raw(reference_raw, args.reference_duration, args.reference_seed)
        print("Generating synthetic visitor calibration segment...")
        # A different simulated heart rate is enough to make the two segments
        # meaningfully distinct distributions, so fitting isn't a no-op identity map.
        generate_raw(calibration_raw, args.calibration_duration, args.calibration_seed, heart_rate=85.0)

        print("Extracting online-schema features from both...")
        reference_features = extract_features(reference_raw)
        calibration_features = extract_features(calibration_raw)
        print(f"  reference: {len(reference_features)} feature rows")
        print(f"  calibration: {len(calibration_features)} feature rows")

        transformer = ZScoreTransformer()
        transformer.fit(reference_features, calibration_features)
        transformer.save(transformer_path)
        print(f"Saved transformer: {transformer_path}")

    print("Generating synthetic live (visitor) segment for replay...")
    generate_raw(live_path, args.live_duration, args.live_seed, heart_rate=85.0)
    print(f"Saved replay data: {live_path}")

    print("\nDone. See README.md's \"Simplest possible test\" section for how to use these.")


if __name__ == '__main__':
    main()
