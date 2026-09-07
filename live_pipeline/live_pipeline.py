"""Live pipeline: biodata (OSC in) -> W vector (OSC out to Autolume).

Receives raw biodata samples over OSC, feeds them through
OnlineFeatureExtractor.push() (see modules/online_feature_extractor.py),
runs each newly-finalized feature row through a pre-trained cross-subject
alignment transformer and the Stage 5 biodata->W regressor, and sends the
resulting 512-dim W vector out over OSC to Autolume
(/home/tats/Documents/workspace/autolume -- a separate live StyleGAN
performance app, not part of this repo). Autolume renders and displays the
face itself; this script never touches StyleGAN2 or does any rendering.

Protocol (this script's own design -- see PIPELINE.md's live-readiness
section; no established hardware protocol existed to match):
  In:  one OSC message per raw sample, 3 floats [heart, gsr, respiration].
  Out: one OSC message per finalized feature row, 512 floats (the W
       vector). Autolume's latent-vector OSC handler expects exactly this
       shape -- set its "vec" OSC address to match --out-address, and
       leave its "project" checkbox UNCHECKED (our W is already W-space,
       not Z-space, so it must not be re-mapped).

Usage (from the repo root):
    python live_pipeline/live_pipeline.py \
        --regressor latent_pipeline/outputs/stage5_regressor_online/regressor.joblib \
        --transformer biodata_pipeline/models/transformer_ot_classconditional_online.pkl \
        --calibration-csv live_pipeline/data/erin_calibration_segment.csv

Test without real hardware or Autolume: run replay_biodata_as_osc.py (same
folder) in another process to feed this script, and pass --log-only to
print outgoing W vectors instead of (or alongside) sending OSC.

Run with biodata_pipeline/venv's interpreter -- needs OnlineFeatureExtractor
and the alignment transformer (sklearn), not torch/StyleGAN.
"""

import argparse
import signal
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

REPO_ROOT = Path(__file__).resolve().parent.parent
BIODATA_PIPELINE_DIR = REPO_ROOT / 'biodata_pipeline'
sys.path.insert(0, str(BIODATA_PIPELINE_DIR))

from modules.online_feature_extractor import OnlineFeatureExtractor
from modules.alignment_transformer import load_transformer


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Live biodata -> W pipeline: OSC in, OSC out to Autolume',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--regressor', required=True, help='Path to regressor.joblib')
    parser.add_argument('--transformer', required=True, help='Path to alignment transformer .pkl')
    parser.add_argument('--calibration-csv', default=None,
                        help='Raw biodata CSV to prime filter state + SCR threshold from '
                             '(heart/gsr/respiration columns). Must be a recording that ends '
                             'where the live stream begins (a real calibration period followed '
                             'by the live one) -- NOT the same recording you then replay as '
                             '"live" for testing. Replaying data that was already used for '
                             'calibration creates a filter-state discontinuity (the filters '
                             'jump from "just finished this recording" back to its start) that '
                             'produces wildly implausible output; this is a testing-setup '
                             'mistake, not a pipeline bug -- confirmed by testing. If omitted, '
                             "falls back to OnlineFeatureExtractor's default of using the live "
                             'stream\'s own first 30s.')
    parser.add_argument('--sampling-rate', type=int, default=100, help='Hz')
    parser.add_argument('--in-host', default='127.0.0.1')
    parser.add_argument('--in-port', type=int, default=9000)
    parser.add_argument('--in-address', default='/crocodile/biodata',
                        help='OSC address this script listens on for [heart, gsr, respiration]')
    parser.add_argument('--out-host', default='127.0.0.1', help="Autolume's host")
    parser.add_argument('--out-port', type=int, default=1338, help="Autolume's default OSC input port")
    parser.add_argument('--out-address', default='/crocodile/w',
                        help='OSC address to send the 512-float W vector to -- must match '
                             "the address configured in Autolume's latent-vector OSC menu")
    parser.add_argument('--log-only', action='store_true',
                        help='Print outgoing W vectors instead of sending OSC (no Autolume needed)')
    return parser


def main():
    args = build_arg_parser().parse_args()

    print(f"Loading regressor from {args.regressor}")
    reg_data = joblib.load(args.regressor)
    model, scaler = reg_data['model'], reg_data['scaler']
    feature_cols, w_cols = reg_data['feature_cols'], reg_data['w_cols']
    print(f"  {reg_data['model_type']}, {len(feature_cols)} features -> {len(w_cols)} W dims")

    print(f"Loading alignment transformer from {args.transformer}")
    transformer = load_transformer(args.transformer)
    print(f"  {transformer.__class__.__name__}, emotions: {transformer.common_emotions}")

    missing = [c for c in feature_cols if c not in transformer.feature_cols]
    if missing:
        raise ValueError(f"Transformer is missing regressor's expected features: {missing}")

    extractor = OnlineFeatureExtractor(sampling_rate=args.sampling_rate)
    if args.calibration_csv:
        print(f"Calibrating from {args.calibration_csv}")
        calibration_df = pd.read_csv(args.calibration_csv)
        extractor.calibrate(calibration_df)
        print("  Calibration complete")
    else:
        print("No --calibration-csv given -- using the live stream's own first 30s "
              "(OnlineFeatureExtractor's default)")

    osc_client = None if args.log_only else SimpleUDPClient(args.out_host, args.out_port)
    signal_cols = {'eda': 'gsr', 'ppg': 'heart', 'resp': 'respiration'}

    n_rows_sent = 0
    n_rows_skipped = 0

    def on_biodata(unused_address, *osc_args):
        nonlocal n_rows_sent, n_rows_skipped
        if len(osc_args) != 3:
            print(f"  WARNING: expected 3 args [heart, gsr, respiration], got {len(osc_args)} -- dropping")
            return
        heart, gsr, respiration = osc_args
        row_df = pd.DataFrame([{'heart': heart, 'gsr': gsr, 'respiration': respiration}])

        for row in extractor.push(row_df, signal_cols=signal_cols, feature_interval_s=1.0):
            aligned = transformer.transform(pd.DataFrame([row])[transformer.feature_cols])
            aligned_df = pd.DataFrame(aligned, columns=transformer.feature_cols)
            X = aligned_df[feature_cols].values

            if not np.isfinite(X).all():
                n_rows_skipped += 1
                print(f"  Skipping row {n_rows_sent + n_rows_skipped}: non-finite feature values "
                      f"(still warming up? {n_rows_skipped} skipped so far)")
                continue

            w = model.predict(scaler.transform(X))[0]
            if osc_client is not None:
                osc_client.send_message(args.out_address, w.tolist())
            else:
                print(f"  row {n_rows_sent}: W norm={np.linalg.norm(w):.2f} "
                      f"first5={np.round(w[:5], 2).tolist()}")
            n_rows_sent += 1

    dispatcher = Dispatcher()
    dispatcher.map(args.in_address, on_biodata)
    server = BlockingOSCUDPServer((args.in_host, args.in_port), dispatcher)

    def handle_shutdown(*_):
        print(f"\nShutting down. Rows sent: {n_rows_sent}, skipped: {n_rows_skipped}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)

    print(f"Listening for biodata on {args.in_host}:{args.in_port}{args.in_address}")
    if osc_client is not None:
        print(f"Sending W vectors to {args.out_host}:{args.out_port}{args.out_address}")
    else:
        print("--log-only: printing W vectors instead of sending OSC")
    server.serve_forever()


if __name__ == '__main__':
    main()
