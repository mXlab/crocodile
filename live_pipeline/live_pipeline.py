"""Live pipeline: a persistent OSC server, biodata (OSC in) -> W vector
(OSC out to Autolume), running one visitor "session" at a time.

Long-running process: load the regressor/alignment transformer once,
then handle any number of visitors in sequence, each driven through an
explicit session state machine by a small OSC control protocol (see
PIPELINE.md's "Live pipeline" section for the full state diagram):

    IDLE --session/start--> READY --calibration/start--> CALIBRATING
                              |                                |
                              |                       calibration/stop
                              |                                v
                              |                           CALIBRATED
                              |                                |
                              +-----------live/start------------+
                                           |
                                           v
                                         LIVE --calibration/recalibrate--> (stays LIVE)
                                           |
                                (any state) session/end
                                           v
                                         IDLE

Once `calibration/start` fires, every incoming biodata sample is pushed
through OnlineFeatureExtractor.push() continuously for the rest of the
session (through CALIBRATING, CALIBRATED, and LIVE) -- nothing ever stops
feeding the extractor once started. Only what happens to the *returned*
finalized rows differs: discarded during CALIBRATING/CALIBRATED (matches
what calibrate() already does internally), aligned+regressed+sent as W
during LIVE. See SessionState.handle_biodata() below.

Biodata (OSC in, unchanged from before):
  One OSC message per raw sample, 3 floats [heart, gsr, respiration], on
  --in-address (default /crocodile/biodata).

W (OSC out to Autolume, unchanged from before):
  One OSC message per finalized feature row during LIVE, 512 floats, on
  --out-address (default /crocodile/w). Autolume renders and displays the
  face itself; this script never touches StyleGAN2. Autolume's
  latent-vector OSC handler expects exactly this shape -- set its "vec"
  OSC address to match --out-address, and leave its "project" checkbox
  UNCHECKED (our W is already W-space, not Z-space).

Session control (OSC in, new):
  /crocodile/session/start        [session_id: str] (optional)
  /crocodile/calibration/start
  /crocodile/calibration/stop
  /crocodile/live/start
  /crocodile/session/end
  /crocodile/calibration/recalibrate   (LIVE only; does not change phase)
An invalid transition (wrong current state) is logged and ignored, never
crashes the server -- the operator is a human clicking buttons live.

Session status (OSC out, new, separate from the W stream -- a different
consumer, an operator control surface, not Autolume):
  --status-out-address (default /crocodile/session/status), args
  [state: str, session_id: str]. Sent after every successful transition.

Usage (from the repo root):
    python live_pipeline/live_pipeline.py \
        --regressor latent_pipeline/outputs/stage5_regressor_online/regressor.joblib \
        --transformer biodata_pipeline/models/transformer_ot_classconditional_online.pkl \
        --record-dir live_pipeline/data/sessions

Drive it with live_pipeline/session_control.py (session lifecycle) and
replay_biodata_as_osc.py (biodata) to test without real hardware or
Autolume -- pass --log-only to print outgoing W vectors instead of (or
alongside) sending OSC.

Run with biodata_pipeline/venv's interpreter -- needs OnlineFeatureExtractor
and the alignment transformer (sklearn), not torch/StyleGAN.
"""

import argparse
import csv
import signal
import sys
import time
from datetime import datetime
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

SIGNAL_COLS = {'eda': 'gsr', 'ppg': 'heart', 'resp': 'respiration'}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Live biodata -> W pipeline server: OSC session control, OSC out to Autolume',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--regressor', required=True, help='Path to regressor.joblib')
    parser.add_argument('--transformer', required=True, help='Path to alignment transformer .pkl')
    parser.add_argument('--sampling-rate', type=int, default=100, help='Hz')
    parser.add_argument('--in-host', default='127.0.0.1')
    parser.add_argument('--in-port', type=int, default=9000,
                        help='Port for both biodata and session-control OSC messages')
    parser.add_argument('--in-address', default='/crocodile/biodata',
                        help='OSC address this script listens on for [heart, gsr, respiration]')
    parser.add_argument('--out-host', default='127.0.0.1', help="Autolume's host")
    parser.add_argument('--out-port', type=int, default=1338, help="Autolume's default OSC input port")
    parser.add_argument('--out-address', default='/crocodile/w',
                        help='OSC address to send the 512-float W vector to -- must match '
                             "the address configured in Autolume's latent-vector OSC menu")
    parser.add_argument('--status-out-host', default='127.0.0.1',
                        help='Host for session-status broadcasts (an operator control surface, not Autolume)')
    parser.add_argument('--status-out-port', type=int, default=9001)
    parser.add_argument('--status-out-address', default='/crocodile/session/status',
                        help='OSC address to broadcast [state, session_id] to after every transition')
    parser.add_argument('--record-dir', default=None,
                        help='If given, save each session\'s raw biodata (from calibration/start '
                             'onward) to {record-dir}/{session_id}.csv for later retraining/analysis. '
                             'Columns: heart, gsr, respiration, session_phase, timestamp. Reusable '
                             'directly by the existing offline toolchain (extra columns are ignored '
                             'by scripts that only read heart/gsr/respiration). If omitted, no recording.')
    parser.add_argument('--calibration-csv', default=None,
                        help='Optional pre-recorded calibration CSV (heart/gsr/respiration columns) '
                             'to prime every fresh session\'s extractor with immediately at '
                             'session/start, before the state machine even starts (session stays in '
                             'READY afterward, same as without this flag). Use when a suitable '
                             'calibration recording already exists (e.g. a generic baseline, or '
                             'reusing a visitor\'s own earlier recording) and the live calibration/ '
                             'start-stop phase can be skipped entirely -- or still run it afterward '
                             'to layer live-recorded priming on top; calibrate() and push() are the '
                             'same underlying operation, so the two compose cleanly. NOT the same '
                             'recording you then replay as "live" for testing -- reusing it creates a '
                             'filter-state discontinuity (see replay_biodata_as_osc.py\'s docstring).')
    parser.add_argument('--log-only', action='store_true',
                        help='Print outgoing W vectors instead of sending OSC (no Autolume needed)')
    return parser


class SessionState:
    """Owns the current session's lifecycle: phase, the live
    OnlineFeatureExtractor instance, the optional recording file, and
    per-session counters. One instance for the whole server process --
    session/start replaces `extractor` with a fresh one per visitor so no
    state leaks between visitors; the regressor/transformer stay loaded
    for the life of the process (session-agnostic)."""

    VALID_TRANSITIONS = {
        'start_session': {'IDLE'},
        'start_calibration': {'READY'},
        'stop_calibration': {'CALIBRATING'},
        'start_live': {'READY', 'CALIBRATED'},
        'end_session': {'READY', 'CALIBRATING', 'CALIBRATED', 'LIVE'},
        'recalibrate': {'LIVE'},
    }

    def __init__(self, sampling_rate, record_dir, calibration_df, status_client, status_address,
                 model, scaler, feature_cols, w_cols, transformer, osc_client, out_address, log_only):
        self.sampling_rate = sampling_rate
        self.record_dir = Path(record_dir) if record_dir else None
        self.calibration_df = calibration_df
        self.status_client = status_client
        self.status_address = status_address

        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.w_cols = w_cols
        self.transformer = transformer
        self.osc_client = osc_client
        self.out_address = out_address
        self.log_only = log_only

        self.phase = 'IDLE'
        self.session_id = None
        self.extractor = None
        self._record_file = None
        self._record_writer = None
        self.n_rows_sent = 0
        self.n_rows_skipped = 0

    def _check(self, action):
        if self.phase not in self.VALID_TRANSITIONS[action]:
            print(f"  WARNING: '{action}' invalid from phase {self.phase} -- ignored")
            return False
        return True

    def _broadcast_status(self):
        if self.status_client is not None:
            self.status_client.send_message(self.status_address, [self.phase, self.session_id or ''])

    def start_session(self, session_id):
        if not self._check('start_session'):
            return
        self.session_id = session_id or datetime.now().strftime('session_%Y%m%d_%H%M%S')
        self.extractor = OnlineFeatureExtractor(sampling_rate=self.sampling_rate)
        if self.calibration_df is not None:
            print("  Priming from --calibration-csv")
            self.extractor.calibrate(self.calibration_df)
        self.n_rows_sent = 0
        self.n_rows_skipped = 0
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            record_path = self.record_dir / f"{self.session_id}.csv"
            self._record_file = open(record_path, 'w', newline='')
            self._record_writer = csv.writer(self._record_file)
            self._record_writer.writerow(['heart', 'gsr', 'respiration', 'session_phase', 'timestamp'])
            print(f"  Recording to {record_path}")
        self.phase = 'READY'
        print(f"Session started: {self.session_id}")
        self._broadcast_status()

    def start_calibration(self):
        if not self._check('start_calibration'):
            return
        self.phase = 'CALIBRATING'
        print("Calibration started")
        self._broadcast_status()

    def stop_calibration(self):
        if not self._check('stop_calibration'):
            return
        self.phase = 'CALIBRATED'
        print("Calibration stopped")
        self._broadcast_status()

    def start_live(self):
        if not self._check('start_live'):
            return
        self.phase = 'LIVE'
        print("Live output started")
        self._broadcast_status()

    def recalibrate(self):
        if not self._check('recalibrate'):
            return
        self.extractor.recalibrate()
        print("Recalibration triggered -- SCR threshold will refresh over the next ~30s "
              "of live data; W output continues uninterrupted")

    def end_session(self):
        if not self._check('end_session'):
            return
        if self._record_file is not None:
            self._record_file.close()
            self._record_file = None
            self._record_writer = None
        print(f"Session ended: {self.session_id}. Rows sent: {self.n_rows_sent}, "
              f"skipped: {self.n_rows_skipped}")
        self.phase = 'IDLE'
        self.session_id = None
        self.extractor = None
        self._broadcast_status()

    def handle_biodata(self, heart, gsr, respiration):
        if self.phase not in ('CALIBRATING', 'CALIBRATED', 'LIVE'):
            return  # IDLE/READY: no session active yet, or visitor not yet ready to be measured

        if self._record_writer is not None:
            record_phase = 'live' if self.phase == 'LIVE' else 'calib'
            self._record_writer.writerow([heart, gsr, respiration, record_phase, time.time()])

        row_df = pd.DataFrame([{'heart': heart, 'gsr': gsr, 'respiration': respiration}])
        rows = self.extractor.push(row_df, signal_cols=SIGNAL_COLS, feature_interval_s=1.0)

        if self.phase != 'LIVE':
            return  # CALIBRATING/CALIBRATED: keep priming state, discard finalized rows

        for row in rows:
            aligned = self.transformer.transform(pd.DataFrame([row])[self.transformer.feature_cols])
            aligned_df = pd.DataFrame(aligned, columns=self.transformer.feature_cols)
            X = aligned_df[self.feature_cols].values

            if not np.isfinite(X).all():
                self.n_rows_skipped += 1
                print(f"  Skipping row {self.n_rows_sent + self.n_rows_skipped}: non-finite feature "
                      f"values (still warming up? {self.n_rows_skipped} skipped so far)")
                continue

            w = self.model.predict(self.scaler.transform(X))[0]
            if self.osc_client is not None:
                self.osc_client.send_message(self.out_address, w.tolist())
            if self.log_only:
                print(f"  row {self.n_rows_sent}: W norm={np.linalg.norm(w):.2f} "
                      f"first5={np.round(w[:5], 2).tolist()}")
            self.n_rows_sent += 1


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

    calibration_df = None
    if args.calibration_csv:
        print(f"Loading calibration CSV from {args.calibration_csv}")
        calibration_df = pd.read_csv(args.calibration_csv)

    osc_client = None if args.log_only else SimpleUDPClient(args.out_host, args.out_port)
    status_client = SimpleUDPClient(args.status_out_host, args.status_out_port)

    session = SessionState(
        sampling_rate=args.sampling_rate, record_dir=args.record_dir, calibration_df=calibration_df,
        status_client=status_client, status_address=args.status_out_address,
        model=model, scaler=scaler, feature_cols=feature_cols, w_cols=w_cols,
        transformer=transformer, osc_client=osc_client, out_address=args.out_address,
        log_only=args.log_only)

    def on_biodata(unused_address, *osc_args):
        if len(osc_args) != 3:
            print(f"  WARNING: expected 3 args [heart, gsr, respiration], got {len(osc_args)} -- dropping")
            return
        session.handle_biodata(*osc_args)

    def on_session_start(unused_address, *osc_args):
        session.start_session(osc_args[0] if osc_args else None)

    def on_calibration_start(unused_address, *_):
        session.start_calibration()

    def on_calibration_stop(unused_address, *_):
        session.stop_calibration()

    def on_live_start(unused_address, *_):
        session.start_live()

    def on_session_end(unused_address, *_):
        session.end_session()

    def on_recalibrate(unused_address, *_):
        session.recalibrate()

    dispatcher = Dispatcher()
    dispatcher.map(args.in_address, on_biodata)
    dispatcher.map('/crocodile/session/start', on_session_start)
    dispatcher.map('/crocodile/calibration/start', on_calibration_start)
    dispatcher.map('/crocodile/calibration/stop', on_calibration_stop)
    dispatcher.map('/crocodile/live/start', on_live_start)
    dispatcher.map('/crocodile/session/end', on_session_end)
    dispatcher.map('/crocodile/calibration/recalibrate', on_recalibrate)
    server = BlockingOSCUDPServer((args.in_host, args.in_port), dispatcher)

    def handle_shutdown(*_):
        print("\nShutting down.")
        if session.phase != 'IDLE':
            session.end_session()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)

    print(f"Listening for biodata + session control on {args.in_host}:{args.in_port}")
    if osc_client is not None:
        print(f"Sending W vectors to {args.out_host}:{args.out_port}{args.out_address}")
    else:
        print("--log-only: printing W vectors instead of sending OSC")
    print(f"Broadcasting session status to {args.status_out_host}:{args.status_out_port}{args.status_out_address}")
    print("Waiting for /crocodile/session/start ...")
    server.serve_forever()


if __name__ == '__main__':
    main()
