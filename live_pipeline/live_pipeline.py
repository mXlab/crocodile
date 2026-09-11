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

W (OSC out to the live latent controller, changed):
  One OSC message per finalized feature row during LIVE, 512 floats, on
  --out-address (default /crocodile/latent/user). This no longer goes
  straight to Autolume -- it goes to the live latent controller (Open
  Stage Control + crocodile-control-module.js, see
  live_pipeline/run_control_panel.sh), which composites it with the
  actress vector and forwards the result on to Autolume itself. This
  script never touches StyleGAN2.

Session control (OSC in, new):
  /crocodile/session/start        [session_id: str] (optional)
  /crocodile/calibration/start
  /crocodile/calibration/set_emotion  [label: str]   (CALIBRATING only)
  /crocodile/calibration/stop
  /crocodile/live/start
  /crocodile/session/end
  /crocodile/calibration/recalibrate   (LIVE only; does not change phase)
  /crocodile/calibration/refit         (CALIBRATED/LIVE; does not change phase)
An invalid transition (wrong current state) is logged and ignored, never
crashes the server -- the operator is a human clicking buttons live.

Live per-visitor alignment fit (new, optional -- needs --reference-features):
  Every biodata sample pushed during CALIBRATING is tagged with the
  session's "current calibration emotion" (default 'neu', changeable via
  calibration/set_emotion) and buffered. At calibration/stop (or on
  calibration/refit), that buffer is used to fit a fresh alignment
  transformer against --reference-features. --live-transformer-method auto
  (the default) picks, in order: ClassConditionalOTTransformer if the
  buffer has >=2 emotion labels with enough samples each
  (--min-samples-per-emotion) -- a guided, multi-emotion calibration;
  CORALTransformer if it has enough total rows regardless of labels
  (--min-samples-for-covariance) -- a single, class-blind but still
  covariance-aware fit; otherwise ZScoreTransformer (needs no labels,
  robust from very little data, but no covariance correction -- can
  produce visually glitchy output; see its docstring in
  biodata_pipeline/modules/alignment_transformer.py and PIPELINE.md's
  "Live per-visitor alignment fit" section). Any of zscore/coral/ot_global
  can be forced explicitly to always pool class-blind regardless of how
  much labeled data is available. Fit failures are logged and never crash
  the server; the previous transformer (the --transformer startup
  fallback, until a live fit first succeeds) stays active. If
  --reference-features is omitted, this is disabled entirely and every
  session just uses --transformer, unchanged from before.

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

No calibration data at all? Pass --transformer with no --reference-features
(as above) and go straight from session/start to live/start -- see README.md's
"Simplest possible test", which uses the pre-fit
live_pipeline/data/synthetic_test_transformer.pkl for exactly this.

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
from modules.alignment_transformer import load_transformer, create_transformer

SIGNAL_COLS = {'eda': 'gsr', 'ppg': 'heart', 'resp': 'respiration'}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Live biodata -> W pipeline server: OSC session control, OSC out to Autolume',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--regressor', required=True, help='Path to regressor.joblib')
    parser.add_argument('--transformer', default=None,
                        help='Path to alignment transformer .pkl -- optional startup fallback a '
                             'session uses only until/unless a live per-visitor fit succeeds (see '
                             '--reference-features). If omitted, a session simply sends no W output '
                             'at all until its own calibration produces a fit (rows are skipped with '
                             'a warning, never crash) -- the right choice when simulating a genuinely '
                             'new visitor with no pre-existing model to fall back to.')
    parser.add_argument('--sampling-rate', type=int, default=100, help='Hz')
    parser.add_argument('--in-host', default='127.0.0.1')
    parser.add_argument('--in-port', type=int, default=9000,
                        help='Port for both biodata and session-control OSC messages')
    parser.add_argument('--in-address', default='/crocodile/biodata',
                        help='OSC address this script listens on for [heart, gsr, respiration]')
    parser.add_argument('--out-host', default='127.0.0.1', help="The live latent controller's host")
    parser.add_argument('--out-port', type=int, default=9001,
                        help="The live latent controller's OSC-in port (same port its status "
                             "broadcasts already use)")
    parser.add_argument('--out-address', default='/crocodile/latent/user',
                        help='OSC address to send the 512-float user W vector to -- the live '
                             'latent controller listens here, composites it with the actress '
                             'vector, and forwards the result on to Autolume')
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
    parser.add_argument('--reference-features', default=None,
                        help='Path to the actress\' online-schema feature CSV with \'emotion\' '
                             'labels -- her own biodata, recorded synchronized with the video '
                             'frames the regressor was trained on (e.g. biodata_pipeline/data/'
                             'processed/continuous_features_online.csv -- NOT erin_features_online'
                             '.csv, Erin is a separate test subject, not the actress). If given, '
                             'each session\'s calibration '
                             'recording is used to fit a fresh alignment transformer against this '
                             'reference at calibration/stop (and on-demand via calibration/refit), '
                             'replacing the --transformer fallback for that session. If omitted, '
                             'this is disabled entirely and every session just uses --transformer.')
    parser.add_argument('--live-transformer-method',
                        choices=['auto', 'zscore', 'coral', 'ot_global', 'ot_classconditional'],
                        default='auto',
                        help='Method for the live per-visitor fit above. auto picks, in order: '
                             'ot_classconditional if the calibration buffer has >=2 emotion labels '
                             'with >=--min-samples-per-emotion rows each (guided multi-emotion '
                             'calibration); else coral if it has >=--min-samples-for-covariance '
                             'rows total, regardless of labels (enough for a well-conditioned '
                             'covariance estimate even class-blind); else zscore (robust from very '
                             'little data, but no covariance/cross-feature correction -- see '
                             'ZScoreTransformer\'s docstring on the resulting glitch risk). Force '
                             'coral/ot_global/zscore explicitly to always pool class-blind '
                             'regardless of how much labeled data is available -- see '
                             'alignment_transformer.py.')
    parser.add_argument('--min-samples-per-emotion', type=int, default=30,
                        help='Minimum rows for an emotion label to count toward auto\'s '
                             'ot_classconditional selection above. A starting value, not rigorously '
                             'derived -- tune once tested against real induced-emotion calibration '
                             'data.')
    parser.add_argument('--min-samples-for-covariance', type=int, default=300,
                        help='Minimum total calibration rows (any/no labels) for auto\'s coral '
                             'selection above -- a rule-of-thumb multiple of the 53-feature count '
                             'for a well-conditioned covariance estimate (roughly 5-6x), not a '
                             'rigorously derived number. Below this, auto falls back to zscore.')
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
        'set_calibration_emotion': {'CALIBRATING'},
        'stop_calibration': {'CALIBRATING'},
        'start_live': {'READY', 'CALIBRATED'},
        'end_session': {'READY', 'CALIBRATING', 'CALIBRATED', 'LIVE'},
        'recalibrate': {'LIVE'},
        'refit_transformer': {'CALIBRATED', 'LIVE'},
    }

    def __init__(self, sampling_rate, record_dir, calibration_df, status_client, status_address,
                 model, scaler, feature_cols, w_cols, static_transformer, reference_df,
                 live_transformer_method, min_samples_per_emotion, min_samples_for_covariance,
                 osc_client, out_address, log_only):
        self.sampling_rate = sampling_rate
        self.record_dir = Path(record_dir) if record_dir else None
        self.calibration_df = calibration_df
        self.status_client = status_client
        self.status_address = status_address

        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.w_cols = w_cols
        self.static_transformer = static_transformer
        self.reference_df = reference_df
        self.live_transformer_method = live_transformer_method
        self.min_samples_per_emotion = min_samples_per_emotion
        self.min_samples_for_covariance = min_samples_for_covariance
        self.osc_client = osc_client
        self.out_address = out_address
        self.log_only = log_only

        self.phase = 'IDLE'
        self.session_id = None
        self.extractor = None
        self.active_transformer = static_transformer
        self.current_calibration_emotion = 'neu'
        self._calibration_rows = []
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
        self.active_transformer = self.static_transformer  # reset for the new visitor
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
        self._calibration_rows = []
        self.current_calibration_emotion = 'neu'
        self.phase = 'CALIBRATING'
        print("Calibration started")
        self._broadcast_status()

    def set_calibration_emotion(self, label):
        if not self._check('set_calibration_emotion'):
            return
        self.current_calibration_emotion = label
        print(f"  Calibration emotion set to '{label}'")

    def stop_calibration(self):
        if not self._check('stop_calibration'):
            return
        self.phase = 'CALIBRATED'
        print("Calibration stopped")
        self._fit_live_transformer()
        self._broadcast_status()

    def refit_transformer(self):
        """Re-fit from the same stored calibration buffer, without re-running
        calibration -- e.g. to retry after a failed fit, or to pick up a
        --live-transformer-method change. Does not change phase or touch
        biodata flow."""
        if not self._check('refit_transformer'):
            return
        self._fit_live_transformer()

    def _fit_live_transformer(self):
        if self.reference_df is None:
            return  # feature disabled (no --reference-features)
        if not self._calibration_rows:
            print("  No calibration data buffered -- keeping current transformer")
            return

        subject_df = pd.DataFrame(self._calibration_rows)
        counts = subject_df['emotion'].value_counts()
        viable_emotions = counts[counts >= self.min_samples_per_emotion].index.tolist()

        method = self.live_transformer_method
        if method == 'auto':
            if len(viable_emotions) >= 2:
                method = 'ot_classconditional'
            elif len(subject_df) >= self.min_samples_for_covariance:
                method = 'coral'
            else:
                method = 'zscore'

        try:
            new_transformer = create_transformer(method)
            if method in ('zscore', 'coral', 'ot_global'):
                # Single dominant label (usually just 'neu' from an un-cued
                # calibration, or any pooled/forced class-blind fit) --
                # match reference rows with that same label specifically;
                # pool the whole reference if multiple/no labels present.
                emotion = subject_df['emotion'].mode().iloc[0] if len(counts) == 1 else None
                new_transformer.fit(self.reference_df, subject_df, emotion=emotion)
            else:
                new_transformer.fit(self.reference_df, subject_df[subject_df['emotion'].isin(viable_emotions)])
            self.active_transformer = new_transformer
            print(f"  Live transformer fit: {method} on {len(subject_df)} calibration rows "
                  f"({dict(counts)})")
            if self.record_dir:
                self.record_dir.mkdir(parents=True, exist_ok=True)
                out_path = self.record_dir / f"{self.session_id}_live_transformer.pkl"
                new_transformer.save(out_path)
        except Exception as e:
            print(f"  WARNING: live transformer fit failed ({e}) -- keeping current transformer")

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

        row_dict = {'heart': heart, 'gsr': gsr, 'respiration': respiration}
        if self.phase == 'CALIBRATING':
            row_dict['emotion'] = self.current_calibration_emotion
        row_df = pd.DataFrame([row_dict])
        rows = self.extractor.push(row_df, signal_cols=SIGNAL_COLS, feature_interval_s=1.0)

        if self.phase == 'CALIBRATING':
            self._calibration_rows.extend(rows)

        if self.phase != 'LIVE':
            return  # CALIBRATING/CALIBRATED: keep priming state, discard finalized rows here

        for row in rows:
            transformer = self.active_transformer
            if transformer is None:
                self.n_rows_skipped += 1
                print(f"  Skipping row {self.n_rows_sent + self.n_rows_skipped}: no transformer yet "
                      f"(no --transformer fallback and no live fit has succeeded for this session)")
                continue
            aligned = transformer.transform(pd.DataFrame([row])[transformer.feature_cols])
            aligned_df = pd.DataFrame(aligned, columns=transformer.feature_cols)
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

    static_transformer = None
    if args.transformer:
        print(f"Loading alignment transformer from {args.transformer}")
        static_transformer = load_transformer(args.transformer)
        print(f"  {static_transformer.__class__.__name__}, emotions: {static_transformer.common_emotions}")

        missing = [c for c in feature_cols if c not in static_transformer.feature_cols]
        if missing:
            raise ValueError(f"Transformer is missing regressor's expected features: {missing}")
    else:
        print("--transformer not given: sessions send no W output until their own "
              "calibration produces a live fit (see --reference-features)")

    calibration_df = None
    if args.calibration_csv:
        print(f"Loading calibration CSV from {args.calibration_csv}")
        calibration_df = pd.read_csv(args.calibration_csv)

    reference_df = None
    if args.reference_features:
        print(f"Loading reference features from {args.reference_features} "
              f"(live per-visitor transformer fitting enabled, method={args.live_transformer_method})")
        reference_df = pd.read_csv(args.reference_features)
    else:
        print("--reference-features not given: live per-visitor transformer fitting disabled, "
              "every session uses --transformer as-is" if static_transformer is not None else
              "--reference-features not given: live per-visitor transformer fitting disabled")

    if static_transformer is None and reference_df is None:
        print("WARNING: neither --transformer nor --reference-features given -- no session will "
              "ever produce W output (every LIVE row will be skipped)")

    osc_client = None if args.log_only else SimpleUDPClient(args.out_host, args.out_port)
    status_client = SimpleUDPClient(args.status_out_host, args.status_out_port)

    session = SessionState(
        sampling_rate=args.sampling_rate, record_dir=args.record_dir, calibration_df=calibration_df,
        status_client=status_client, status_address=args.status_out_address,
        model=model, scaler=scaler, feature_cols=feature_cols, w_cols=w_cols,
        static_transformer=static_transformer, reference_df=reference_df,
        live_transformer_method=args.live_transformer_method,
        min_samples_per_emotion=args.min_samples_per_emotion,
        min_samples_for_covariance=args.min_samples_for_covariance,
        osc_client=osc_client, out_address=args.out_address, log_only=args.log_only)

    def on_biodata(unused_address, *osc_args):
        if len(osc_args) != 3:
            print(f"  WARNING: expected 3 args [heart, gsr, respiration], got {len(osc_args)} -- dropping")
            return
        session.handle_biodata(*osc_args)

    def on_session_start(unused_address, *osc_args):
        session.start_session(osc_args[0] if osc_args else None)

    def on_calibration_start(unused_address, *_):
        session.start_calibration()

    def on_calibration_set_emotion(unused_address, *osc_args):
        if not osc_args:
            print("  WARNING: calibration/set_emotion needs a label argument -- dropping")
            return
        session.set_calibration_emotion(osc_args[0])

    def on_calibration_stop(unused_address, *_):
        session.stop_calibration()

    def on_live_start(unused_address, *_):
        session.start_live()

    def on_session_end(unused_address, *_):
        session.end_session()

    def on_recalibrate(unused_address, *_):
        session.recalibrate()

    def on_refit_transformer(unused_address, *_):
        session.refit_transformer()

    dispatcher = Dispatcher()
    dispatcher.map(args.in_address, on_biodata)
    dispatcher.map('/crocodile/session/start', on_session_start)
    dispatcher.map('/crocodile/calibration/start', on_calibration_start)
    dispatcher.map('/crocodile/calibration/set_emotion', on_calibration_set_emotion)
    dispatcher.map('/crocodile/calibration/stop', on_calibration_stop)
    dispatcher.map('/crocodile/live/start', on_live_start)
    dispatcher.map('/crocodile/session/end', on_session_end)
    dispatcher.map('/crocodile/calibration/recalibrate', on_recalibrate)
    dispatcher.map('/crocodile/calibration/refit', on_refit_transformer)
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
