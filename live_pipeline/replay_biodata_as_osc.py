"""Replay a recorded raw biodata CSV as live OSC messages.

Stands in for real sensor hardware -- no confirmed hardware/OSC protocol
exists yet for how biodata will actually arrive live (see PIPELINE.md's
live-readiness section), so this and live_pipeline.py (same folder) define
and test against a protocol of our own: one OSC message per raw sample, 3
floats [heart, gsr, respiration].

Usage:
    python live_pipeline/replay_biodata_as_osc.py --input live_pipeline/data/erin_live_segment.csv
    python live_pipeline/replay_biodata_as_osc.py --input <file> --speed 5  # faster, for quicker testing
    python live_pipeline/replay_biodata_as_osc.py --input <file> --loop        # repeat forever, Ctrl+C to stop
    python live_pipeline/replay_biodata_as_osc.py --input <file> --loop 3      # repeat exactly 3 times

CAUTION on --speed: this sends 100Hz * speed OSC messages/sec over a local
UDP socket, and live_pipeline.py's OSC server (BlockingOSCUDPServer)
processes them synchronously, one at a time. At high speeds (empirically,
50-80x locally), the receiver can't keep up and the OS drops UDP packets --
NOT a live_pipeline.py bug, but it silently produces gappy, corrupted input
that can make the whole downstream pipeline (alignment + regressor) output
wildly implausible W vectors (norms in the hundreds instead of ~10), which
is easy to misdiagnose as a real bug in the pipeline rather than a
replay-speed testing artifact. Confirmed clean up to 5x in testing; use
--speed 1.0 (real-time) for anything where output correctness matters, not
just "does it run."
"""

import argparse
import itertools
import time

import pandas as pd
from pythonosc.udp_client import SimpleUDPClient


def replay_once(df, client, address, sampling_rate, interval, has_labels):
    """Sends one full pass over df at real-time-scaled intervals. Returns elapsed seconds."""
    last_emotion, last_feeling_it = None, None
    start = time.perf_counter()
    for i, row in enumerate(df.itertuples(index=False)):
        client.send_message(address, [float(row.heart), float(row.gsr), float(row.respiration)])

        if has_labels and (row.emotion != last_emotion or row.feeling_it != last_feeling_it):
            t_s = i / sampling_rate
            print(f"  [{t_s:8.1f}s] emotion={row.emotion} feeling_it={row.feeling_it}")
            last_emotion, last_feeling_it = row.emotion, row.feeling_it

        target = start + (i + 1) * interval
        sleep_s = target - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        if (i + 1) % (sampling_rate * 10) == 0:
            print(f"  sent {i + 1:,}/{len(df):,} samples")

    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(
        description='Replay a raw biodata CSV as live OSC messages (stands in for sensor hardware)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', required=True, help='Raw biodata CSV (heart/gsr/respiration columns)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--address', default='/crocodile/biodata')
    parser.add_argument('--sampling-rate', type=int, default=100, help='Hz, matches the CSV\'s own rate')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier (1.0 = real-time)')
    parser.add_argument('--limit', type=int, default=None, help='Only replay the first N samples')
    parser.add_argument('--loop', nargs='?', type=int, const=0, default=None, metavar='N',
                        help='Repeat the replay instead of sending it once -- for an extended demo/show '
                             'run rather than a one-off test. Bare flag loops forever (Ctrl+C to stop); '
                             'give a count (--loop 3) to repeat exactly N times.')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.iloc[:args.limit]
    print(f"Loaded {len(df):,} samples ({len(df) / args.sampling_rate:.1f}s at {args.sampling_rate}Hz) "
          f"from {args.input}")

    has_labels = 'emotion' in df.columns and 'feeling_it' in df.columns
    if has_labels:
        print("Ground-truth 'emotion'/'feeling_it' columns found -- will log on change")

    client = SimpleUDPClient(args.host, args.port)
    interval = 1.0 / args.sampling_rate / args.speed
    print(f"Sending to {args.host}:{args.port}{args.address} at {args.speed}x speed "
          f"({interval * 1000:.2f}ms/sample)")

    passes = itertools.count() if args.loop == 0 else range(args.loop if args.loop is not None else 1)
    n_completed = 0
    try:
        for pass_num in passes:
            if args.loop is not None:
                label = 'forever' if args.loop == 0 else str(args.loop)
                print(f"-- pass {pass_num + 1}/{label} --")
            elapsed = replay_once(df, client, args.address, args.sampling_rate, interval, has_labels)
            n_completed += 1
            print(f"Done: sent {len(df):,} samples in {elapsed:.1f}s")
    except KeyboardInterrupt:
        print(f"\nStopped after {n_completed} pass(es)")


if __name__ == '__main__':
    main()
