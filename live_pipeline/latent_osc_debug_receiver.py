"""Lightweight debug receiver: listens for the W-over-OSC stream sent by
live_pipeline.py and reports receipt statistics (count, rate, vector norm)
without rendering anything.

Unlike latent_osc_debug_viewer.py (same folder), this needs no StyleGAN2 model
or torch -- just python-osc and numpy, both already in
biodata_pipeline/venv -- so it works even before the (privacy-restricted,
not shared on GitHub) StyleGAN2 checkpoint is available. Use this as the
first smoke test that W vectors are actually arriving with the right shape
and plausible magnitude; use latent_osc_debug_viewer.py afterward for an actual
visual check once the StyleGAN2 model is in place. See INSTALL.md.

Runs under biodata_pipeline/venv.

Usage (from the repo root):
    python live_pipeline/latent_osc_debug_receiver.py
    python live_pipeline/latent_osc_debug_receiver.py --in-port 1338 --expected-dim 512
"""

import argparse
import time

import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


def main():
    parser = argparse.ArgumentParser(
        description="Listen for the live pipeline's W-over-OSC stream and report receipt stats "
                    '(no StyleGAN2/torch required)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in-host', default='0.0.0.0',
                        help='Bind address. 127.0.0.1 only accepts packets sent from this machine -- '
                             'use 0.0.0.0 (default) to receive from a remote sender like TouchDesigner')
    parser.add_argument('--in-port', type=int, default=1338,
                        help="Same port live_pipeline.py's --out-port sends to")
    parser.add_argument('--in-address', default='/crocodile/latent/final')
    parser.add_argument('--expected-dim', type=int, default=512, help='Expected W vector length')
    parser.add_argument('--report-every', type=int, default=10, help='Print a summary every N received messages')
    args = parser.parse_args()

    stats = {'n_received': 0, 'n_bad_dim': 0, 'start': time.perf_counter(), 'last_report': time.perf_counter()}

    def on_w(unused_address, *osc_args):
        stats['n_received'] += 1
        if len(osc_args) != args.expected_dim:
            stats['n_bad_dim'] += 1
            print(f"  WARNING: expected {args.expected_dim} floats, got {len(osc_args)}")
            return
        if stats['n_received'] % args.report_every == 0:
            w = np.array(osc_args, dtype=np.float32)
            now = time.perf_counter()
            rate = args.report_every / (now - stats['last_report']) if now > stats['last_report'] else 0.0
            stats['last_report'] = now
            print(f"  received={stats['n_received']:,} rate={rate:.1f}/s "
                  f"norm={np.linalg.norm(w):.2f} first5={np.round(w[:5], 2).tolist()}")

    dispatcher = Dispatcher()
    dispatcher.map(args.in_address, on_w)
    server = BlockingOSCUDPServer((args.in_host, args.in_port), dispatcher)
    print(f"Listening for W vectors on {args.in_host}:{args.in_port}{args.in_address} "
          f"(expecting {args.expected_dim}-float vectors)")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.perf_counter() - stats['start']
        print(f"\nStopped after {elapsed:.1f}s. Received: {stats['n_received']:,} "
              f"({stats['n_bad_dim']} with unexpected dimension)")


if __name__ == '__main__':
    main()
