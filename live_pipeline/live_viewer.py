"""Live viewer: listens for the same W-over-OSC stream normally sent to
Autolume (via crocodile-control-module.js, the latent controller), renders
each vector via this project's own StyleGAN2 loading code, and displays it --
a drop-in replacement for Autolume when a full VJ-style performance app isn't
needed, just a straightforward live display, optionally fullscreen.

Defaults match Autolume's own port/address (1338, /crocodile/latent/final),
so pointing the latent controller at this script instead of Autolume needs no
config changes anywhere else in the pipeline.

Thin CLI wrapper around latent_viewer_core.LatentOscViewer (same folder),
shared with latent_osc_debug_viewer.py so both scripts render identically.

NDI output isn't implemented yet. LatentOscViewer.run()'s on_frame callback
is the hook point for adding it later -- it's called with each newly
rendered frame right before display.

Runs in latent_pipeline/.venv (needs torch/StyleGAN2) -- separate from
live_pipeline.py, which runs in biodata_pipeline/venv and never touches
StyleGAN2 at all.

Usage (from the repo root):
    python live_pipeline/live_viewer.py --config latent_pipeline/configs/default.yaml
    python live_pipeline/live_viewer.py --fullscreen
"""

import argparse

from latent_viewer_core import LatentOscViewer


def main():
    parser = argparse.ArgumentParser(
        description='Render the live W-over-OSC stream -- a drop-in replacement for Autolume',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default='latent_pipeline/configs/default.yaml')
    parser.add_argument('--in-host', default='0.0.0.0',
                        help='Bind address. 127.0.0.1 only accepts packets sent from this machine -- '
                             'use 0.0.0.0 (default) to receive from a remote sender like TouchDesigner')
    parser.add_argument('--in-port', type=int, default=1338,
                        help="Autolume's own default OSC-input port -- same one the latent controller "
                             'already sends to')
    parser.add_argument('--in-address', default='/crocodile/latent/final')
    parser.add_argument('--resolution', type=int, default=1024,
                        help='StyleGAN2 output is resized to this size before display')
    parser.add_argument('--fps', type=float, default=30.0, help='Render/display cadence, independent of OSC rate')
    parser.add_argument('--fullscreen', action='store_true',
                        help='Start in fullscreen mode (toggle anytime with the "f" key)')
    parser.add_argument('--overlay', action='store_true',
                        help='Show a received-count debug overlay on the output (off by default -- '
                             'this is meant to be the actual show display)')
    args = parser.parse_args()

    viewer = LatentOscViewer(
        config_path=args.config,
        in_host=args.in_host,
        in_port=args.in_port,
        in_address=args.in_address,
        render_resolution=args.resolution,
        fps=args.fps,
        window_name='Crocodile live viewer',
        show_overlay=args.overlay,
    )
    viewer.run(fullscreen=args.fullscreen)


if __name__ == '__main__':
    main()
