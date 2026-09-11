"""Debug viewer: listens for the same W-over-OSC stream sent to Autolume by
live_pipeline.py (same folder), renders each vector via this project's own
StyleGAN2 loading code, and shows a live-updating preview window -- for
visually sanity-checking the live pipeline without needing Autolume running
or configured.

Thin CLI wrapper around latent_viewer_core.LatentOscViewer (same folder),
shared with live_viewer.py so both scripts render identically.

Runs in latent_pipeline/.venv (needs torch/StyleGAN2) -- separate from
live_pipeline.py, which runs in biodata_pipeline/venv and never touches
StyleGAN2 at all.

Usage (from the repo root):
    python live_pipeline/latent_osc_debug_viewer.py --config latent_pipeline/configs/default.yaml
    python live_pipeline/latent_osc_debug_viewer.py --in-port 1338 --in-address /crocodile/latent/final
"""

import argparse

from latent_viewer_core import LatentOscViewer


def main():
    parser = argparse.ArgumentParser(
        description='Debug viewer: render the live W-over-OSC stream locally',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default='latent_pipeline/configs/default.yaml')
    parser.add_argument('--in-host', default='0.0.0.0',
                        help='Bind address. 127.0.0.1 only accepts packets sent from this machine -- '
                             'use 0.0.0.0 (default) to receive from a remote sender like TouchDesigner')
    parser.add_argument('--in-port', type=int, default=1338,
                        help="Same port live_pipeline.py's --out-port sends to")
    parser.add_argument('--in-address', default='/crocodile/latent/final')
    parser.add_argument('--preview-resolution', type=int, default=512,
                        help='Downsample StyleGAN2 output to this size for faster live rendering')
    parser.add_argument('--fps', type=float, default=10.0, help='Render/display cadence, independent of OSC rate')
    args = parser.parse_args()

    viewer = LatentOscViewer(
        config_path=args.config,
        in_host=args.in_host,
        in_port=args.in_port,
        in_address=args.in_address,
        render_resolution=args.preview_resolution,
        fps=args.fps,
        window_name="Crocodile live preview (debug)",
        show_overlay=True,
    )
    viewer.run(fullscreen=False)


if __name__ == '__main__':
    main()
