"""Debug viewer: listens for the same W-over-OSC stream sent to Autolume by
live_pipeline.py (same folder), renders each vector via this project's own
StyleGAN2 loading code, and shows a live-updating preview window -- for
visually sanity-checking the live pipeline without needing Autolume running
or configured.

Runs in latent_pipeline/.venv (needs torch/StyleGAN2) -- separate from
live_pipeline.py, which runs in biodata_pipeline/venv and never touches
StyleGAN2 at all.

OSC messages arrive on a background thread (python-osc's threading server);
the main thread renders at a fixed, throttled cadence independent of
message arrival rate, always using only the MOST RECENT W received --
messages that arrive faster than rendering keeps up are simply superseded,
never queued, so the preview never falls behind.

Usage (from the repo root):
    python live_pipeline/latent_osc_debug_viewer.py --config latent_pipeline/configs/default.yaml
    python live_pipeline/latent_osc_debug_viewer.py --in-port 1338 --in-address /crocodile/latent/final
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

REPO_ROOT = Path(__file__).resolve().parent.parent
LATENT_PIPELINE_DIR = REPO_ROOT / 'latent_pipeline'
sys.path.insert(0, str(LATENT_PIPELINE_DIR))

from models.stylegan import load_stylegan, generate


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


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

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading StyleGAN2...")
    G = load_stylegan(config, device)

    latest_w = None
    latest_w_lock = threading.Lock()
    n_received = 0

    def on_w(unused_address, *osc_args):
        print("Received message")
        nonlocal latest_w, n_received
        if len(osc_args) != G.w_dim:
            print(f"  WARNING: expected {G.w_dim} floats, got {len(osc_args)} -- dropping")
            return
        with latest_w_lock:
            latest_w = np.array(osc_args, dtype=np.float32)
            print(latest_w)
            n_received += 1

    dispatcher = Dispatcher()
    dispatcher.map(args.in_address, on_w)
    server = ThreadingOSCUDPServer((args.in_host, args.in_port), dispatcher)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Listening for W vectors on {args.in_host}:{args.in_port}{args.in_address}")
    print("Press 'q' in the preview window (or Ctrl+C) to quit")

    window_name = "Crocodile live preview (debug)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    frame_interval = 1.0 / args.fps
    last_rendered_w = None

    try:
        while True:
            start = time.perf_counter()
            with latest_w_lock:
                w_to_render = latest_w

            if w_to_render is not None and (last_rendered_w is None
                                             or not np.array_equal(w_to_render, last_rendered_w)):
                with torch.no_grad():
                    w_tensor = torch.tensor(w_to_render, dtype=torch.float32, device=device).unsqueeze(0)
                    img = generate(G, w_tensor)
                    img = F.interpolate(img, size=args.preview_resolution, mode='bilinear', align_corners=False)
                arr = ((img[0].cpu().clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).numpy().astype(np.uint8)
                arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                cv2.putText(arr_bgr, f"received: {n_received}", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(window_name, arr_bgr)
                last_rendered_w = w_to_render

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            elapsed = time.perf_counter() - start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nShutting down. W vectors received: {n_received}")
        server.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
