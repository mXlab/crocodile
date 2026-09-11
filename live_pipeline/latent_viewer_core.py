"""Shared W-over-OSC listen + StyleGAN2 render loop, used by both
latent_osc_debug_viewer.py (quick sanity check) and live_viewer.py (the
"production" display, meant to stand in for Autolume) -- kept in one place
so their rendering behavior doesn't drift apart.

Runs in latent_pipeline/.venv (needs torch/StyleGAN2).
"""

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


class LatentOscViewer:
    """Listens for W vectors over OSC and renders each one via StyleGAN2 in a
    live-updating OpenCV window.

    OSC messages arrive on a background thread (python-osc's threading
    server); the main thread renders at a fixed, throttled cadence
    independent of message arrival rate, always using only the MOST RECENT W
    received -- messages that arrive faster than rendering keeps up are
    simply superseded, never queued, so the display never falls behind.
    """

    def __init__(self, config_path, in_host, in_port, in_address,
                 render_resolution, fps, window_name, show_overlay=False):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print("Loading StyleGAN2...")
        self.G = load_stylegan(self.config, self.device)

        self.in_host = in_host
        self.in_port = in_port
        self.in_address = in_address
        self.render_resolution = render_resolution
        self.fps = fps
        self.window_name = window_name
        self.show_overlay = show_overlay

        self._latest_w = None
        self._latest_w_lock = threading.Lock()
        self.n_received = 0
        self._server = None

    def _on_w(self, unused_address, *osc_args):
        if len(osc_args) != self.G.w_dim:
            print(f"  WARNING: expected {self.G.w_dim} floats, got {len(osc_args)} -- dropping")
            return
        with self._latest_w_lock:
            self._latest_w = np.array(osc_args, dtype=np.float32)
            self.n_received += 1

    def _render(self, w):
        with torch.no_grad():
            w_tensor = torch.tensor(w, dtype=torch.float32, device=self.device).unsqueeze(0)
            img = generate(self.G, w_tensor)
            img = F.interpolate(img, size=self.render_resolution, mode='bilinear', align_corners=False)
        arr = ((img[0].cpu().clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if self.show_overlay:
            cv2.putText(arr_bgr, f"received: {self.n_received}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return arr_bgr

    def _set_fullscreen(self, enabled):
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL)

    def run(self, fullscreen=False, on_frame=None):
        """Runs the listen/render/display loop until 'q'/Esc is pressed or Ctrl+C.

        'f' toggles fullscreen at runtime regardless of the starting mode.
        on_frame, if given, is called with each newly rendered BGR frame
        right before it's displayed -- the hook point for a future NDI
        sender.
        """
        dispatcher = Dispatcher()
        dispatcher.map(self.in_address, self._on_w)
        self._server = ThreadingOSCUDPServer((self.in_host, self.in_port), dispatcher)
        server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        server_thread.start()
        print(f"Listening for W vectors on {self.in_host}:{self.in_port}{self.in_address}")
        print("Press 'q' or Esc to quit, 'f' to toggle fullscreen (or Ctrl+C to quit)")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        is_fullscreen = fullscreen
        if is_fullscreen:
            self._set_fullscreen(True)

        frame_interval = 1.0 / self.fps
        last_rendered_w = None

        try:
            while True:
                start = time.perf_counter()
                with self._latest_w_lock:
                    w_to_render = self._latest_w

                if w_to_render is not None and (last_rendered_w is None
                                                 or not np.array_equal(w_to_render, last_rendered_w)):
                    frame = self._render(w_to_render)
                    if on_frame is not None:
                        on_frame(frame)
                    cv2.imshow(self.window_name, frame)
                    last_rendered_w = w_to_render

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 27 == Escape
                    break
                elif key == ord('f'):
                    is_fullscreen = not is_fullscreen
                    self._set_fullscreen(is_fullscreen)

                elapsed = time.perf_counter() - start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        except KeyboardInterrupt:
            pass
        finally:
            print(f"\nShutting down. W vectors received: {self.n_received}")
            self._server.shutdown()
            cv2.destroyAllWindows()
