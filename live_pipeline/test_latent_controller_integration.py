"""
Test script for the live latent controller: launches the controller headless,
a debug receiver standing in for Autolume, feeds it fake OSC input, and
checks the composited output behaves as designed.

Run from the repo root with biodata_pipeline/venv active (has python-osc):
    source biodata_pipeline/venv/bin/activate
    python3 live_pipeline/test_latent_controller_integration.py
"""
import csv
import subprocess
import threading
import time

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

CONTROLLER_PORT = 9001
AUTOLUME_PORT = 1338
W_DIM = 512

print("="*80)
print("LIVE LATENT CONTROLLER INTEGRATION TEST")
print("="*80)

# --- fake Autolume: just record every /crocodile/latent/final we receive ---
received = []

def on_final(unused_address, *args):
    received.append(list(args))

dispatcher = Dispatcher()
dispatcher.map('/crocodile/latent/final', on_final)
fake_autolume = BlockingOSCUDPServer(('127.0.0.1', AUTOLUME_PORT), dispatcher)
fake_autolume_thread = threading.Thread(target=fake_autolume.serve_forever, daemon=True)
fake_autolume_thread.start()

# --- launch the controller headless ---
controller = subprocess.Popen([
    'open-stage-control', '--no-gui',
    '--load', 'live_pipeline/crocodile-control-panel.json',
    '--port', '8092',  # avoid colliding with a real instance on 8090
    '--osc-port', str(CONTROLLER_PORT),
    '--custom-module', 'live_pipeline/crocodile-control-module.js',
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

time.sleep(3)  # let the server + module init() (manifest load) finish

try:
    client = SimpleUDPClient('127.0.0.1', CONTROLLER_PORT)

    # 1. Before any /grid/select, the module has no target yet and tick()
    #    should send nothing at all.
    time.sleep(0.5)
    received.clear()
    time.sleep(0.5)
    assert len(received) == 0, f"expected no output before selecting a target, got {len(received)}"
    print("✓ no output before any target is selected and transition started")

    # 2. Select a real emotion id from the manifest and start the transition.
    with open('emotion_grid/data/manifest.csv') as f:
        first_row = next(csv.DictReader(f))
    target_id = first_row['id']

    client.send_message('/crocodile/grid/select', target_id)
    client.send_message('/crocodile/transition/time', 0.01)  # jump immediately for a fast test
    client.send_message('/crocodile/transition/running', 1)

    time.sleep(0.5)
    assert len(received) > 0, "expected output after selecting a target and starting the transition"
    last = received[-1]
    assert len(last) == W_DIM, f"expected {W_DIM} floats, got {len(last)}"
    print(f"✓ received {len(received)} /crocodile/latent/final messages, each with {W_DIM} floats")

    # 3. Feed a fake user vector and set mix to 0 (pure user) -- output should
    #    converge toward it. Truncation disabled here since it deliberately pulls
    #    the output toward w_avg -- this step is isolating the mix math, not it.
    client.send_message('/crocodile/output/truncation', 1.0)
    fake_w_u = [0.0] * W_DIM
    fake_w_u[0] = 42.0
    client.send_message('/crocodile/latent/user', fake_w_u)
    client.send_message('/crocodile/mix/amount', 0.0)
    time.sleep(0.5)
    last = received[-1]
    assert abs(last[0] - 42.0) < 5.0, f"expected output to converge toward the user vector (42.0), got {last[0]}"
    print(f"✓ mix=0 output converges toward the incoming user vector (got {last[0]:.2f}, expected near 42.0)")

    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80)

finally:
    controller.terminate()
    try:
        stdout, _ = controller.communicate(timeout=5)
        if stdout:
            print("\n--- controller subprocess output ---")
            print(stdout)
    except subprocess.TimeoutExpired:
        controller.kill()
        controller.wait()
    fake_autolume.shutdown()
