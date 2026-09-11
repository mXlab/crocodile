#!/usr/bin/env bash
# Runs live_viewer.py with latent_pipeline/.venv's interpreter (needs
# torch/StyleGAN2 -- same requirement as run_debug_viewer.sh).
# Any extra args are passed straight through, e.g.:
#   live_pipeline/run_live_viewer.sh --fullscreen
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/latent_pipeline/.venv/bin/python3" "$REPO_ROOT/live_pipeline/live_viewer.py" "$@"
