#!/usr/bin/env bash
# Runs latent_osc_debug_viewer.py with latent_pipeline/.venv's interpreter (needs
# torch/StyleGAN2 -- the only one of the three live scripts that does).
# Any extra args are passed straight through, e.g.:
#   live_pipeline/run_debug_viewer.sh --config latent_pipeline/configs/default.yaml
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/latent_pipeline/.venv/bin/python3" "$REPO_ROOT/live_pipeline/latent_osc_debug_viewer.py" "$@"
