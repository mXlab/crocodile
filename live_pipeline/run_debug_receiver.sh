#!/usr/bin/env bash
# Runs latent_osc_debug_receiver.py with biodata_pipeline/venv's interpreter --
# no torch/StyleGAN2 needed (unlike run_debug_viewer.sh), so this works even
# before the StyleGAN2 checkpoint is available. See INSTALL.md.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/biodata_pipeline/venv/bin/python3" "$REPO_ROOT/live_pipeline/latent_osc_debug_receiver.py" "$@"
