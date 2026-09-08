#!/usr/bin/env bash
# Runs generate_synthetic_biodata.py with biodata_pipeline/venv's interpreter.
# Any extra args are passed straight through, e.g.:
#   live_pipeline/run_generate_synthetic.sh --duration 60 --seed 1 \
#       --output live_pipeline/data/synthetic_calibration.csv
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/biodata_pipeline/venv/bin/python3" "$REPO_ROOT/live_pipeline/generate_synthetic_biodata.py" "$@"
