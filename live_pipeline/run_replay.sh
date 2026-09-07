#!/usr/bin/env bash
# Runs replay_biodata_as_osc.py with biodata_pipeline/venv's interpreter.
# Any extra args are passed straight through, e.g.:
#   live_pipeline/run_replay.sh --input live_pipeline/data/erin_live_segment.csv --speed 1.0
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/biodata_pipeline/venv/bin/python3" "$REPO_ROOT/live_pipeline/replay_biodata_as_osc.py" "$@"
