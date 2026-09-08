#!/usr/bin/env bash
# Runs session_control.py with biodata_pipeline/venv's interpreter.
# Any extra args are passed straight through, e.g.:
#   live_pipeline/run_session_control.sh --start-session
#   live_pipeline/run_session_control.sh --start-calibration
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/biodata_pipeline/venv/bin/python3" "$REPO_ROOT/live_pipeline/session_control.py" "$@"
