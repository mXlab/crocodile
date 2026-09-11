#!/usr/bin/env bash
# Runs prepare_synthetic_test_fixtures.py with biodata_pipeline/venv's interpreter.
# One-time offline prep; the resulting two files are already committed, so
# there's normally no need to run this again -- see the script's docstring.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/biodata_pipeline/venv/bin/python3" "$REPO_ROOT/live_pipeline/prepare_synthetic_test_fixtures.py" "$@"
