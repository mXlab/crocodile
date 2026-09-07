#!/usr/bin/env bash
# Launches the Open Stage Control panel (crocodile-control-panel.json) wired
# to live_pipeline.py's default ports: sends session-control OSC to 9000
# (live_pipeline.py's --in-port) and listens for status broadcasts on 9001
# (live_pipeline.py's --status-out-port). HTTP UI served on 8090, not
# open-stage-control's own default of 8080, since that's commonly already
# taken by something else. Override with extra args, e.g.:
#   live_pipeline/run_control_panel.sh -p 8091 -s 127.0.0.1:9100 -o 9101
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec open-stage-control \
    --load "$REPO_ROOT/live_pipeline/crocodile-control-panel.json" \
    --port 8090 \
    --send 127.0.0.1:9000 \
    --osc-port 9001 \
    "$@"
