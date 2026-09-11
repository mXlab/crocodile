#!/usr/bin/env bash
# Launches the Open Stage Control panel (crocodile-control-panel.json) wired
# to live_pipeline.py's default ports: sends session-control OSC to 9000
# (live_pipeline.py's --in-port) and listens for status broadcasts on 9001
# (live_pipeline.py's --status-out-port). Loads the crocodile-control-module.js
# custom module which handles /crocodile/latent/* addresses for real-time
# latent vector control. HTTP UI served on 8090, not open-stage-control's own
# default of 8080, since that's commonly already taken by something else.
# --theme loads emotion_grid/data/theme.css, which each grid_cell widget's
# `class: grid-cell-<id>;` picks up its background-image from -- this is
# the only place the (private, per-machine) thumbnail location needs
# resolving; theme.css's own image paths are relative to itself, so the
# panel JSON never needs machine-specific absolute paths. Regenerate
# theme.css with emotion_grid/build_grid.py, not by hand.
# Override with extra args, e.g.:
#   live_pipeline/run_control_panel.sh -p 8091 -s 127.0.0.1:9100 -o 9101
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec open-stage-control \
    --load "$REPO_ROOT/live_pipeline/crocodile-control-panel.json" \
    --theme "$REPO_ROOT/emotion_grid/data/theme.css" \
    --port 8090 \
    --send 127.0.0.1:9000 \
    --osc-port 9001 \
    --custom-module "$REPO_ROOT/live_pipeline/crocodile-control-module.js" \
    "$@"
