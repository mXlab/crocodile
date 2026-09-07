#!/usr/bin/env bash
# Runs live_pipeline.py with biodata_pipeline/venv's interpreter, from
# whatever directory this script itself lives in -- so it works regardless
# of the caller's cwd. Any extra args are passed straight through, e.g.:
#   live_pipeline/run_live.sh \
#       --regressor latent_pipeline/outputs/stage5_regressor_online/regressor.joblib \
#       --transformer biodata_pipeline/models/transformer_ot_classconditional_online.pkl \
#       --calibration-csv live_pipeline/data/erin_calibration_segment.csv
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/biodata_pipeline/venv/bin/python3" "$REPO_ROOT/live_pipeline/live_pipeline.py" "$@"
