#!/usr/bin/env bash
# Symlinks every private/gitignored asset (StyleGAN2 checkpoint, regressors,
# alignment transformers, real biodata, emotion grid data) from a single
# consolidated crocodile-private/ folder into the exact repo-relative paths
# the code already expects -- no config/CLI changes needed either way.
#
# Usage:
#   scripts/link_private_data.sh [path-to-crocodile-private]
#
# Defaults to a sibling directory next to this checkout
# (../crocodile-private, i.e. workspace/crocodile-private if this repo is at
# workspace/crocodile) if no path is given. See INSTALL.md for how to obtain
# that folder from a teammate.
#
# Safe to re-run: skips anything already correctly linked, and never
# overwrites a real (non-symlink) file or directory that happens to already
# exist at a target path -- if you have local private data you built
# yourself sitting at one of these paths, move it aside first.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIVATE_ROOT="${1:-$REPO_ROOT/../crocodile-private}"
PRIVATE_ROOT="$(cd "$PRIVATE_ROOT" 2>/dev/null && pwd || true)"

if [ -z "$PRIVATE_ROOT" ]; then
    echo "Error: private data folder not found at '${1:-$REPO_ROOT/../crocodile-private}'" >&2
    echo "Usage: $0 [path-to-crocodile-private]" >&2
    exit 1
fi

echo "Linking private data from: $PRIVATE_ROOT"
echo "Into repo:                 $REPO_ROOT"
echo

# repo-relative-path : private-relative-path
MAPPINGS=(
    "models/finalModel_Crocodile.pkl:models/finalModel_Crocodile.pkl"
    "latent_pipeline/outputs/stage5_regressor:latent_pipeline/outputs/stage5_regressor"
    "latent_pipeline/outputs/stage5_regressor_continuous:latent_pipeline/outputs/stage5_regressor_continuous"
    "latent_pipeline/outputs/stage5_regressor_online:latent_pipeline/outputs/stage5_regressor_online"
    "biodata_pipeline/models:biodata_pipeline/models"
    "biodata_pipeline/data/raw:biodata_pipeline/data/raw"
    "biodata_pipeline/data/processed/continuous_features_online.csv:biodata_pipeline/data/processed/continuous_features_online.csv"
    "emotion_grid/data:emotion_grid/data"
    "luana-Crocodile-with-data:luana-Crocodile-with-data"
    "live_pipeline/data/erin_calibration_anx.csv:live_pipeline/data/erin_calibration_anx.csv"
    "live_pipeline/data/erin_calibration_long_neu.csv:live_pipeline/data/erin_calibration_long_neu.csv"
    "live_pipeline/data/erin_calibration_sad.csv:live_pipeline/data/erin_calibration_sad.csv"
    "live_pipeline/data/erin_calibration_segment.csv:live_pipeline/data/erin_calibration_segment.csv"
    "live_pipeline/data/erin_live_after_long_neu.csv:live_pipeline/data/erin_live_after_long_neu.csv"
    "live_pipeline/data/erin_live_scenario3.csv:live_pipeline/data/erin_live_scenario3.csv"
    "live_pipeline/data/erin_live_segment.csv:live_pipeline/data/erin_live_segment.csv"
    "live_pipeline/data/sessions:live_pipeline/data/sessions"
)

n_linked=0
n_already=0
n_skipped_missing=0
n_skipped_conflict=0

for mapping in "${MAPPINGS[@]}"; do
    repo_rel="${mapping%%:*}"
    priv_rel="${mapping##*:}"
    src="$PRIVATE_ROOT/$priv_rel"
    dst="$REPO_ROOT/$repo_rel"

    if [ ! -e "$src" ]; then
        echo "  SKIP (not in private folder): $repo_rel"
        n_skipped_missing=$((n_skipped_missing + 1))
        continue
    fi

    if [ -L "$dst" ]; then
        if [ "$(readlink "$dst")" = "$src" ]; then
            n_already=$((n_already + 1))
            continue
        fi
        rm "$dst"
    elif [ -e "$dst" ]; then
        echo "  CONFLICT (real file/dir already there, not touching): $repo_rel"
        n_skipped_conflict=$((n_skipped_conflict + 1))
        continue
    fi

    mkdir -p "$(dirname "$dst")"
    ln -s "$src" "$dst"
    echo "  Linked: $repo_rel"
    n_linked=$((n_linked + 1))
done

echo
echo "Done. Linked: $n_linked, already linked: $n_already, missing from private folder: $n_skipped_missing, conflicts: $n_skipped_conflict"
if [ "$n_skipped_conflict" -gt 0 ]; then
    echo "Conflicts need manual attention -- move the existing file/dir aside, then re-run."
fi
