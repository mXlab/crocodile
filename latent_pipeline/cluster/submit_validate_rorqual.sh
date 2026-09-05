#!/bin/bash
#SBATCH --job-name=crocodile-validate
#SBATCH --account=def-sofian
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Stage 3: Validate encoder quality (CNN vs optimization-based inversion).
# Much lighter than training — 30 frames x 1000 optimization steps, minutes
# on an H100 vs ~60-90min on a laptop GPU.
#
# IMPORTANT: pass --checkpoint explicitly. The training job (submit_train_rorqual.sh)
# writes to latent_pipeline/outputs/best.pt and latest.pt on its own schedule —
# if it's still running, do NOT validate against the bare default path, since
# it may be mid-write or not the checkpoint you intend to check. Upload the
# checkpoint you want validated under a distinct name first, e.g.:
#
#   rsync -avz --progress latent_pipeline/outputs/best.pt \
#       sofian@rorqual.alliancecan.ca:~/links/projects/def-sofian/sofian/crocodile/latent_pipeline/outputs/best_local.pt
#
# Usage:
#   cd ~/links/projects/def-sofian/sofian/crocodile
#   sbatch latent_pipeline/cluster/submit_validate_rorqual.sh --checkpoint latent_pipeline/outputs/best_local.pt

set -euo pipefail

# Load modules and activate env — same requirements as training (opencv/cuda
# must load before venv activation, see setup_rorqual.sh for why).
module load python/3.11 scipy-stack/2024a gcc opencv/4.14.0 cuda/12.6
source "$HOME/envs/crocodile/bin/activate"

# Move to repo root
cd "$SLURM_SUBMIT_DIR"

echo "=== Job info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Dir:    $(pwd)"
echo ""

CONFIG="latent_pipeline/configs/rorqual.yaml"

echo "Starting stage3_validate with config: $CONFIG"
python latent_pipeline/scripts/stage3_validate.py \
    --config "$CONFIG" \
    "$@"

echo ""
echo "=== Validation complete ==="
echo "Outputs in: latent_pipeline/outputs/"
