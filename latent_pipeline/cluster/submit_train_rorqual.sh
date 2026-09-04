#!/bin/bash
#SBATCH --job-name=crocodile-encoder
#SBATCH --account=def-sofian
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Phase 2B: Fine-tune EmotionEncoder on real frames (Rorqual, H100 80GB).
#
# --time is set for finishing the ~9 remaining epochs of the current
# 20-epoch schedule (resuming from epoch 10/11) — raise it if you're
# starting a longer run instead. Rorqual's cap is 168:00:00 (7 days).
#
# Usage:
#   cd ~/links/projects/def-sofian/crocodile
#
#   # Continue the interrupted laptop run (uploaded latest.pt, epoch 10):
#   sbatch latent_pipeline/cluster/submit_train_rorqual.sh --resume latent_pipeline/outputs/latest.pt
#
#   # Resume after timeout/preemption (continue from latest cluster checkpoint):
#   sbatch latent_pipeline/cluster/submit_train_rorqual.sh --resume latent_pipeline/outputs/latest.pt

set -euo pipefail

# Load modules and activate env
module load python/3.11 scipy-stack/2024a
source "$HOME/envs/crocodile/bin/activate"

# Move to repo root
cd "$SLURM_SUBMIT_DIR"

echo "=== Job info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM:   $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "Dir:    $(pwd)"
echo ""

CONFIG="latent_pipeline/configs/rorqual.yaml"

# Pass any arguments (e.g. --resume <path>) directly to the training script
echo "Starting train_frames with config: $CONFIG"
python latent_pipeline/scripts/stage2b_train_frames.py \
    --config "$CONFIG" \
    "$@"

echo ""
echo "=== Training complete ==="
echo "Outputs in: latent_pipeline/outputs/"
