#!/bin/bash
#SBATCH --job-name=crocodile-encoder
#SBATCH --account=def-CHANGEME
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Phase 2B: Fine-tune EmotionEncoder on real frames (Alliance cluster).
#
# With 48GB VRAM we can run batch_size=8 at full 2048 gen,
# vs batch_size=1 on the laptop's 6GB RTX 3060.
#
# Usage:
#   cd ~/projects/def-<PI>/crocodile/code
#
#   # Continue fine-tuning from uploaded laptop checkpoint (first run):
#   sbatch latent_pipeline/cluster/submit_train.sh --resume latent_pipeline/outputs/best.pt
#
#   # Resume after timeout/preemption (continue from latest cluster checkpoint):
#   sbatch latent_pipeline/cluster/submit_train.sh --resume latent_pipeline/outputs/latest.pt

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

CONFIG="latent_pipeline/configs/vulcan.yaml"

# Pass any arguments (e.g. --resume <path>) directly to the training script
echo "Starting train_frames with config: $CONFIG"
python latent_pipeline/scripts/train_frames.py \
    --config "$CONFIG" \
    "$@"

echo ""
echo "=== Training complete ==="
echo "Outputs in: latent_pipeline/outputs/"
