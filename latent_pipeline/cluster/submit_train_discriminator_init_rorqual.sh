#!/bin/bash
#SBATCH --job-name=crocodile-disc-encoder
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
# Phase 2A (discriminator-init variant): pretrain a DiscriminatorEncoder
# (StyleGAN2's own discriminator trunk, repurposed as an image -> W encoder)
# on synthetic (image, W) pairs, staged so a freshly-initialised head
# doesn't distort the pretrained trunk. See
# models/discriminator_encoder.py and scripts/stage2a_discriminator_init.py
# for the full writeup and PIPELINE.md's session notes (2026-09-11/12).
#
# --time is a guess (untested on Rorqual) -- the laptop's 26-epoch schedule
# ran ~5.5min/epoch on an 8GB RTX 5060 with num_workers=4 before hitting a
# DataLoader deadlock (fixed here via train_discriminator_init.num_workers=8,
# same fix already proven for stage2b on this cluster); an H100 with 4x the
# batch size should be well under this budget, but check the first run's
# actual per-epoch time and adjust before relying on it.
#
# Unlike stage2b_train_frames.py (fine-tunes an existing checkpoint), this
# script always starts fresh -- there's no --resume/--pretrained here because
# a single run covers both phase 1 (frozen trunk) and phase 2 (progressive
# unfreeze) in one schedule. Re-running from scratch is the only way to
# change phase timings (train_discriminator_init.phase1_epochs etc in the
# config) after the fact.
#
# Usage:
#   cd ~/links/projects/def-sofian/sofian/crocodile
#   sbatch latent_pipeline/cluster/submit_train_discriminator_init_rorqual.sh
#
# Next step after this completes -- fine-tune on real frames (same job type
# as the existing EmotionEncoder path, just pointed at this checkpoint):
#   sbatch latent_pipeline/cluster/submit_train_rorqual.sh \
#       --encoder-arch discriminator_init \
#       --pretrained latent_pipeline/outputs/train_discriminator_init/best.pt

set -euo pipefail

# Load modules and activate env. gcc+opencv+cuda must be loaded before the
# venv is activated — see setup_rorqual.sh for why (without cuda loaded,
# torch silently runs on CPU with no error). Keep versions in sync with
# whatever's in setup_rorqual.sh.
module load python/3.11 scipy-stack/2024a gcc opencv/4.14.0 cuda/12.6
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

echo "Starting stage2a_discriminator_init with config: $CONFIG"
python latent_pipeline/scripts/stage2a_discriminator_init.py \
    --config "$CONFIG" \
    "$@"

echo ""
echo "=== Training complete ==="
echo "Outputs in: latent_pipeline/outputs/train_discriminator_init/"
