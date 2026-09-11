#!/bin/bash
#SBATCH --job-name=crocodile-gen-synthetic
#SBATCH --account=def-sofian
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Generates the synthetic (image, W) pairs that stage2a_train_synthetic.py
# and stage2a_discriminator_init.py both pretrain on. Not privacy-restricted
# (pure StyleGAN2 samples, no actress recordings) but at ~25GB for 10k images
# it was never going to arrive via git pull -- generating it fresh here is
# simpler than transferring the laptop's copy over the network, and doesn't
# need to be the literal same images: any freshly sampled set serves the
# same pretraining purpose.
#
# --time is a guess (untested on Rorqual) -- check the .out log's actual
# elapsed time and adjust before relying on it for a much larger --n.
#
# IMPORTANT: --output must be passed explicitly and must match
# paths.synthetic_dir in the config (latent_pipeline/data/synthetic) --
# generate_synthetic.py's own default is paths.outputs_dir/synthetic
# (a different directory) when --output is omitted.
#
# Usage:
#   cd ~/links/projects/def-sofian/sofian/crocodile
#   sbatch latent_pipeline/cluster/submit_generate_synthetic_rorqual.sh

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

echo "Starting generate_synthetic with config: $CONFIG"
python latent_pipeline/scripts/generate_synthetic.py \
    --config "$CONFIG" \
    --output latent_pipeline/data/synthetic \
    --n 10000 --resolution 256 --psi 0.7 --seed 42 \
    --batch-size 16 \
    "$@"

echo ""
echo "=== Generation complete ==="
echo "Outputs in: latent_pipeline/data/synthetic/"
