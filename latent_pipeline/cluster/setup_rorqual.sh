#!/bin/bash
# One-time setup on Rorqual (run from login node).
#
# Expected directory layout on Rorqual:
#   ~/projects/def-<PI>/crocodile/         <- main project dir
#       code/                              <- this git repo
#       stylegan_Autolume/                 <- StyleGAN2 code
#
# Data uploaded via Globus goes to:
#   ~/projects/def-<PI>/crocodile/code/latent_pipeline/data/
#   ~/projects/def-<PI>/crocodile/code/models/
#
# Usage:
#   cd ~/projects/def-<PI>/crocodile/code
#   bash latent_pipeline/cluster/setup_rorqual.sh
#
# Note: the venv below is created under $HOME, same as the Vulcan setup.
# Rorqual and Vulcan do NOT share a home filesystem, so this must be run
# separately on each cluster even if you've already set up Vulcan.

set -euo pipefail

echo "=== Rorqual environment setup ==="

# Load modules (same Alliance national software stack as other clusters)
module load python/3.11 scipy-stack/2024a

# Create virtualenv in home space (persists across jobs)
VENV_DIR="$HOME/envs/crocodile"
if [ -d "$VENV_DIR" ]; then
    echo "Virtualenv already exists at $VENV_DIR"
else
    echo "Creating virtualenv at $VENV_DIR ..."
    virtualenv --no-download "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --no-index --upgrade pip

# Install packages available from Alliance software stack
echo "Installing from Alliance stack (--no-index) ..."
pip install --no-index torch torchvision
pip install --no-index opencv-python pillow
pip install --no-index pandas numpy matplotlib scikit-learn tqdm pyyaml

# lpips is not in the Alliance stack — download wheel from PyPI via proxy
echo "Installing lpips from PyPI (via proxy) ..."
pip install lpips

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Upload data via Globus (see latent_pipeline/cluster/VULCAN_TRANSFER.md"
echo "     for the file list — same layout applies to Rorqual)"
echo "  2. Edit latent_pipeline/configs/rorqual.yaml with your paths"
echo "  3. Edit latent_pipeline/cluster/submit_train_rorqual.sh with your account"
echo "  4. Submit: sbatch latent_pipeline/cluster/submit_train_rorqual.sh --resume latent_pipeline/outputs/latest.pt"
