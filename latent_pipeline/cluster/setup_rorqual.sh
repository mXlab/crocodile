#!/bin/bash
# One-time setup on Rorqual (run from login node).
#
# Expected directory layout on Rorqual (mirrors the local layout — crocodile/
# and stylegan_Autolume/ are siblings under the project space, no wrapper dir):
#   ~/links/projects/def-sofian/sofian/crocodile/         <- this git repo
#   ~/links/projects/def-sofian/sofian/stylegan_Autolume/ <- StyleGAN2 code
#
# Data goes to:
#   ~/links/projects/def-sofian/sofian/crocodile/latent_pipeline/data/
#   ~/links/projects/def-sofian/sofian/crocodile/models/
#
# Usage:
#   cd ~/links/projects/def-sofian/sofian/crocodile
#   bash latent_pipeline/cluster/setup_rorqual.sh
#
# Note: the venv below is created under $HOME, same as the Vulcan setup.
# Rorqual and Vulcan do NOT share a home filesystem, so this must be run
# separately on each cluster even if you've already set up Vulcan.

set -euo pipefail

echo "=== Rorqual environment setup ==="

# Load modules (same Alliance national software stack as other clusters).
# opencv-python is NOT a real pip package on the Alliance stack — it's a
# dummy wheel that dlopens the system module, so gcc+opencv must be loaded
# BEFORE the venv is created/activated. Same story for cuda: the --no-index
# torch wheel doesn't bundle CUDA, it dynamically links against whatever
# cuda module is loaded — without it, torch silently falls back to CPU
# (torch.cuda.is_available() == False, no error). opencv/4.14.0 and
# cuda/12.6 are Rorqual's defaults (`module avail opencv`/`cuda`) as of 2026-09.
module load python/3.11 scipy-stack/2024a gcc opencv/4.14.0 cuda/12.6

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
pip install --no-index pandas numpy matplotlib scikit-learn tqdm pyyaml requests

# lpips is not in the Alliance stack — download wheel from PyPI via proxy
echo "Installing lpips from PyPI (via proxy) ..."
pip install lpips

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Transfer data (rsync or Globus — see latent_pipeline/cluster/VULCAN_TRANSFER.md"
echo "     for the file list; same files apply to Rorqual)"
echo "  2. Edit latent_pipeline/configs/rorqual.yaml with your paths"
echo "  3. Edit latent_pipeline/cluster/submit_train_rorqual.sh with your account"
echo "  4. Submit: sbatch latent_pipeline/cluster/submit_train_rorqual.sh --resume latent_pipeline/outputs/latest.pt"
