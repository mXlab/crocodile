#!/bin/bash
# One-time setup on Vulcan (run from login node).
#
# Expected directory layout on Vulcan:
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
#   bash latent_pipeline/cluster/setup_vulcan.sh

set -euo pipefail

echo "=== Vulcan environment setup ==="

# Load modules
module load python/3.11 scipy-stack/2024a

# Create virtualenv in project space (persists across jobs)
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
# (Vulcan has a squid proxy with whitelisted domains)
echo "Installing lpips from PyPI (via proxy) ..."
pip install lpips

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Upload data via Globus (see TRANSFER_CHECKLIST below)"
echo "  2. Edit latent_pipeline/configs/vulcan.yaml with your paths"
echo "  3. Submit: sbatch latent_pipeline/cluster/submit_train.sh"
