#!/usr/bin/env python3
"""Quick reconstruction grid from a checkpoint — no training needed.

Usage:
    python latent_pipeline/scripts/quick_recon.py --config latent_pipeline/configs/default.yaml
    python latent_pipeline/scripts/quick_recon.py --checkpoint latent_pipeline/outputs/best.pt
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from models.encoder import EmotionEncoder
from models.stylegan import load_stylegan, generate
from data.dataset import CrocodileEncoderDataset, get_train_val_pools


def main():
    parser = argparse.ArgumentParser(description='Quick reconstruction grid')
    parser.add_argument('--config', default='latent_pipeline/configs/default.yaml')
    parser.add_argument('--checkpoint', default=None,
                        help='Checkpoint path (default: outputs/latest.pt)')
    parser.add_argument('--output', default=None,
                        help='Output PNG path (default: outputs/quick_recon.png)')
    parser.add_argument('--n-images', type=int, default=16)
    parser.add_argument('--pool', default='val',
                        choices=['val', 'train', 'all'],
                        help='Which pool to sample from')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    repo_root = config['paths']['repo_root']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load StyleGAN2
    G = load_stylegan(config, device)

    # Load encoder
    ec = config['encoder']
    encoder = EmotionEncoder(
        channels=tuple(ec['channels']),
        w_dim=ec['w_dim'],
        dropout=ec['dropout'],
    ).to(device)

    output_dir = os.path.join(repo_root, config['train_frames']['checkpoint_dir'])
    ckpt_path = args.checkpoint or os.path.join(output_dir, 'latest.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()
    epoch = ckpt['epoch']
    print(f"Loaded checkpoint: {ckpt_path} (epoch {epoch})")

    # Load dataset
    frames_dir = os.path.join(repo_root, config['paths']['frames_dir'])
    metadata_dir = os.path.join(repo_root, config['paths']['metadata_dir'])
    manifest_path = os.path.join(metadata_dir, 'cnn_training_manifest.csv')
    train_pools, val_pools = get_train_val_pools()

    if args.pool == 'val':
        pools = val_pools
    elif args.pool == 'train':
        pools = train_pools
    else:
        pools = None
    dataset = CrocodileEncoderDataset(manifest_path, frames_dir, repo_root,
                                       pool_names=pools)
    print(f"Dataset: {len(dataset)} frames (pool={args.pool})")

    # Sample evenly spaced frames
    step = max(1, len(dataset) // args.n_images)
    indices = list(range(0, len(dataset), step))[:args.n_images]

    # Generate reconstructions
    originals = []
    reconstructions = []
    labels = []

    with torch.no_grad():
        for idx in indices:
            img, meta = dataset[idx]
            img_batch = img.unsqueeze(0).to(device)
            w = encoder(img_batch)
            gen_full = generate(G, w)
            gen_img = F.interpolate(gen_full, size=256, mode='bilinear',
                                    align_corners=False)
            del gen_full

            orig = (img + 1) / 2
            recon = (gen_img[0].cpu().clamp(-1, 1) + 1) / 2
            originals.append(orig.permute(1, 2, 0).numpy())
            reconstructions.append(recon.permute(1, 2, 0).numpy())
            labels.append(f"{meta['pool_name']}:{meta['emotion_label']}")

    # Plot grid
    n = len(originals)
    cols = min(8, n)
    rows = math.ceil(n / cols) * 2

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1:
        axes = axes[None, :]

    for i in range(n):
        r = (i // cols) * 2
        c = i % cols
        axes[r, c].imshow(originals[i])
        axes[r, c].set_title(f'Orig: {labels[i]}', fontsize=6)
        axes[r, c].axis('off')
        axes[r + 1, c].imshow(reconstructions[i])
        axes[r + 1, c].set_title('Recon', fontsize=6)
        axes[r + 1, c].axis('off')

    for r in range(rows):
        for c in range(cols):
            idx = (r // 2) * cols + c
            if idx >= n:
                axes[r, c].axis('off')

    fig.suptitle(f'Reconstruction @ epoch {epoch}', fontsize=12)
    plt.tight_layout()

    out_path = args.output or os.path.join(output_dir, 'quick_recon.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
