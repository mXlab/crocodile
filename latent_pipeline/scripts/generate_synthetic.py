#!/usr/bin/env python3
"""Generate a synthetic (W, image) dataset using StyleGAN2.

Samples W vectors from the generator's learned distribution (with optional
truncation), synthesises the corresponding face images at a chosen resolution,
and writes both to disk for later use.

Outputs
-------
<output_dir>/
    images/          PNG images named 000000.png, 000001.png, …
    w_vectors.csv    n_images rows × 512 columns (w0 … w511)
    grid.png         Optional quick overview grid (use --no-grid to skip)

Usage examples
--------------
# 512 images at 256x256, psi=0.7, save everything
python latent_pipeline/scripts/generate_synthetic.py \\
    --config latent_pipeline/configs/default.yaml \\
    --n 512 --resolution 256 --psi 0.7 --seed 42

# 64 diagnostic images at full 2048x2048 (SLOW — uses a lot of VRAM)
python latent_pipeline/scripts/generate_synthetic.py \\
    --config latent_pipeline/configs/default.yaml \\
    --n 64 --resolution 2048 --psi 1.0

# W vectors only, no image files (fastest — just the CSV)
python latent_pipeline/scripts/generate_synthetic.py \\
    --config latent_pipeline/configs/default.yaml \\
    --n 10000 --no-images --no-grid

# Quick sanity check: 16 images, show a grid
python latent_pipeline/scripts/generate_synthetic.py \\
    --config latent_pipeline/configs/default.yaml \\
    --n 16 --resolution 128 --psi 0.5
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from models.stylegan import load_stylegan, compute_w_mean, sample_w, generate


def save_grid(image_paths, output_path, max_cols=8, img_size=128):
    """Save a quick overview grid from a list of image file paths."""
    n = len(image_paths)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    # Normalise axes to a 2-D array regardless of shape
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i, path in enumerate(image_paths):
        r, c = divmod(i, cols)
        img = Image.open(path).resize((img_size, img_size), Image.LANCZOS)
        axes[r][c].imshow(np.array(img))
        axes[r][c].set_title(str(i), fontsize=6)
        axes[r][c].axis('off')

    # Hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved grid: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic (W, image) dataset from StyleGAN2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', default='latent_pipeline/configs/default.yaml',
                        help='Path to YAML config')
    parser.add_argument('--n', type=int, default=256,
                        help='Number of (W, image) pairs to generate')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Output image resolution (e.g. 128, 256, 512, 1024, 2048). '
                             'Values < 2048 use truncated synthesis for speed.')
    parser.add_argument('--psi', type=float, default=0.7,
                        help='Truncation psi: 1.0=full diversity, 0.0=all identical. '
                             'Lower values → more "average" but higher quality faces.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: random)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Generator batch size (reduce if VRAM is limited)')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: latent_pipeline/outputs/synthetic)')
    parser.add_argument('--no-images', action='store_true',
                        help='Skip saving PNG image files (only write w_vectors.csv)')
    parser.add_argument('--no-grid', action='store_true',
                        help='Skip saving the overview grid.png')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    repo_root = config['paths']['repo_root']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Output directory
    output_dir = args.output or os.path.join(repo_root,
                                              config['paths']['outputs_dir'],
                                              'synthetic')
    images_dir = os.path.join(output_dir, 'images')
    if not args.no_images:
        os.makedirs(images_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output: {output_dir}")
    print(f"Generating {args.n} images at {args.resolution}x{args.resolution}, "
          f"psi={args.psi}, seed={args.seed}")

    # Load StyleGAN2
    G = load_stylegan(config, device)

    # Compute W mean for truncation (only if psi < 1.0)
    w_mean = None
    if args.psi < 1.0:
        print(f"Computing W mean for truncation (psi={args.psi})…")
        w_mean = compute_w_mean(G, device)

    # Sample all W vectors at once (fast, in CPU to save VRAM)
    print("Sampling W vectors…")
    all_w = sample_w(G, args.n, psi=args.psi, seed=args.seed,
                     device=device, w_mean=w_mean)
    print(f"  W shape: {all_w.shape}  "
          f"norm: mean={all_w.norm(dim=1).mean():.2f} "
          f"std={all_w.norm(dim=1).std():.2f}")

    # Generate images in batches
    saved_paths = []
    w_rows = []
    n_batches = math.ceil(args.n / args.batch_size)

    print(f"Generating images in {n_batches} batches of {args.batch_size}…")
    for b in tqdm(range(n_batches), desc="Generating"):
        start = b * args.batch_size
        end = min(start + args.batch_size, args.n)
        w_batch = all_w[start:end].to(device)

        imgs = generate(G, w_batch, resolution=args.resolution)
        # imgs: (bs, 3, H, W) in [-1, 1]

        for i, (img_t, w_t) in enumerate(zip(imgs, w_batch)):
            idx = start + i
            w_np = w_t.cpu().numpy()
            w_rows.append(w_np)

            if not args.no_images:
                # Convert to uint8 PNG
                img_np = ((img_t.cpu().clamp(-1, 1) + 1) / 2 * 255).byte()
                img_np = img_np.permute(1, 2, 0).numpy()
                img_path = os.path.join(images_dir, f'{idx:06d}.png')
                Image.fromarray(img_np).save(img_path)
                saved_paths.append(img_path)

        del imgs, w_batch

    # Save W vectors CSV
    w_array = np.stack(w_rows, axis=0)  # (n, 512)
    w_dim = w_array.shape[1]
    cols = [f'w{i}' for i in range(w_dim)]
    df = pd.DataFrame(w_array, columns=cols)
    df.insert(0, 'idx', range(args.n))
    csv_path = os.path.join(output_dir, 'w_vectors.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved W vectors: {csv_path}  ({args.n} rows × {w_dim} dims)")

    # Save grid
    if not args.no_grid and saved_paths:
        grid_n = min(64, len(saved_paths))
        grid_paths = saved_paths[:grid_n]
        grid_path = os.path.join(output_dir, 'grid.png')
        save_grid(grid_paths, grid_path)

    print("\nDone.")
    print(f"  Images: {output_dir}/images/  ({len(saved_paths)} files)")
    print(f"  W CSV:  {csv_path}")


if __name__ == '__main__':
    main()
