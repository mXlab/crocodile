#!/usr/bin/env python3
"""Phase 2A: Train EmotionEncoder on synthetic (image, W) pairs.

Trains the encoder as a supervised W-space regressor using pre-generated
images and their corresponding W vectors.  No StyleGAN2 generator is
needed during training — images are loaded from disk and W vectors from
a CSV file.

Loss: MSE(encoder(image_resized), w_target)

Because there is no generator in the loop, large batch sizes (32+) are
possible and training is fast (~milliseconds per batch).

Every visual_check_every epochs a recon_epoch_*.png grid is saved showing
synthetic originals (resized to train_resolution) alongside their
StyleGAN2 reconstructions via the current encoder weights.

The saved checkpoint is compatible with train_frames.py (same encoder
state_dict format), so the pre-trained weights can be used as
initialisation for Phase 2B fine-tuning.

Usage:
    python latent_pipeline/scripts/train_synthetic.py \\
        --config latent_pipeline/configs/default.yaml

    # Resume from checkpoint:
    python latent_pipeline/scripts/train_synthetic.py \\
        --config latent_pipeline/configs/default.yaml \\
        --resume latent_pipeline/outputs/train_synthetic/latest.pt
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from models.encoder import EmotionEncoder
from models.stylegan import load_stylegan, generate


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticWDataset(Dataset):
    """Dataset of pre-generated (image, W) pairs.

    Images are read from <synthetic_dir>/images/<idx:06d>.png and resized
    to target_size × target_size.  W vectors are read from w_vectors.csv
    where row i corresponds to image i.

    Images are normalised to [-1, 1] to match the encoder's expected input.
    """

    def __init__(self, synthetic_dir: str, target_size: int = 128):
        self.images_dir = os.path.join(synthetic_dir, 'images')
        self.target_size = target_size

        csv_path = os.path.join(synthetic_dir, 'w_vectors.csv')
        df = pd.read_csv(csv_path)
        w_cols = [c for c in df.columns if c.startswith('w')]
        self.w_vectors = torch.tensor(df[w_cols].values, dtype=torch.float32)
        self.n = len(self.w_vectors)
        print(f"SyntheticWDataset: {self.n} samples, "
              f"W dim={self.w_vectors.shape[1]}, "
              f"target_size={target_size}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, f'{idx:06d}.png')
        img = Image.open(img_path).convert('RGB')
        if img.size[0] != self.target_size:
            img = img.resize((self.target_size, self.target_size), Image.LANCZOS)
        # Convert to tensor in [-1, 1]
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
        w = self.w_vectors[idx]
        return img_t, w


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(encoder, loader, optimizer, device):
    encoder.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for imgs, w_target in tqdm(loader, desc='  train', leave=False):
        imgs = imgs.to(device)
        w_target = w_target.to(device)

        w_pred = encoder(imgs)
        loss = F.mse_loss(w_pred, w_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(encoder, loader, device):
    encoder.eval()
    total_loss = 0.0
    n_batches = 0
    for imgs, w_target in loader:
        imgs = imgs.to(device)
        w_target = w_target.to(device)
        w_pred = encoder(imgs)
        total_loss += F.mse_loss(w_pred, w_target).item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Visual grid
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_visual_grid(encoder, G, dataset, device, output_path, n_images=16):
    """Save side-by-side grid: synthetic original (resized) vs StyleGAN2 recon."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    encoder.eval()
    originals = []
    reconstructions = []

    step = max(1, len(dataset) // n_images)
    indices = list(range(0, len(dataset), step))[:n_images]

    for idx in indices:
        img, _ = dataset[idx]
        img_batch = img.unsqueeze(0).to(device)
        w = encoder(img_batch)
        gen_full = generate(G, w)
        gen_img = F.interpolate(gen_full, size=img.shape[-1], mode='bilinear',
                                align_corners=False)
        del gen_full

        orig = (img + 1) / 2
        recon = (gen_img[0].cpu().clamp(-1, 1) + 1) / 2
        originals.append(orig.permute(1, 2, 0).numpy())
        reconstructions.append(recon.permute(1, 2, 0).numpy())

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
        axes[r, c].set_title('Synthetic', fontsize=5)
        axes[r, c].axis('off')
        axes[r + 1, c].imshow(reconstructions[i])
        axes[r + 1, c].set_title('Recon', fontsize=5)
        axes[r + 1, c].axis('off')

    for r in range(rows):
        for c in range(cols):
            if (r // 2) * cols + c >= n:
                axes[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Phase 2A: train encoder on synthetic (image, W) pairs'
    )
    parser.add_argument('--config', default='latent_pipeline/configs/default.yaml')
    parser.add_argument('--resume', default=None,
                        help='Checkpoint path to resume from')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tc = config['train_synthetic']
    ec = config['encoder']
    repo_root = config['paths']['repo_root']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = os.path.join(repo_root, tc['checkpoint_dir'])
    os.makedirs(output_dir, exist_ok=True)

    # StyleGAN2 (for visual grids only — not used in training loss)
    G = load_stylegan(config, device)

    # Dataset
    synthetic_dir = os.path.join(repo_root, config['paths']['synthetic_dir'])
    full_dataset = SyntheticWDataset(synthetic_dir, target_size=tc['train_resolution'])

    val_n = int(len(full_dataset) * tc['val_split'])
    train_n = len(full_dataset) - val_n
    train_dataset, val_dataset = random_split(
        full_dataset, [train_n, val_n],
        generator=torch.Generator().manual_seed(0),
    )
    print(f"Train: {train_n}  Val: {val_n}")

    train_loader = DataLoader(train_dataset, batch_size=tc['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=tc['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)

    # Encoder
    encoder = EmotionEncoder(
        channels=tuple(ec['channels']),
        w_dim=ec['w_dim'],
        dropout=ec['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Encoder parameters: {n_params:,}")

    optimizer = torch.optim.Adam(encoder.parameters(), lr=tc['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=tc['epochs'], eta_min=tc['lr_min'])

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt['encoder'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    log_path = os.path.join(output_dir, 'train_synthetic_log.json')
    log_entries = []

    print(f"\nStarting train_synthetic for {tc['epochs']} epochs "
          f"at {tc['train_resolution']}×{tc['train_resolution']}, "
          f"batch_size={tc['batch_size']}")

    for epoch in range(start_epoch, tc['epochs']):
        t0 = time.time()
        train_loss = train_one_epoch(encoder, train_loader, optimizer, device)
        val_loss = validate(encoder, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d} | train_mse={train_loss:.6f}  "
              f"val_mse={val_loss:.6f} | lr={lr_now:.2e} | {elapsed:.0f}s")

        entry = {'epoch': epoch, 'train_mse': train_loss,
                 'val_mse': val_loss, 'lr': lr_now, 'elapsed_s': elapsed}
        log_entries.append(entry)

        ckpt = {
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config,
        }
        torch.save(ckpt, os.path.join(output_dir, 'latest.pt'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt['best_val_loss'] = best_val_loss
            torch.save(ckpt, os.path.join(output_dir, 'best.pt'))
            print(f"  -> New best val_mse: {best_val_loss:.6f}")

        visual_every = tc.get('visual_check_every', 5)
        if (epoch + 1) % visual_every == 0 or epoch == 0:
            grid_path = os.path.join(output_dir, f'recon_epoch_{epoch:03d}.png')
            save_visual_grid(encoder, G, val_dataset, device, grid_path)
            print(f"  -> Saved visual grid: {grid_path}")

        with open(log_path, 'w') as f:
            json.dump(log_entries, f, indent=2)

    print(f"\ntrain_synthetic complete. Best val_mse: {best_val_loss:.6f}")
    print(f"Checkpoint: {output_dir}/best.pt")
    print(f"\nNext: fine-tune on real frames with train_frames.py --resume {output_dir}/best.pt")


if __name__ == '__main__':
    main()
