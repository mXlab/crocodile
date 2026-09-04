#!/usr/bin/env python3
"""Stage 2B: Train the EmotionEncoder (face image → W-space) on real frames.

Fine-tunes weights from Stage 2A (stage2a_train_synthetic.py) — pass them
via --pretrained, or this starts from a randomly-initialized encoder.

Losses:
  1. LPIPS reconstruction (all pools, feeling_it-weighted)
  2. Pixel MSE reconstruction (direct gradient signal)
  3. W-diversity regularization (prevents encoder collapse)
  4. Temporal smoothness (consecutive frames from temporal pools)
  5. Emotion contrastive (pools with emotions, valid labels only)

Usage:
    python latent_pipeline/scripts/stage2b_train_frames.py --config latent_pipeline/configs/default.yaml \\
        --pretrained latent_pipeline/outputs/train_synthetic/best.pt
"""

import argparse
import json
import math
import os
import sys
import time

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from models.encoder import EmotionEncoder
from models.stylegan import load_stylegan, generate, generate_with_grad
from data.dataset import (
    CrocodileEncoderDataset, TemporalAwareSampler, ALL_REAL_POOLS
)


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)



def collate_fn(batch):
    """Custom collate that handles metadata dicts."""
    images = torch.stack([b[0] for b in batch])
    metadata = {
        'pool_name': [b[1]['pool_name'] for b in batch],
        'frame_number': torch.tensor([b[1]['frame_number'] for b in batch]),
        'emotion_label': torch.tensor([b[1]['emotion_label'] for b in batch]),
        'feeling_it': torch.tensor([b[1]['feeling_it'] for b in batch]),
        'has_emotions': torch.tensor([b[1]['has_emotions'] for b in batch]),
        'has_temporal_continuity': torch.tensor([b[1]['has_temporal_continuity'] for b in batch]),
        'is_consecutive': torch.tensor([b[1]['is_consecutive'] for b in batch]),
        'prev_idx': torch.tensor([b[1]['prev_idx'] for b in batch]),
        'manifest_idx': torch.tensor([b[1]['manifest_idx'] for b in batch]),
    }
    return images, metadata


class WQueue:
    """FIFO queue of recent W vectors for computing diversity over more than batch_size samples.

    With batch_size=2, per-batch std is meaningless. This queue accumulates
    recent W vectors so diversity can be computed over a larger window.
    """

    def __init__(self, max_size=256, w_dim=512, device='cpu'):
        self.max_size = max_size
        self.queue = torch.zeros(max_size, w_dim, device=device)
        self.ptr = 0
        self.full = False

    def push(self, w_batch):
        """Add a batch of W vectors (detached) to the queue."""
        w_detached = w_batch.detach()
        for i in range(w_detached.shape[0]):
            self.queue[self.ptr] = w_detached[i]
            self.ptr = (self.ptr + 1) % self.max_size
            if self.ptr == 0:
                self.full = True

    def get_all(self):
        """Return all valid entries in the queue."""
        if self.full:
            return self.queue
        return self.queue[:self.ptr]

    def size(self):
        return self.max_size if self.full else self.ptr


class EmotionContrastiveLoss(nn.Module):
    """Contrastive loss: same-emotion pairs closer, different-emotion pairs apart."""

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, w_vectors, emotion_labels, mask):
        """
        Args:
            w_vectors: (B, 512)
            emotion_labels: (B,) int, -1 for no emotion
            mask: (B,) bool, True for frames with valid emotions
        Returns:
            loss: scalar
        """
        # Filter to valid emotion frames
        valid = mask & (emotion_labels >= 0)
        if valid.sum() < 2:
            return torch.tensor(0.0, device=w_vectors.device)

        w_valid = w_vectors[valid]
        labels_valid = emotion_labels[valid]
        n = w_valid.shape[0]

        # Pairwise distances
        dists = torch.cdist(w_valid.unsqueeze(0), w_valid.unsqueeze(0)).squeeze(0)

        # Same/different masks
        label_eq = labels_valid.unsqueeze(0) == labels_valid.unsqueeze(1)
        # Exclude diagonal
        diag_mask = ~torch.eye(n, dtype=torch.bool, device=w_vectors.device)
        same_mask = label_eq & diag_mask
        diff_mask = ~label_eq & diag_mask

        loss = torch.tensor(0.0, device=w_vectors.device)
        count = 0

        # Pull same-emotion pairs together
        if same_mask.any():
            loss = loss + dists[same_mask].mean()
            count += 1

        # Push different-emotion pairs apart
        if diff_mask.any():
            loss = loss + F.relu(self.margin - dists[diff_mask]).mean()
            count += 1

        return loss / max(count, 1)


def train_one_epoch(encoder, G, train_loader, optimizer, lpips_fn,
                    contrastive_loss_fn, config, device, epoch,
                    all_w_cache=None, w_queue=None):
    """Train for one epoch with gradient accumulation."""
    encoder.train()
    tc = config['train_frames']
    accum_steps = tc['gradient_accumulation_steps']
    feeling_it_mult = tc['loss']['feeling_it_multiplier']
    temporal_weight = tc['loss']['temporal_weight']
    emotion_weight = tc['loss']['emotion_weight']
    mse_weight = tc['loss'].get('mse_weight', 1.0)
    diversity_weight = tc['loss'].get('diversity_weight', 0.5)

    running_loss = {'total': 0, 'lpips': 0, 'mse': 0, 'diversity': 0,
                    'temporal': 0, 'emotion': 0}
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, (images, meta) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", unit='batch')):
        images = images.to(device)
        emotion_labels = meta['emotion_label'].to(device)
        feeling_it = meta['feeling_it'].to(device)
        has_emotions = meta['has_emotions'].to(device)
        is_consecutive = meta['is_consecutive']
        prev_indices = meta['prev_idx']

        B = images.shape[0]

        # Forward: encoder
        w = encoder(images)  # (B, 512)

        # Forward: generator (full 2048, then downsample)
        # G.requires_grad_(False) is set, so G's parameters won't accumulate
        # gradients, but the computation graph through G IS tracked so that
        # reconstruction losses (LPIPS, MSE) can backpropagate to w → encoder.
        gen_full = generate_with_grad(G, w)  # (B, 3, 2048, 2048)
        gen_img = F.interpolate(gen_full, size=256, mode='bilinear',
                                align_corners=False)
        del gen_full

        # 1. LPIPS loss (all frames)
        lpips_per_sample = lpips_fn(images, gen_img).squeeze()  # (B,) or scalar
        if lpips_per_sample.dim() == 0:
            lpips_per_sample = lpips_per_sample.unsqueeze(0)

        # feeling_it weighting
        weights = torch.where(feeling_it > 0.5,
                              torch.full_like(feeling_it, feeling_it_mult),
                              torch.ones_like(feeling_it))
        loss_lpips = (lpips_per_sample * weights).mean()

        # 2. Pixel MSE loss (direct gradient signal to prevent collapse)
        #    Clamp gen output to [-1,1] to match input range and keep MSE meaningful
        gen_img_clamped = gen_img.clamp(-1, 1)
        loss_mse = F.mse_loss(gen_img_clamped, images)

        # 3. W-diversity regularization via queue
        #    With batch_size=2, per-batch std is useless. Instead, maintain a
        #    queue of recent W vectors and penalize low variance over the queue.
        #    Concatenate current w (with grad) + queue (no grad) so loss is
        #    differentiable w.r.t. current batch.
        if w_queue is not None:
            queue_w = w_queue.get_all()
            if queue_w.shape[0] >= 16:
                combined_w = torch.cat([w, queue_w.detach()], dim=0)
                w_std = combined_w.std(dim=0).mean()
                loss_diversity = -torch.log(w_std + 1e-4)
            else:
                loss_diversity = torch.tensor(0.0, device=device)
            w_queue.push(w)  # push after computing loss
        else:
            loss_diversity = torch.tensor(0.0, device=device)

        # 4. Temporal smoothness loss
        loss_temporal = torch.tensor(0.0, device=device)
        consec_mask = is_consecutive.bool()
        if consec_mask.any() and all_w_cache is not None:
            # Get previous W vectors from cache
            consec_indices = torch.where(consec_mask)[0]
            prev_ws = []
            curr_ws = []
            for ci in consec_indices:
                pi = prev_indices[ci.item()].item()
                if pi >= 0 and pi in all_w_cache:
                    prev_ws.append(all_w_cache[pi])
                    curr_ws.append(w[ci.item()])
            if prev_ws:
                prev_ws = torch.stack(prev_ws)
                curr_ws = torch.stack(curr_ws)
                loss_temporal = F.mse_loss(curr_ws, prev_ws)

        # 5. Emotion contrastive loss
        loss_emotion = contrastive_loss_fn(w, emotion_labels, has_emotions.bool())

        # Total loss
        loss = (loss_lpips
                + mse_weight * loss_mse
                + diversity_weight * loss_diversity
                + temporal_weight * loss_temporal
                + emotion_weight * loss_emotion)
        loss = loss / accum_steps

        loss.backward()

        # Update W cache for temporal loss
        if all_w_cache is not None:
            manifest_indices = meta['manifest_idx']
            for i in range(B):
                all_w_cache[manifest_indices[i].item()] = w[i].detach()

        # Gradient accumulation step
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(encoder.parameters(),
                                           tc['gradient_clip_max_norm'])
            optimizer.step()
            optimizer.zero_grad()

        running_loss['total'] += loss.item() * accum_steps
        running_loss['lpips'] += loss_lpips.item()
        running_loss['mse'] += loss_mse.item()
        running_loss['diversity'] += (loss_diversity.item() if torch.is_tensor(loss_diversity) else loss_diversity)
        running_loss['temporal'] += loss_temporal.item()
        running_loss['emotion'] += loss_emotion.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running_loss.items()}


@torch.no_grad()
def validate(encoder, G, val_loader, lpips_fn, device):
    """Validate: compute mean LPIPS on validation set."""
    encoder.eval()
    total_lpips = 0
    n = 0

    for images, meta in val_loader:
        images = images.to(device)
        w = encoder(images)
        gen_full = generate(G, w)
        gen_img = F.interpolate(gen_full, size=256, mode='bilinear',
                                align_corners=False)
        del gen_full
        lpips_val = lpips_fn(images, gen_img).mean().item()
        total_lpips += lpips_val * images.shape[0]
        n += images.shape[0]

    return total_lpips / max(n, 1)


@torch.no_grad()
def save_visual_grid(encoder, G, dataset, device, output_path, n_images=16):
    """Save side-by-side grid of original vs reconstruction.

    Samples evenly spaced frames across the entire dataset to show diversity.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    encoder.eval()
    originals = []
    reconstructions = []
    labels = []

    # Sample evenly spaced indices across the dataset
    step = max(1, len(dataset) // n_images)
    indices = list(range(0, len(dataset), step))[:n_images]

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

    n = len(originals)
    cols = min(8, n)
    rows = math.ceil(n / cols) * 2  # *2 for orig + recon rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1:
        axes = axes[None, :]

    for i in range(n):
        r = (i // cols) * 2
        c = i % cols
        axes[r, c].imshow(originals[i])
        axes[r, c].set_title(f'Orig: {labels[i]}', fontsize=5)
        axes[r, c].axis('off')
        axes[r + 1, c].imshow(reconstructions[i])
        axes[r + 1, c].set_title('Recon', fontsize=5)
        axes[r + 1, c].axis('off')

    # Turn off remaining axes
    for r in range(rows):
        for c in range(cols):
            if r * cols // 2 + c >= n:
                axes[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Train encoder')
    parser.add_argument('--config', default='latent_pipeline/configs/default.yaml')
    parser.add_argument('--resume', default=None, help='Resume within train_frames (restores epoch, optimizer, scheduler)')
    parser.add_argument('--pretrained', default=None, help='Load encoder weights only (cross-phase transfer, resets epoch to 0)')
    args = parser.parse_args()

    config = load_config(args.config)
    tc = config['train_frames']
    repo_root = config['paths']['repo_root']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directory
    output_dir = os.path.join(repo_root, tc['checkpoint_dir'])
    os.makedirs(output_dir, exist_ok=True)

    # Load StyleGAN2
    G = load_stylegan(config, device)

    # Initialize encoder
    ec = config['encoder']
    encoder = EmotionEncoder(
        channels=tuple(ec['channels']),
        w_dim=ec['w_dim'],
        dropout=ec['dropout'],
    ).to(device)

    # Loss functions
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    contrastive_loss_fn = EmotionContrastiveLoss(
        margin=tc['loss']['emotion_contrastive_margin']
    )

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(encoder.parameters(), lr=tc['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=tc['epochs'], eta_min=tc['lr_min'])

    # Datasets
    frames_dir = os.path.join(repo_root, config['paths']['frames_dir'])
    metadata_dir = os.path.join(repo_root, config['paths']['metadata_dir'])
    manifest_path = os.path.join(metadata_dir, 'cnn_training_manifest.csv')

    val_fraction = tc.get('val_fraction', 0.1)

    train_dataset = CrocodileEncoderDataset(manifest_path, frames_dir, repo_root,
                                            pool_names=ALL_REAL_POOLS,
                                            val_fraction=val_fraction, split='train')
    val_dataset = CrocodileEncoderDataset(manifest_path, frames_dir, repo_root,
                                          pool_names=ALL_REAL_POOLS,
                                          val_fraction=val_fraction, split='val')

    # All real frames for visual grid
    grid_dataset = CrocodileEncoderDataset(manifest_path, frames_dir, repo_root,
                                           pool_names=ALL_REAL_POOLS)

    print(f"Train: {len(train_dataset)} frames, Val: {len(val_dataset)} frames, Grid: {len(grid_dataset)} frames")

    # Samplers and loaders
    train_sampler = TemporalAwareSampler(train_dataset, batch_size=tc['batch_size'])
    num_workers = tc.get('num_workers', 2)
    train_loader = DataLoader(train_dataset, batch_size=tc['batch_size'],
                              sampler=train_sampler, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=tc['batch_size'],
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True, collate_fn=collate_fn)

    # W cache for temporal loss (maps manifest_idx -> W tensor)
    w_cache = {}

    # W queue for diversity loss (accumulates recent W vectors)
    w_queue = WQueue(max_size=256, w_dim=ec['w_dim'], device=device)

    # Load checkpoint
    start_epoch = 0
    best_val_lpips = float('inf')
    if args.resume:
        # True resume: restores epoch counter, optimizer and scheduler state
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_lpips = ckpt.get('best_val_lpips', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    elif args.pretrained:
        # Cross-phase transfer: encoder weights only, epoch resets to 0
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt['encoder'])
        print(f"Loaded pretrained encoder weights (trained up to epoch {ckpt['epoch']}), starting fine-tuning from epoch 0")

    # Training log — reload existing history on resume so it accumulates
    # across restarts instead of being overwritten each run.
    log_path = os.path.join(output_dir, 'training_log.json')
    log_entries = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            log_entries = json.load(f)
        log_entries = [e for e in log_entries if e['epoch'] < start_epoch]

    print(f"\nStarting training for {tc['epochs']} epochs")
    for epoch in range(start_epoch, tc['epochs']):
        t0 = time.time()

        # Train
        train_losses = train_one_epoch(
            encoder, G, train_loader, optimizer, lpips_fn,
            contrastive_loss_fn, config, device, epoch, w_cache, w_queue
        )

        # Validate
        val_lpips = validate(encoder, G, val_loader, lpips_fn, device)

        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        # Log
        entry = {
            'epoch': epoch,
            'train_loss_total': train_losses['total'],
            'train_loss_lpips': train_losses['lpips'],
            'train_loss_mse': train_losses['mse'],
            'train_loss_diversity': train_losses['diversity'],
            'train_loss_temporal': train_losses['temporal'],
            'train_loss_emotion': train_losses['emotion'],
            'val_loss_lpips': val_lpips,
            'lr': lr_now,
            'elapsed_s': elapsed,
        }
        log_entries.append(entry)

        print(f"Epoch {epoch:3d} | "
              f"loss={train_losses['total']:.4f} "
              f"lpips={train_losses['lpips']:.4f} "
              f"mse={train_losses['mse']:.4f} "
              f"div={train_losses['diversity']:.2f} "
              f"temp={train_losses['temporal']:.4f} "
              f"emo={train_losses['emotion']:.4f} | "
              f"val_lpips={val_lpips:.4f} | "
              f"lr={lr_now:.2e} | {elapsed:.0f}s")

        # Save latest
        ckpt = {
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_lpips': best_val_lpips,
            'config': config,
        }
        torch.save(ckpt, os.path.join(output_dir, 'latest.pt'))

        # Save best
        if val_lpips < best_val_lpips:
            best_val_lpips = val_lpips
            ckpt['best_val_lpips'] = best_val_lpips
            torch.save(ckpt, os.path.join(output_dir, 'best.pt'))
            print(f"  -> New best val_lpips: {best_val_lpips:.4f}")

        # Visual checks
        if (epoch + 1) % tc['visual_check_every'] == 0 or epoch == 0:
            grid_path = os.path.join(output_dir, f'recon_epoch_{epoch:03d}.png')
            save_visual_grid(encoder, G, grid_dataset, device, grid_path)
            print(f"  -> Saved visual grid: {grid_path}")

        # Save log
        with open(log_path, 'w') as f:
            json.dump(log_entries, f, indent=2)

    print(f"\nTraining complete. Best val_lpips: {best_val_lpips:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()
