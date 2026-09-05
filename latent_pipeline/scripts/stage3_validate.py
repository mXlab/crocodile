#!/usr/bin/env python3
"""Stage 3: Validate encoder quality via optimization-based inversion comparison.

1. Select ~30 biodata pool frames (diverse emotions + feeling_it)
2. Run optimization-based inversion (Adam, 1000 steps, LPIPS+L2)
3. Compare CNN encoder LPIPS vs optimization LPIPS
4. PCA/t-SNE visualization of W vectors colored by emotion
5. Visual grid: original / CNN recon / optimization recon

Usage:
    python latent_pipeline/scripts/stage3_validate.py --config latent_pipeline/configs/default.yaml
"""

import argparse
import math
import os
import sys

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from models.encoder import EmotionEncoder
from models.stylegan import load_stylegan, generate, generate_with_grad
from data.dataset import CrocodileEncoderDataset, EMOTION_LABELS


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def select_validation_frames(dataset, n_frames=30):
    """Select diverse frames: various emotions + feeling_it=True."""
    # Group by emotion
    emotion_groups = {}
    feeling_it_indices = []

    for i in range(len(dataset)):
        row = dataset.manifest.iloc[i]
        pool = row['pool_name']
        # Only from biodata pools (session_[1-4]S)
        if not pool.startswith('session_') or pool == 'session_1X':
            continue

        emo = row['emotion_label']
        fi = row['feeling_it']

        if emo != 'none':
            emotion_groups.setdefault(emo, []).append(i)
        if fi > 0.5:
            feeling_it_indices.append(i)

    # Select: at least 1 per emotion, fill rest with feeling_it
    selected = set()
    for emo, indices in emotion_groups.items():
        idx = indices[len(indices) // 2]  # middle frame
        selected.add(idx)

    # Add feeling_it frames
    for idx in feeling_it_indices:
        if len(selected) >= n_frames:
            break
        selected.add(idx)

    # Fill if needed
    all_biodata = [i for i in range(len(dataset))
                   if dataset.manifest.iloc[i]['pool_name'].startswith('session_')
                   and dataset.manifest.iloc[i]['pool_name'] != 'session_1X']
    for idx in all_biodata:
        if len(selected) >= n_frames:
            break
        selected.add(idx)

    return sorted(selected)[:n_frames]


@torch.no_grad()
def encoder_inversion(encoder, G, images, device):
    """Get W vectors from encoder.

    Runs the frozen generator one image at a time — a full batch at 2048px
    generator resolution is too much VRAM for anything smaller than a
    data-center GPU (fine on an H100, OOMs on an 8GB laptop card).
    """
    encoder.eval()
    ws, gen_imgs = [], []
    for i in range(images.shape[0]):
        img = images[i:i + 1].to(device)
        w = encoder(img)
        gen_full = generate(G, w)
        gen_img = F.interpolate(gen_full, size=256, mode='bilinear', align_corners=False)
        del gen_full
        ws.append(w)
        gen_imgs.append(gen_img)
    return torch.cat(ws), torch.cat(gen_imgs)


def optimization_inversion(G, target_images, lpips_fn, device, lr=0.01,
                           steps=1000):
    """Optimization-based inversion starting from mean W."""
    B = target_images.shape[0]
    results_w = []
    results_img = []

    for i in tqdm(range(B), desc="Optimization inversion"):
        target = target_images[i:i+1].to(device)

        # Initialize from zeros (mean W approximation)
        w = torch.zeros(1, G.w_dim, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            gen_full = generate_with_grad(G, w)
            gen_img = F.interpolate(gen_full, size=256, mode='bilinear',
                                    align_corners=False)
            del gen_full
            loss_lpips = lpips_fn(target, gen_img).mean()
            loss_l2 = F.mse_loss(target, gen_img) * 0.1
            loss = loss_lpips + loss_l2
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            gen_full = generate(G, w)
            gen_img = F.interpolate(gen_full, size=256, mode='bilinear',
                                    align_corners=False)
            del gen_full

        results_w.append(w.detach())
        results_img.append(gen_img.detach())

    return torch.cat(results_w), torch.cat(results_img)


def save_comparison_grid(originals, cnn_recons, opt_recons, output_path):
    """Save 3-column grid: original / CNN / optimization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = originals.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        for j, (img, title) in enumerate([
            (originals[i], 'Original'),
            (cnn_recons[i], 'CNN Encoder'),
            (opt_recons[i], 'Optimization'),
        ]):
            img_np = ((img.cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
            axes[i, j].imshow(img_np)
            axes[i, j].set_title(title, fontsize=8)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison grid: {output_path}")


def save_w_visualization(w_vectors, emotion_labels, output_path):
    """PCA/t-SNE of W vectors colored by emotion."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    w_np = w_vectors.cpu().numpy()
    labels = emotion_labels.cpu().numpy()

    # PCA to 2D
    pca = PCA(n_components=2)
    w_2d = pca.fit_transform(w_np)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))

    for i, label_idx in enumerate(unique_labels):
        mask = labels == label_idx
        label_name = EMOTION_LABELS[label_idx] if 0 <= label_idx < len(EMOTION_LABELS) else 'none'
        ax.scatter(w_2d[mask, 0], w_2d[mask, 1], c=[cmap(i)], label=label_name,
                   s=50, alpha=0.7)

    ax.legend(fontsize=8)
    ax.set_title('W-space PCA by emotion')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved W visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Stage 3: Validate encoder')
    parser.add_argument('--config', default='latent_pipeline/configs/default.yaml')
    parser.add_argument('--checkpoint', default=None,
                        help='Encoder checkpoint (default: outputs/best.pt)')
    args = parser.parse_args()

    config = load_config(args.config)
    vc = config['validation']
    repo_root = config['paths']['repo_root']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = os.path.join(repo_root, config['train_frames']['checkpoint_dir'])
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    G = load_stylegan(config, device)

    # Load encoder
    ec = config['encoder']
    encoder = EmotionEncoder(
        channels=tuple(ec['channels']),
        w_dim=ec['w_dim'],
        dropout=ec['dropout'],
    ).to(device)

    ckpt_path = args.checkpoint or os.path.join(output_dir, 'best.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    print(f"Loaded encoder from {ckpt_path} (epoch {ckpt['epoch']})")

    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # Load dataset (all pools for training sessions)
    frames_dir = os.path.join(repo_root, config['paths']['frames_dir'])
    metadata_dir = os.path.join(repo_root, config['paths']['metadata_dir'])
    manifest_path = os.path.join(metadata_dir, 'cnn_training_manifest.csv')

    # Use train sessions only for validation frame selection
    train_pools = ['session_1S', 'session_2S', 'session_3S']
    dataset = CrocodileEncoderDataset(manifest_path, frames_dir, repo_root,
                                      pool_names=train_pools)

    # Select validation frames
    val_indices = select_validation_frames(dataset, n_frames=vc['n_frames'])
    print(f"Selected {len(val_indices)} validation frames")

    # Collect images and metadata
    images_list = []
    emotions_list = []
    for idx in val_indices:
        img, meta = dataset[idx]
        images_list.append(img)
        emotions_list.append(meta['emotion_label'])

    images = torch.stack(images_list)
    emotion_labels = torch.tensor(emotions_list)

    # CNN encoder inversion
    print("\nCNN encoder inversion...")
    cnn_w, cnn_recon = encoder_inversion(encoder, G, images, device)

    # Compute CNN LPIPS
    with torch.no_grad():
        cnn_lpips_vals = lpips_fn(images.to(device), cnn_recon).squeeze()

    print(f"CNN LPIPS: mean={cnn_lpips_vals.mean():.4f}, "
          f"std={cnn_lpips_vals.std():.4f}")

    # Optimization-based inversion
    print("\nOptimization-based inversion...")
    opt_w, opt_recon = optimization_inversion(
        G, images, lpips_fn, device,
        lr=vc['optim_lr'], steps=vc['optim_steps']
    )

    # Compute optimization LPIPS
    with torch.no_grad():
        opt_lpips_vals = lpips_fn(images.to(device), opt_recon).squeeze()

    print(f"Opt LPIPS: mean={opt_lpips_vals.mean():.4f}, "
          f"std={opt_lpips_vals.std():.4f}")

    # Compare
    gap = (cnn_lpips_vals - opt_lpips_vals).mean().item()
    print(f"\nLPIPS gap (CNN - Opt): {gap:.4f}")
    if gap <= vc['lpips_threshold']:
        print(f"PASS: Gap ({gap:.4f}) <= threshold ({vc['lpips_threshold']})")
    else:
        print(f"WARN: Gap ({gap:.4f}) > threshold ({vc['lpips_threshold']})")

    # Save visualizations
    save_comparison_grid(images, cnn_recon, opt_recon,
                         os.path.join(output_dir, 'validation_comparison.png'))
    save_w_visualization(cnn_w, emotion_labels,
                         os.path.join(output_dir, 'w_space_pca.png'))

    # Summary report
    report = {
        'n_frames': len(val_indices),
        'cnn_lpips_mean': cnn_lpips_vals.mean().item(),
        'cnn_lpips_std': cnn_lpips_vals.std().item(),
        'opt_lpips_mean': opt_lpips_vals.mean().item(),
        'opt_lpips_std': opt_lpips_vals.std().item(),
        'lpips_gap': gap,
        'threshold': vc['lpips_threshold'],
        'pass': gap <= vc['lpips_threshold'],
    }
    import json
    report_path = os.path.join(output_dir, 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")


if __name__ == '__main__':
    main()
