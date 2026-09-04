#!/usr/bin/env python3
"""Sweep resolutions to find the gradient/speed sweet spot.

For each resolution (128, 256, 512, 1024, full 2048):
  - Measure gradient norm through encoder
  - Measure generator output diversity
  - Time a forward+backward pass
  - Report VRAM usage
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from models.encoder import EmotionEncoder, generate_from_w, generate_from_w_truncated
from data.dataset import CrocodileEncoderDataset, get_train_val_pools


def test_resolution(encoder, G, images_256, device, res, n_trials=3):
    """Test a single resolution: grad norm, diversity, speed, VRAM."""
    # Resize input images to match resolution
    if res <= 256:
        images = F.interpolate(images_256, size=res, mode='bilinear',
                               align_corners=False)
    else:
        images = images_256  # encoder always gets 256 (or smaller)

    # For encoder input, use min(res, 256) since encoder expects reasonable sizes
    enc_input_size = min(res, 256)
    enc_images = F.interpolate(images_256, size=enc_input_size, mode='bilinear',
                               align_corners=False) if enc_input_size != 256 else images_256

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated(device) / 1024**2

    grad_norms = []
    diversities = []
    times = []

    for trial in range(n_trials):
        encoder.train()
        encoder.zero_grad()

        t0 = time.time()

        w = encoder(enc_images)

        if res < 2048:
            gen_img = generate_from_w_truncated(G, w, res)
        else:
            gen_img = generate_from_w(G, w)

        # Downsample gen to match encoder input for loss
        if gen_img.shape[-1] != enc_input_size:
            gen_for_loss = F.interpolate(gen_img, size=enc_input_size,
                                         mode='bilinear', align_corners=False)
        else:
            gen_for_loss = gen_img

        loss = F.mse_loss(gen_for_loss, enc_images)
        loss.backward()

        elapsed = time.time() - t0
        times.append(elapsed)

        total_grad = sum(p.grad.norm().item() ** 2
                         for p in encoder.parameters() if p.grad is not None) ** 0.5
        grad_norms.append(total_grad)

        if gen_img.shape[0] >= 2:
            div = (gen_img[0] - gen_img[1]).abs().mean().item()
            diversities.append(div)

        del gen_img, gen_for_loss, loss
        if res >= 1024:
            torch.cuda.empty_cache()

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2

    return {
        'res': res,
        'grad_norm': sum(grad_norms) / len(grad_norms),
        'diversity': sum(diversities) / len(diversities) if diversities else 0,
        'time_ms': sum(times) / len(times) * 1000,
        'peak_vram_mb': peak_mem,
        'enc_input': enc_input_size,
    }


def main():
    config_path = 'latent_pipeline/configs/default.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    repo_root = config['paths']['repo_root']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load StyleGAN2
    sys.path.insert(0, config['paths']['stylegan_code'])
    import dnnlib, legacy
    import torch_utils.ops.bias_act as _ba
    import torch_utils.ops.upfirdn2d as _up
    _ba._init = lambda: False
    _up._init = lambda: False

    model_path = os.path.join(repo_root, config['paths']['stylegan_model'])
    with dnnlib.util.open_url(model_path) as f:
        G = legacy.load_network_pkl(f)['G_ema']
    G = G.to(device).eval()
    G.requires_grad_(False)

    # Load encoder
    ec = config['encoder']
    encoder = EmotionEncoder(
        channels=tuple(ec['channels']),
        w_dim=ec['w_dim'],
        dropout=ec['dropout'],
    ).to(device)

    # Load 2 images at 256
    frames_dir = os.path.join(repo_root, config['paths']['frames_dir'])
    metadata_dir = os.path.join(repo_root, config['paths']['metadata_dir'])
    manifest_path = os.path.join(metadata_dir, 'cnn_training_manifest.csv')
    _, val_pools = get_train_val_pools()
    dataset = CrocodileEncoderDataset(manifest_path, frames_dir, repo_root,
                                       pool_names=val_pools, target_size=256)

    img0, _ = dataset[0]
    img1, _ = dataset[len(dataset) // 2]
    images_256 = torch.stack([img0, img1]).to(device)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print()

    resolutions = [128, 256, 512, 1024, 2048]
    results = []

    for res in resolutions:
        print(f"Testing {res}x{res}...", end=' ', flush=True)
        try:
            r = test_resolution(encoder, G, images_256, device, res)
            results.append(r)
            print(f"OK ({r['time_ms']:.0f}ms, {r['peak_vram_mb']:.0f}MB)")
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                print(f"OOM!")
                results.append({'res': res, 'grad_norm': 0, 'diversity': 0,
                                'time_ms': 0, 'peak_vram_mb': 0, 'enc_input': 0,
                                'oom': True})
            else:
                raise
        torch.cuda.empty_cache()

    # Print table
    print(f"\n{'Res':>6} {'Enc In':>7} {'Grad Norm':>10} {'Diversity':>10} "
          f"{'Time(ms)':>9} {'VRAM(MB)':>9} {'Status':>8}")
    print('-' * 70)

    baseline = results[-1]['grad_norm'] if results[-1].get('grad_norm', 0) > 0 else results[-2].get('grad_norm', 1)

    for r in results:
        if r.get('oom'):
            print(f"{r['res']:>6} {'':>7} {'':>10} {'':>10} {'':>9} {'':>9} {'OOM':>8}")
        else:
            ratio = r['grad_norm'] / baseline * 100 if baseline > 0 else 0
            print(f"{r['res']:>6} {r['enc_input']:>7} {r['grad_norm']:>10.3f} "
                  f"{r['diversity']:>10.6f} {r['time_ms']:>9.0f} "
                  f"{r['peak_vram_mb']:>9.0f} {ratio:>7.0f}%")


if __name__ == '__main__':
    main()
