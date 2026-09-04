#!/usr/bin/env python3
"""Diagnose gradient flow through encoder → generator → loss → backward.

Checks:
1. Do gradients reach encoder parameters through generate_from_w_truncated?
2. Do gradients reach encoder parameters through generate_from_w (full)?
3. Are W vectors diverse or collapsed?
4. Is the generator output sensitive to W changes?
"""

import os
import sys

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


def check_gradient_flow(encoder, G, images, device, gen_fn, label):
    """Run forward+backward and check if encoder gets gradients."""
    encoder.train()
    encoder.zero_grad()

    w = encoder(images)
    print(f"\n=== {label} ===")
    print(f"  w.requires_grad: {w.requires_grad}")
    print(f"  w.shape: {w.shape}")
    print(f"  w stats: mean={w.mean().item():.4f}, std={w.std().item():.4f}")
    print(f"  w[0] vs w[1] L2: {(w[0] - w[1]).norm().item():.6f}")

    gen_img = gen_fn(G, w)
    print(f"  gen_img.shape: {gen_img.shape}")
    print(f"  gen_img.requires_grad: {gen_img.requires_grad}")
    print(f"  gen_img stats: mean={gen_img.mean().item():.4f}, std={gen_img.std().item():.4f}")

    # Pixel difference between generated images for the two inputs
    if gen_img.shape[0] >= 2:
        diff = (gen_img[0] - gen_img[1]).abs().mean().item()
        print(f"  gen_img[0] vs gen_img[1] mean abs diff: {diff:.8f}")

    # Simple MSE loss against input (resized if needed)
    target = F.interpolate(images, size=gen_img.shape[-1], mode='bilinear',
                           align_corners=False) if images.shape[-1] != gen_img.shape[-1] else images
    loss = F.mse_loss(gen_img, target)
    print(f"  MSE loss: {loss.item():.4f}")
    print(f"  loss.requires_grad: {loss.requires_grad}")

    loss.backward()

    # Check encoder gradients
    total_grad_norm = 0
    n_params_with_grad = 0
    n_params_without_grad = 0
    for name, p in encoder.named_parameters():
        if p.grad is not None:
            grad_norm = p.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            n_params_with_grad += 1
            if 'features.0' in name or 'regressor' in name:
                print(f"  grad {name}: norm={grad_norm:.6f}")
        else:
            n_params_without_grad += 1
    total_grad_norm = total_grad_norm ** 0.5
    print(f"  Encoder: {n_params_with_grad} params WITH grad, "
          f"{n_params_without_grad} params WITHOUT grad")
    print(f"  Total encoder grad norm: {total_grad_norm:.6f}")

    return total_grad_norm


def check_generator_sensitivity(G, device, train_res=128):
    """Test if generator produces different outputs for different W vectors."""
    print("\n=== Generator sensitivity to W ===")

    # Two very different W vectors
    w1 = torch.randn(1, 512, device=device) * 0.5
    w2 = torch.randn(1, 512, device=device) * 0.5

    with torch.no_grad():
        img1_trunc = generate_from_w_truncated(G, w1, train_res)
        img2_trunc = generate_from_w_truncated(G, w2, train_res)
        diff_trunc = (img1_trunc - img2_trunc).abs().mean().item()
        print(f"  Truncated ({train_res}): diff={diff_trunc:.6f}")

        img1_full = generate_from_w(G, w1)
        img2_full = generate_from_w(G, w2)
        diff_full = (img1_full - img2_full).abs().mean().item()
        print(f"  Full (2048): diff={diff_full:.6f}")

    # Same W vector, check determinism
    with torch.no_grad():
        img1a = generate_from_w_truncated(G, w1, train_res)
        img1b = generate_from_w_truncated(G, w1, train_res)
        det_diff = (img1a - img1b).abs().mean().item()
        print(f"  Determinism check (same W): diff={det_diff:.8f}")


def check_w_gradient_through_truncated(G, device, train_res=128):
    """Directly test: does loss.backward() produce dL/dw through truncated gen?"""
    print(f"\n=== Direct w gradient test (truncated @ {train_res}) ===")

    w = torch.randn(1, 512, device=device, requires_grad=True)
    img = generate_from_w_truncated(G, w, train_res)
    loss = img.mean()  # trivial loss
    loss.backward()

    if w.grad is not None:
        print(f"  w.grad norm: {w.grad.norm().item():.6f} (GOOD - gradients flow)")
    else:
        print(f"  w.grad is None (BAD - NO gradient flow!)")

    print(f"\n=== Direct w gradient test (full 2048) ===")
    w2 = torch.randn(1, 512, device=device, requires_grad=True)
    img2 = generate_from_w(G, w2)
    loss2 = img2.mean()
    loss2.backward()

    if w2.grad is not None:
        print(f"  w.grad norm: {w2.grad.norm().item():.6f} (GOOD - gradients flow)")
    else:
        print(f"  w.grad is None (BAD - NO gradient flow!)")


def check_block_details(G, train_res=128):
    """Print block architecture details to verify torgb exists at each level."""
    print(f"\n=== StyleGAN2 block details ===")
    print(f"  block_resolutions: {G.synthesis.block_resolutions}")
    print(f"  num_ws: {G.num_ws}")

    for res in G.synthesis.block_resolutions:
        block = getattr(G.synthesis, f'b{res}')
        has_torgb = hasattr(block, 'torgb') and block.torgb is not None
        print(f"  b{res}: num_conv={block.num_conv}, num_torgb={block.num_torgb}, "
              f"has_torgb={has_torgb}, architecture={getattr(block, 'architecture', '?')}")
        if res >= train_res:
            print(f"    ^ truncation point")
            break


def main():
    config_path = 'latent_pipeline/configs/default.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    repo_root = config['paths']['repo_root']
    train_res = config['training'].get('train_resolution', 256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, train_resolution: {train_res}")

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

    # Check block details
    check_block_details(G, train_res)

    # Direct gradient test (most important)
    check_w_gradient_through_truncated(G, device, train_res)

    # Generator sensitivity
    check_generator_sensitivity(G, device, train_res)

    # Load encoder (fresh, random init)
    ec = config['encoder']
    encoder = EmotionEncoder(
        channels=tuple(ec['channels']),
        w_dim=ec['w_dim'],
        dropout=ec['dropout'],
    ).to(device)

    # Load 2 images
    frames_dir = os.path.join(repo_root, config['paths']['frames_dir'])
    metadata_dir = os.path.join(repo_root, config['paths']['metadata_dir'])
    manifest_path = os.path.join(metadata_dir, 'cnn_training_manifest.csv')
    _, val_pools = get_train_val_pools()
    dataset = CrocodileEncoderDataset(manifest_path, frames_dir, repo_root,
                                       pool_names=val_pools, target_size=train_res)

    img0, _ = dataset[0]
    img1, _ = dataset[len(dataset) // 2]
    images = torch.stack([img0, img1]).to(device)
    print(f"\nLoaded 2 images, shape: {images.shape}")
    print(f"Image pixel diff: {(images[0] - images[1]).abs().mean().item():.4f}")

    # Test gradient flow: truncated
    def trunc_gen(G, w):
        return generate_from_w_truncated(G, w, train_res)

    grad_trunc = check_gradient_flow(encoder, G, images, device, trunc_gen,
                                      f"Truncated gen @ {train_res}")

    # Test gradient flow: full (for comparison)
    def full_gen(G, w):
        img = generate_from_w(G, w)
        return F.interpolate(img, size=train_res, mode='bilinear', align_corners=False)

    grad_full = check_gradient_flow(encoder, G, images, device, full_gen,
                                     "Full gen → downsample")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Truncated gen encoder grad norm: {grad_trunc:.6f}")
    print(f"  Full gen encoder grad norm:      {grad_full:.6f}")
    if grad_trunc < 1e-8:
        print(f"  DIAGNOSIS: Gradient flow BROKEN through truncated generator!")
    elif grad_trunc < grad_full * 0.01:
        print(f"  DIAGNOSIS: Truncated gradients are {grad_full/max(grad_trunc,1e-10):.0f}x "
              f"weaker than full - may cause slow learning")
    else:
        print(f"  DIAGNOSIS: Gradient flow looks OK through both paths")


if __name__ == '__main__':
    main()
