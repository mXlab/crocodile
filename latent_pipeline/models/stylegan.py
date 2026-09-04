"""StyleGAN2 utility module for the latent pipeline.

Centralises all StyleGAN2 interactions so that training scripts, generation
scripts, and validation scripts share the same loading and generation logic.

Public API
----------
load_stylegan(config)               -> G (nn.Module, eval, on device)
compute_w_mean(G, device, ...)      -> Tensor (512,)
sample_w(G, n, psi, seed, device, w_mean) -> Tensor (n, 512)
generate(G, w_batch, resolution)    -> Tensor (B, 3, H, W) in [-1, 1]
"""

import os
import sys

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_stylegan(config: dict, device: torch.device | str = 'cpu') -> nn.Module:
    """Load the StyleGAN2 generator from the pickle path in *config*.

    Sets up pure-PyTorch reference ops (no CUDA JIT compilation required) and
    returns G_ema on *device* in eval mode with gradients disabled.

    Parameters
    ----------
    config:
        Parsed YAML config dict (must have paths.stylegan_code,
        paths.repo_root, paths.stylegan_model).
    device:
        Torch device string or object.

    Returns
    -------
    G : nn.Module
        StyleGAN2 generator (G_ema), eval, frozen.
    """
    stylegan_code = config['paths']['stylegan_code']
    sys.path.insert(0, stylegan_code)

    import dnnlib
    import legacy

    # Force pure-PyTorch reference implementations (avoids CUDA toolkit
    # version mismatch when JIT-compiling custom ops).
    import torch_utils.ops.bias_act as _bias_act_mod
    import torch_utils.ops.upfirdn2d as _upfirdn2d_mod
    _bias_act_mod._init = lambda: False
    _upfirdn2d_mod._init = lambda: False

    model_path = os.path.join(
        config['paths']['repo_root'],
        config['paths']['stylegan_model'],
    )
    print(f"Loading StyleGAN2 from {model_path}")
    with dnnlib.util.open_url(model_path) as f:
        data = legacy.load_network_pkl(f)
    G = data['G_ema'].to(device).eval()
    G.requires_grad_(False)
    print(f"  Resolution: {G.img_resolution}, W dim: {G.w_dim}, num_ws: {G.num_ws}")
    return G


# ---------------------------------------------------------------------------
# W sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_w_mean(
    G: nn.Module,
    device: torch.device | str,
    n_samples: int = 10_000,
    batch_size: int = 64,
) -> torch.Tensor:
    """Compute the mean W vector by averaging many Z → W mappings.

    Used as the anchor for the truncation trick:
        W_trunc = W_mean + psi * (W - W_mean)

    Parameters
    ----------
    G:
        Loaded StyleGAN2 generator (output of load_stylegan).
    device:
        Device to run inference on.
    n_samples:
        Number of random Z samples to average over.
    batch_size:
        Batch size for G.mapping calls.

    Returns
    -------
    w_mean : Tensor (w_dim,)
    """
    w_sum = torch.zeros(G.w_dim, device=device)
    count = 0
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        bs = end - start
        z = torch.randn(bs, G.z_dim, device=device)
        w = G.mapping(z, c=None)  # (bs, num_ws, w_dim)
        w_sum += w[:, 0, :].sum(dim=0)  # use first W (all identical in W+ = W)
        count += bs
    return w_sum / count


@torch.no_grad()
def sample_w(
    G: nn.Module,
    n: int,
    psi: float = 1.0,
    seed: int | None = None,
    device: torch.device | str = 'cpu',
    w_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample *n* W vectors, optionally applying truncation.

    Parameters
    ----------
    G:
        Loaded StyleGAN2 generator.
    n:
        Number of W vectors to sample.
    psi:
        Truncation psi. 1.0 = no truncation (full diversity).
        0.0 = all samples equal w_mean (no diversity).
        Typical values: 0.5–0.8 for realistic but diverse faces.
    seed:
        Optional random seed for reproducibility.
    device:
        Device for computation.
    w_mean:
        Precomputed mean W tensor (w_dim,).  If None and psi < 1.0,
        compute_w_mean() is called automatically (slow for first call).

    Returns
    -------
    w : Tensor (n, w_dim)
    """
    if seed is not None:
        torch.manual_seed(seed)

    z = torch.randn(n, G.z_dim, device=device)
    w = G.mapping(z, c=None)[:, 0, :]  # (n, w_dim) — first layer of W

    if psi != 1.0:
        if w_mean is None:
            print("  [sample_w] Computing W mean for truncation (first call)…")
            w_mean = compute_w_mean(G, device)
        w_mean = w_mean.to(device)
        w = w_mean + psi * (w - w_mean)

    return w  # (n, w_dim)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    G: nn.Module,
    w_batch: torch.Tensor,
    resolution: int | None = None,
) -> torch.Tensor:
    """Generate images from W vectors at full 2048×2048, then downsample.

    Always runs the full StyleGAN2 synthesis network to produce correct,
    high-quality output.  If *resolution* is specified and smaller than
    G.img_resolution, the result is bilinearly downsampled to that size.

    This function is decorated with @torch.no_grad() and is suitable for
    inference and dataset generation.  For training loops where gradients
    must flow back through G to the encoder, use generate_with_grad() instead.

    Parameters
    ----------
    G:
        Loaded StyleGAN2 generator.
    w_batch:
        Tensor of shape (B, w_dim).  Each W is broadcast to all num_ws layers.
    resolution:
        Output resolution, or None to return the native G.img_resolution.

    Returns
    -------
    images : Tensor (B, 3, H, W) in [-1, 1]
        Where H = W = resolution (or G.img_resolution when resolution=None).
    """
    import torch.nn.functional as F
    w_broadcast = w_batch.unsqueeze(1).repeat(1, G.num_ws, 1)  # (B, num_ws, w_dim)
    img = G.synthesis(w_broadcast, noise_mode='const')          # (B, 3, 2048, 2048)
    if resolution is not None and resolution < G.img_resolution:
        img = F.interpolate(img, size=resolution, mode='bilinear', align_corners=False)
    return img


def generate_with_grad(
    G: nn.Module,
    w_batch: torch.Tensor,
    resolution: int | None = None,
) -> torch.Tensor:
    """Same as generate() but WITHOUT the @torch.no_grad() decorator.

    Use this inside a training loop where you need gradients to flow from the
    reconstruction loss back through G to the encoder W prediction.

    Always runs full 2048×2048 synthesis then downsamples — never truncated
    synthesis.  G must already have requires_grad_(False) set (done by
    load_stylegan); gradients flow through activations but not into G's weights.

    Parameters
    ----------
    G, w_batch, resolution:
        Same as generate().

    Returns
    -------
    images : Tensor (B, 3, H, W) in [-1, 1]
    """
    import torch.nn.functional as F
    w_broadcast = w_batch.unsqueeze(1).repeat(1, G.num_ws, 1)
    img = G.synthesis(w_broadcast, noise_mode='const')
    if resolution is not None and resolution < G.img_resolution:
        img = F.interpolate(img, size=resolution, mode='bilinear', align_corners=False)
    return img
