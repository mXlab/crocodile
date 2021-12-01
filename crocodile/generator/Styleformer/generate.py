# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
from typing import Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--num_frames', default=100, type=int, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    num_frames: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail(
                'Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    start_z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
    end_z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)

    # Generate images.
    for i in range(num_frames):
        alpha = i/(num_frames-1)
        z = (1 - alpha)*start_z + alpha*end_z
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 +
               128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(
            f'{outdir}/{i:04d}.png')


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
