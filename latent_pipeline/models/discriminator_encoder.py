"""DiscriminatorEncoder: an image -> W-space encoder built from the trained
StyleGAN2 discriminator's own convolutional trunk, instead of EmotionEncoder's
from-scratch VGG trunk.

Motivation
----------
G and D are architectural mirrors (G: code -> image via upsampling, D: image
-> scalar via downsampling), and D was adversarially trained on the exact
same face distribution as G -- including the Diverse/ pool's deliberately
extreme expressions (closed eyes, bared teeth) that EmotionEncoder's training
set undersamples relative to the mostly-neutral biodata sessions. So D's
trunk starts out already sensitive to content EmotionEncoder currently
regresses toward the mean on. See PIPELINE.md session notes (2026-09-11).

Construction
------------
Reuses D's residual blocks from `entry_resolution` down to its 4x4 epilogue
(discarding the higher-resolution blocks an entry_resolution=256 input never
reaches), and replaces only the epilogue's final real/fake scalar head with a
freshly-initialised head regressing to a w_dim-dim W vector. Because
`entry_resolution` isn't D's native top resolution, it has no `fromrgb` of
its own there (StyleGAN2's resnet-architecture discriminator only converts
RGB -> features once, at its top block) -- a new `fromrgb`-equivalent conv is
added to fill that gap, sized to match whatever the entry block expects.

Training staging (see scripts/stage2a_discriminator_init.py)
--------------------------------------------------------------
Two new params (`fromrgb`, `head`) start randomly initialised; everything
else is pretrained. Fine-tuning all of it jointly from step one lets the
random head's large early gradients distort the pretrained trunk -- LP-FT
(Kumar et al., ICLR 2022) shows this hurts worst on exactly the
out-of-distribution portion of the data, which for us is the rare extreme-
expression frames this whole detour is about. So: freeze the whole
pretrained trunk, train only the new params first (`new_parameters()`), then
progressively unfreeze pretrained groups deepest-first
(`unfreeze_order()` / `pretrained_group_parameters(name)`) -- FreezeD (Mo et
al., 2020) found D's deeper/task-specific layers need the most adaptation
and its shallower/generic ones the least, so shallower groups should get a
smaller LR once unfrozen than the most-recently-unfrozen deep ones.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from models.stylegan import load_discriminator


class DiscriminatorEncoder(nn.Module):
    """Image (B, 3, entry_resolution, entry_resolution) in [-1, 1] -> W (B, w_dim).

    Same forward signature as EmotionEncoder, so it's a drop-in replacement
    anywhere an encoder is constructed -- but unlike EmotionEncoder, the
    input resolution is fixed at construction time (the transplanted block
    chain must land exactly on D's 4x4 epilogue).
    """

    def __init__(self, D, entry_resolution=256, w_dim=512, img_channels=3):
        super().__init__()
        # Imported here (not at module scope) because these classes only
        # become importable after load_discriminator has inserted
        # stylegan_Autolume onto sys.path -- see models/stylegan.py.
        from training.networks_stylegan2 import Conv2dLayer, FullyConnectedLayer

        self.entry_resolution = entry_resolution
        self.w_dim = w_dim

        all_res = list(D.block_resolutions)  # descending, e.g. [2048, ..., 8]
        trunk_res = [r for r in all_res if r <= entry_resolution]
        if not trunk_res or trunk_res[0] != entry_resolution:
            raise ValueError(
                f"entry_resolution={entry_resolution} must exactly equal one of "
                f"D's block resolutions {all_res} (the block chain must land "
                f"exactly on D's fixed 4x4 epilogue)."
            )
        # Shallowest (closest to pixels) first, e.g. [256, 128, 64, 32, 16, 8].
        self.block_res_order = trunk_res

        # New entry point, replacing the fromrgb the discarded higher-res
        # blocks would otherwise have provided. Sized to match what the
        # shallowest kept block expects as input.
        entry_block = getattr(D, f'b{trunk_res[0]}')
        tmp_channels = entry_block.conv0.in_channels
        self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1,
                                    activation='lrelu')

        # Transplanted pretrained blocks (same module objects as D's -- keeps
        # trained weights without a state_dict round-trip).
        self.blocks = nn.ModuleDict({f'b{r}': getattr(D, f'b{r}') for r in trunk_res})

        # Transplanted epilogue, minus its final scalar head.
        epilogue = D.b4
        self.mbstd = epilogue.mbstd
        self.epilogue_conv = epilogue.conv
        self.epilogue_fc = epilogue.fc
        feat_dim = epilogue.fc.out_features

        # Brand-new head: D's original `out` mapped this to a single real/
        # fake logit; this maps it to a W vector instead.
        self.head = FullyConnectedLayer(feat_dim, w_dim, activation='linear')

    def forward(self, x):
        feat = self.fromrgb(x)
        img = None
        for r in self.block_res_order:
            feat, img = self.blocks[f'b{r}'](feat, img)
        if self.mbstd is not None:
            feat = self.mbstd(feat)
        feat = self.epilogue_conv(feat)
        feat = self.epilogue_fc(feat.flatten(1))
        return self.head(feat)

    # -- Staged freeze/unfreeze -------------------------------------------

    def new_parameters(self):
        """Freshly-initialised parameters: the entry adapter and the W head."""
        return list(self.fromrgb.parameters()) + list(self.head.parameters())

    def pretrained_group_names(self):
        """Pretrained parameter groups, deepest (closest to the new head)
        first -- also the intended progressive-unfreeze order."""
        return ['epilogue'] + [f'b{r}' for r in reversed(self.block_res_order)]

    def pretrained_group_parameters(self, name):
        if name == 'epilogue':
            return list(self.epilogue_conv.parameters()) + list(self.epilogue_fc.parameters())
        return list(self.blocks[name].parameters())

    def freeze_all_pretrained(self):
        for name in self.pretrained_group_names():
            for p in self.pretrained_group_parameters(name):
                p.requires_grad_(False)

    def unfreeze_pretrained_group(self, name):
        for p in self.pretrained_group_parameters(name):
            p.requires_grad_(True)


def build_discriminator_encoder(config, device, entry_resolution=None, w_dim=None):
    """Load D from the configured StyleGAN2 checkpoint and wrap it as a
    DiscriminatorEncoder, with the pretrained trunk frozen (caller drives
    unfreezing via the staged schedule in stage2a_discriminator_init.py)."""
    dc = config.get('discriminator_encoder', {})
    entry_resolution = entry_resolution or dc.get('entry_resolution', 256)
    w_dim = w_dim or dc.get('w_dim', 512)

    D = load_discriminator(config, device)
    encoder = DiscriminatorEncoder(D, entry_resolution=entry_resolution, w_dim=w_dim).to(device)
    encoder.freeze_all_pretrained()
    n_new = sum(p.numel() for p in encoder.new_parameters())
    n_pretrained = sum(p.numel() for name in encoder.pretrained_group_names()
                       for p in encoder.pretrained_group_parameters(name))
    print(f"DiscriminatorEncoder: entry_resolution={entry_resolution}, "
          f"blocks={encoder.block_res_order}, "
          f"new_params={n_new:,}, pretrained_params={n_pretrained:,} (frozen)")
    return encoder
