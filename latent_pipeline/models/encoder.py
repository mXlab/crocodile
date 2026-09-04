"""EmotionEncoder: VGG-style CNN that maps face images to StyleGAN2 W-space.

Input:  (B, 3, 256, 256) in [-1, 1]
Output: (B, 512) unbounded W vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionEncoder(nn.Module):
    """5-block VGG-style encoder with spatial attention."""

    def __init__(self, channels=(32, 64, 128, 256, 512), w_dim=512, dropout=0.3):
        super().__init__()
        self.w_dim = w_dim

        # VGG-style feature blocks: Conv-BN-ReLU-Conv-BN-ReLU-MaxPool
        blocks = []
        in_ch = 3
        for out_ch in channels:
            blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        # After 5 MaxPools: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        # Output: (B, 512, 8, 8)

        # Spatial attention: single-channel attention map
        self.attention = nn.Conv2d(channels[-1], 1, kernel_size=1)

        # W regressor
        self.regressor = nn.Sequential(
            nn.Linear(channels[-1], w_dim),
            nn.LayerNorm(w_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(w_dim, w_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 256, 256) images in [-1, 1]
        Returns:
            w: (B, 512) W vectors
        """
        feat = self.features(x)  # (B, 512, 8, 8)

        # Spatial attention
        attn = self.attention(feat)  # (B, 1, 8, 8)
        B, C, H, W = feat.shape
        attn = attn.view(B, 1, H * W)  # (B, 1, 64)
        attn = F.softmax(attn, dim=2)  # softmax over spatial positions
        attn = attn.view(B, 1, H, W)  # (B, 1, 8, 8)

        # Weighted sum over spatial dimensions
        weighted = (feat * attn).sum(dim=(2, 3))  # (B, 512)

        # Regress to W
        w = self.regressor(weighted)  # (B, 512)
        return w


# Generation utilities have moved to models/stylegan.py.
# Import from there: from models.stylegan import generate, generate_with_grad
