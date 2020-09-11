import torch
from torch import nn
import torch.nn.functional as F
from .sngan import ResBlock
from . import Discriminator, Generator

class SelfAttention(nn.Module):
    def __init__(self, in_channels, attention_channels=None, hidden_channels=None, spectral_norm=False):
        super(SelfAttention, self).__init__()

        self.attention_channels = in_channels // 8 if attention_channels is None else attention_channels
        self.hidden_channels = in_channels // 2 if hidden_channels is None else hidden_channels

        def apply_spectralnorm(m):
            if spectral_norm:
                m = torch.nn.utils.spectral_norm(m)
            return m

        self.features_1 = apply_spectralnorm(nn.Conv2d(in_channels, self.attention_channels,
                                                       1, bias=False))
        self.features_2 = nn.Sequential(
                          apply_spectralnorm(nn.Conv2d(in_channels, self.attention_channels,
                                                       1, bias=False)),
                          nn.MaxPool2d(2, stride=2)
                                        )

        self.conv_1 = nn.Sequential(
                        apply_spectralnorm(nn.Conv2d(in_channels, self.hidden_channels,
                                                     1, bias=False)),
                        nn.MaxPool2d(2, stride=2)
                                    )

        self.conv_2 = apply_spectralnrom(nn.Conv2d(self.hidden_channels, self.in_channels, 1))
        self.sigma = nn.Parameter(torch.zeros(1))

        self.apply(self.initializer)

    def initializer(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        batch_size, num_channels, h, w  = x.size()
        features_1 = self.features_1(x).view(len(x), self.attention_channels, -1)
        features_2 = self.features_2(x).view(len(x), self.attention_channels, -1)

        attention = F.softmax(features_2.transpose(1, 2).bmm(features_1), dim=-1)

        output = self.conv_1(x).view(len(x), self.hidden_channels, -1)
        output = output.bmm(attention).view(len(x), self.hidden_channels, h, w)
        output = self.conv_2(output)

        return x + self.sigma*output


class SAGenerator(Generator):
    def __init__(self):
        super(SAGenerator).__init__()
