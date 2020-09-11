import torch
from torch import nn
import math
from . import Discriminator, Generator


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, kernel_size=3,
                 padding=1, batchnorm=True, activation=nn.ReLU, sampling=None, spectral_norm=True):
        super(ResBlock, self).__init__()
        hidden_channels = out_channels if hidden_channels is None else hidden_channels

        def apply_spectralnorm(m):
            if spectral_norm:
                m = torch.nn.utils.spectral_norm(m)
            return m

        def apply_batchnorm(in_channels):
            layer = nn.Identity()
            if batchnorm:
                layer = nn.BatchNorm2d(in_channels)
            return layer

        self.residual = [apply_batchnorm(in_channels), activation()]
        if sampling == "max_unpool":
            self.residual.append(nn.MaxUnpool2d(2, stride=2, padding=0))
        elif sampling == "up_nearest":
            self.residual.append(nn.UpsamplingNearest2d(scale_factor=2))

        self.residual += [apply_spectralnorm(nn.Conv2d(in_channels, hidden_channels,
                                                        kernel_size, padding=padding)),
                     apply_batchnorm(hidden_channels), activation(),
                     apply_spectralnorm(nn.Conv2d(hidden_channels, out_channels,
                                                  kernel_size, padding=padding))]

        if sampling == "avg_pool":
            self.residual.append(nn.AvgPool2d(2, stride=2, padding=0))

        self.residual = nn.Sequential(*self.residual)
        self.residual.apply(self.residual_initializer)

        self.shortcut = None
        if (in_channels != out_channels) or (sampling is not None):
            self.shortcut = []
            if sampling == "max_unpool":
                self.shortcut.append(nn.MaxUnpool2d(2, stride=2, padding=0))
            elif sampling == "up_nearest":
                self.shortcut.append(nn.UpsamplingNearest2d(scale_factor=2))

            self.shortcut.append(apply_spectralnorm(nn.Conv2d(in_channels, out_channels,
                                                              1, padding=0)))

            if sampling == "avg_pool":
                self.shortcut.append(nn.AvgPool2d(2, stride=2, padding=0))

            self.shortcut = nn.Sequential(*self.shortcut)
            self.shortcut.apply(self.shortcut_initializer)

    def residual_initializer(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2))

    def shortcut_initializer(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        out = self.residual(x)
        if self.shortcut is not None:
            x = self.shortcut(x)
        #print(out.size(), x.size())
        return out + x


class ResNetGenerator(Generator):
    def __init__(self, num_latent, resolution, num_filters, num_layers=5, batch_norm=True,
                 activation=nn.ReLU, spectral_norm=False):
        super(ResNetGenerator, self).__init__(num_latent)
        self.num_filters = num_filters
        self.resolution = resolution
        self.num_layers = num_layers

        self.input = nn.Linear(num_latent, num_filters*(resolution//(2**num_layers))**2)

        _nf = num_filters
        self.network = [ResBlock(_nf, _nf, batchnorm=batch_norm, sampling="up_nearest",
                                 activation=activation, spectral_norm=spectral_norm)]
        for i in range(num_layers-1):
            self.network.append(ResBlock(_nf, _nf//2, batchnorm=batch_norm, sampling="up_nearest",
                                    activation=activation, spectral_norm=spectral_norm))
            _nf = _nf//2

        if batch_norm:
            self.network.append(nn.BatchNorm2d(_nf))
        self.network += [activation(), nn.Conv2d(_nf, 3, 3, stride=1, padding=1), nn.Tanh()]
        self.network = nn.Sequential(*self.network)

        self.apply(self.initializer)

    def initializer(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.input(x).view(-1, self.num_filters, self.resolution//(2**self.num_layers),
                                self.resolution//(2**self.num_layers))
        x = self.network(x)
        return x

class ResNetDiscriminator(Discriminator):
    def __init__(self, resolution, num_filters, num_layers=5, spectral_norm=True, activation=nn.ReLU):
        super(ResNetDiscriminator, self).__init__()

        def apply_spectralnorm(m):
            if spectral_norm:
                m = torch.nn.utils.spectral_norm(m)
            return m

        _nf = num_filters//(2**num_layers)
        self.network = [ResBlock(3, _nf, batchnorm=False, sampling="avg_pool",
                                 activation=activation, spectral_norm=spectral_norm)]
        for i in range(num_layers-1):
            self.network.append(ResBlock(_nf, _nf*2, batchnorm=False, sampling="avg_pool",
                                     activation=activation, spectral_norm=spectral_norm))
            _nf = _nf*2

        self.network += [ResBlock(_nf, _nf, batchnorm=False, sampling=None,
                                 activation=activation, spectral_norm=spectral_norm),
                         activation()]

        self.network = nn.Sequential(*self.network)

        self.output = apply_spectralnorm(nn.Linear(_nf, 1))

        self.apply(self.initializer)

    def initializer(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.network(x).sum(dim=(2,3))
        x = self.output(x)
        return x
