from . import Discriminator, Generator
from torch import nn
import torch


class SmallGenerator(Generator):
    def __init__(self, num_latent, resolution, num_filters, num_layers=3, batch_norm=True, spectral_norm=True):
        super(SmallGenerator, self).__init__(num_latent)
        self.resolution = resolution
        self.num_filters = num_filters
        self.num_layers = num_layers

        network = []
        _nf = num_filters
        for i in range(num_layers):
            if batch_norm:
                network.append(nn.BatchNorm2d(_nf))
            layer = nn.ConvTranspose2d(_nf, _nf//2, 4, stride=2, padding=1)
            if spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            network += [nn.ReLU(True), layer]
            _nf = _nf//2
        if batch_norm:
            network.append(nn.BatchNorm2d(_nf))
        layer = nn.ConvTranspose2d(_nf, 3, 3, stride=1, padding=1)
        if spectral_norm:
            layer = nn.utils.spectral_norm(layer)
        network += [nn.ReLU(True), layer, nn.Tanh()]

        self.input = nn.Linear(num_latent, num_filters*(resolution//(2**num_layers))**2)
        if spectral_norm:
            self.input = nn.utils.spectral_norm(self.input)
        self.network = nn.Sequential(*network)

    def forward(self, x):
        x = self.input(x).view(-1, self.num_filters, self.resolution//(2**self.num_layers),
                                self.resolution//(2**self.num_layers))
        x = self.network(x)
        return x


class SmallDiscriminator(Discriminator):
    def __init__(self, resolution, num_filters, num_layers=3, spectral_norm=True):
        super(SmallDiscriminator, self).__init__()
        self.num_filters = num_filters
        self.resolution = resolution
        self.num_layers = num_layers

        network = []
        _nf = num_filters//(2**num_layers)
        layer = nn.Conv2d(3, _nf, 3, stride=1, padding=1)
        if spectral_norm:
            layer = nn.utils.spectral_norm(layer)
        network += [layer, nn.LeakyReLU(0.1, True)]
        for i in range(num_layers):
            layer = nn.Conv2d(_nf, _nf*2, 4, stride=2, padding=1)
            if spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            network += [layer, nn.LeakyReLU(0.1, True)]
            _nf = _nf*2

        self.network = nn.Sequential(*network)
        self.output = nn.Linear(num_filters*(resolution//(2**num_layers))**2, 1)
        if spectral_norm:
            self.output = nn.utils.spectral_norm(self.output)

    def forward(self, x):
        x = self.network(x).view(-1, self.num_filters*(self.resolution//(2**self.num_layers))**2)
        x = self.output(x)
        return x


class ConditionalSmallDiscriminator(Discriminator):
    def __init__(self, resolution, num_cat, num_filters, num_layers=3, spectral_norm=True):
        super(ConditionalSmallDiscriminator, self).__init__()
        self.num_filters = num_filters
        self.resolution = resolution
        self.num_layers = num_layers

        network = []
        _nf = num_filters//(2**num_layers)
        layer = nn.Conv2d(3, _nf, 3, stride=1, padding=1)
        if spectral_norm:
            layer = nn.utils.spectral_norm(layer)
        network += [layer, nn.LeakyReLU(0.1, True)]
        for i in range(num_layers):
            layer = nn.Conv2d(_nf, _nf*2, 4, stride=2, padding=1)
            if spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            network += [layer, nn.LeakyReLU(0.1, True)]
            _nf = _nf*2

        self.network = nn.Sequential(*network)

        normalize = lambda x: x
        if spectral_norm:
            normalize = nn.utils.spectral_norm

        self.output = nn.Sequential(normalize(nn.Linear(num_filters*(resolution//(2**num_layers))**2 + num_cat, num_filters*(resolution//(2**num_layers))**2)),
                                                nn.LeakyReLU(0.1, True),
                                                normalize(nn.Linear(num_filters*(resolution//(2**num_layers))**2, 1)))

    def forward(self, x, y):
        x = self.network(x).view(-1, self.num_filters*(self.resolution//(2**self.num_layers))**2)
        x = torch.cat([x, y], -1)
        x = self.output(x)
        return x

class ConditionalSmallGenerator(nn.Module):
    def __init__(self, num_latent, num_cat, resolution, num_filters, num_layers=3, batch_norm=True, spectral_norm=True):
        super(ConditionalSmallGenerator, self).__init__()
        self.resolution = resolution
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_cat = num_cat

        network = []
        _nf = num_filters
        for i in range(num_layers):
            if batch_norm:
                network.append(nn.BatchNorm2d(_nf))
            layer = nn.ConvTranspose2d(_nf, _nf//2, 4, stride=2, padding=1)
            if spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            network += [nn.ReLU(True), layer]
            _nf = _nf//2
        if batch_norm:
            network.append(nn.BatchNorm2d(_nf))
        layer = nn.ConvTranspose2d(_nf, 3, 3, stride=1, padding=1)
        if spectral_norm:
            layer = nn.utils.spectral_norm(layer)
        network += [nn.ReLU(True), layer, nn.Tanh()]

        self.input = nn.Linear(2*num_latent, num_filters*(resolution//(2**num_layers))**2)
        self.pre_input = nn.Sequential(nn.Linear(num_cat, num_latent), nn.ReLU(),
                                        nn.Linear(num_latent, num_latent))
        if spectral_norm:
            self.input = nn.utils.spectral_norm(self.input)
        self.network = nn.Sequential(*network)

    def forward(self, x, y):
        y = self.pre_input(y)
        x = torch.cat([x, y], -1)
        x = self.input(x).view(-1, self.num_filters, self.resolution//(2**self.num_layers),
                                self.resolution//(2**self.num_layers))
        x = self.network(x)
        return x
