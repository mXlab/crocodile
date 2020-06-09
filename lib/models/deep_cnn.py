from . import Discriminator
from torch import nn

NUM_FILTERS = (8, 16, 32, 64)

class SmallGenerator(nn.Module):
    def __init__(self, num_latent, resolution, num_filters=NUM_FILTERS, batch_norm=True):
        super(Generator, self).__init__()
        self.resolution = resolution
        self.num_filters = num_filters

        network = []
        for i in range(num_layers):
            if batch_norm:
                network.append(nn.BatchNorm2d(num_filters/))


        self.input = nn.Linear(num_latent, num_filters*(resolution/(num_layers*2))**2)
        self.network = nn.Sequential(nn.BatchNorm2d(num_filters),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(num_filters, num_filters//2, 4, stride=2, padding=1), # 8x8 -> 16x16
                                     nn.BatchNorm2d(num_filters//2),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(num_filters//2, num_filters//4, 4, stride=2, padding=1), # 16x16 -> 32x32
                                     nn.BatchNorm2d(num_filters//4),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(num_filters//4, num_filters//8, 4, stride=2, padding=1), # 32x32 -> 64x64
                                     nn.BatchNorm2d(num_filters//8),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(num_filters//8, 3, 3, stride=1, padding=1),
                                     nn.Tanh())

    def forward(self, x):
        x = self.input(x).view(-1, self.num_filters, self.RESOLUTION//8, self.RESOLUTION//8)
        x = self.network(x)
        return x

class SmallDiscriminator(Discriminator):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(spectral_norm(nn.Conv2d(3, NUM_FILTERS//8, 3, stride=1, padding=1)),
                                     nn.LeakyReLU(0.1, True),
                                     spectral_norm(nn.Conv2d(NUM_FILTERS//8, NUM_FILTERS//4, 4, stride=2, padding=1)), # 64x64 -> 32x32
                                     nn.LeakyReLU(0.1, True),
                                     spectral_norm(nn.Conv2d(NUM_FILTERS//4, NUM_FILTERS//4, 3, stride=1, padding=1)),
                                     nn.LeakyReLU(0.1, True),
                                     spectral_norm(nn.Conv2d(NUM_FILTERS//4, NUM_FILTERS//2, 4, stride=2, padding=1)), # 32x32 -> 16x16
                                     nn.LeakyReLU(0.1, True),
                                     spectral_norm(nn.Conv2d(NUM_FILTERS//2, NUM_FILTERS//2, 3, stride=1, padding=1)),
                                     nn.LeakyReLU(0.1, True),
                                     spectral_norm(nn.Conv2d(NUM_FILTERS//2, NUM_FILTERS, 4, stride=2, padding=1)), # 16x16 -> 8x8
                                     nn.LeakyReLU(0.1, True),
                                     spectral_norm(nn.Conv2d(NUM_FILTERS, NUM_FILTERS, 3, stride=1, padding=1)),
                                     nn.LeakyReLU(0.1, True))
        self.output = spectral_norm(nn.Linear(NUM_FILTERS*(RESOLUTION//8)**2, 1))

    def forward(self, x):
        x = self.network(x).view(-1, NUM_FILTERS*(RESOLUTION//8)**2)
        x = self.output(x)
        return x
