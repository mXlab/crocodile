import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, num_latent):
        super(Generator, self).__init__()
        self.num_latent = num_latent

    def sample(self, num_samples=1):
        z = torch.zeros(num_samples, self.num_latent).normal_().to(self.device)
        x = self.forward(z)
        return x

    def to(self, device=None):
        self.device = device
        return super().to(device=device)


