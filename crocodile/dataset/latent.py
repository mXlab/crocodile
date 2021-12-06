import torch


class LatentDataset:
    def __init__(self, n, dim):
        self.latent = torch.zeros(n, dim)

    def __getitem__(self, index):
        return self.latent[index]

    def __setitem__(self, index, value):
        self.latent[index] = value

    def save(self):
        raise NotImplementedError
