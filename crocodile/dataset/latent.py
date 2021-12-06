import torch


class LatentDataset:
    def __init__(self, n, dim=None, init_func=None):
        if init_func is None:
            self.latent = torch.zeros(n, dim)
        self.latent = init_func(n).cpu()

    def __getitem__(self, index):
        return self.latent[index]

    def __setitem__(self, index, value):
        self.latent[index] = value

    def save(self):
        raise NotImplementedError
