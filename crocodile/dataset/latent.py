import torch
from typing import Optional, Callable
from pathlib import Path


class LatentDataset:
    def __init__(self, n: Optional[int] = None, dim: Optional[int] = None, latent: Optional[torch.Tensor] = None):
        super().__init__()
        if latent is None and (n is None or dim is None):
            raise ValueError("If `latent` is None, then you must specify n and dim.")
        elif latent is None:
            self.latent = torch.zeros(n, dim)
        else:
            self.latent = latent

        self.dim = self.latent.size(1)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.latent[index]

    def __setitem__(self, index: int, value: torch.Tensor):
        self.latent[index] = value

    def save(self, filename: Path):
        torch.save(self.latent, filename)

    @staticmethod
    def load(self, filename: Path):
        latent = torch.load(filename)
        return LatentDataset(latent=latent)
