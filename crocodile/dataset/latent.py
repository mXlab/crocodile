import torch
from typing import Optional, Callable
from pathlib import Path
import torch.nn as nn


class LatentDataset(nn.Module):
    def __init__(self, n: int, dim: Optional[int] = None, init_func: Optional[Callable[[int], torch.Tensor]] = None):
        super().__init__()
        if init_func is None:
            self.latent = torch.zeros(n, dim)
        else:
            self.latent = init_func(n).cpu()

        self.latent = nn.Parameter(self.latent)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.latent[index]

    def __setitem__(self, index: int, value: torch.Tensor):
        self.latent[index] = value

    def save(self, filename: Path):
        torch.save(self.latent, filename)
