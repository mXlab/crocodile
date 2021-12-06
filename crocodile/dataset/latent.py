import torch
from typing import Optional, Callable, List
from pathlib import Path


class LatentDataset:
    def __init__(self, n: int, dim: Optional[int] = None, init_func: Optional[Callable[[int], torch.Tensor]] = None):
        if init_func is None:
            self.latent = torch.zeros(n, dim)
        else:
            self.latent = init_func(n).cpu()

    def parameters(self) -> List[torch.Tensor]:
        return [self.latent]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.latent[index]

    def __setitem__(self, index: int, value: torch.Tensor):
        self.latent[index] = value

    def save(self, filename: Path):
        torch.save(self.latent, filename)
