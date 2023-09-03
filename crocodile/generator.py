"""Generator class""" ""
from typing import Protocol
import torch


class Generator(Protocol):
    """Generator class"""

    def generate(self, noise: torch.Tensor) -> torch.Tensor:
        """Generate images from noise"""
        ...

    def noise(self, n: int) -> torch.Tensor:
        """Generate noise"""
        ...
