from dataclasses import dataclass
from abc import ABC, abstractmethod
from torch import nn


@dataclass
class GeneratorConfig:
    pass


class Generator(ABC, nn.Module):
    @property
    @abstractmethod
    def latent_dim(self):
        ...

    @abstractmethod
    def generate(self):
        ...
