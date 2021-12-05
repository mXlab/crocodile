from abc import ABC, abstractmethod
import torch.nn as nn


class Encoder(ABC, nn.Module):
    def __init__(self, params=None):
        self.params = params
        self.network = None

    @abstractmethod
    def build(self, input_dim: int, output_dim: int, device=None):
        pass
