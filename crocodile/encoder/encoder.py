from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class Encoder(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.network = None

    @abstractmethod
    def build(self, input_dim: int, output_dim: int, device=None):
        pass

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)
