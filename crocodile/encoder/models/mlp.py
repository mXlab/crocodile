from torch import nn
from dataclasses import dataclass


class MLP(nn.Module):
    @dataclass
    class Params:
        num_hidden = 100

    def __init__(self, input_dim: int, output_dim: int, params: Params = Params()):
        self.network = nn.Sequential(nn.Linear(input_dim, params.num_hidden), nn.ReLU(
        ), nn.Linear(params.num_hidden, output_dim))

    def forward(self, x):
        return self.network(x.view(len(x), -1))

    @staticmethod
    def init(input_dim: int, output_dim: int, params: Params = Params(), device=None):
        return MLP(input_dim, output_dim, params).to(device)
