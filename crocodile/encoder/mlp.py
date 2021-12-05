from torch import nn
from dataclasses import dataclass
from .encoder import Encoder


@dataclass
class MLPParams:
    num_hidden = 100


class MLP(Encoder):
    def forward(self, x):
        return self.network(x.view(len(x), -1))

    def build(self, input_dim: int, output_dim: int, device=None):
        self.network = nn.Sequential(nn.Linear(input_dim, self.args.num_hidden), nn.ReLU(
        ), nn.Linear(self.args.num_hidden, output_dim))

        return self
