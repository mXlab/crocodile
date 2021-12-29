from torch import nn
from dataclasses import dataclass
from .encoder import Encoder


@dataclass
class MLPParams:
    num_hidden: int = 100
    num_layers: int = 1


class MLP(Encoder):
    def __init__(self, options: MLPParams = MLPParams()):
        super().__init__()
        self.options = options

    def forward(self, x):
        return self.network(x.view(len(x), -1))

    def build(self, num_channels: int, seq_length: int, output_dim: int, device=None):
        inpût_dim = num_channels*seq_length
        list_layers = []
        for i in range(self.options.num_layers):
            list_layers += [nn.Linear(input_dim, self.options.num_hidden), nn.ReLU()]
            input_dim = self.options.num_hidden
        list_layers += [nn.Linear(input_dim, output_dim)]
        self.network = nn.Sequential(*list_layers)

        return self
