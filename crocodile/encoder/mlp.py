from torch import nn
from dataclasses import dataclass


@dataclass
class MLPParams:
    num_hidden: int = 100
    num_layers: int = 1


def load_mlp(num_channels: int, seq_length: int, output_dim: int, options: MLPParams = MLPParams()):
    return MLP(num_channels, seq_length, output_dim, options)


class MLP(nn.Module):
    def __init__(self, num_channels: int, seq_length: int, output_dim: int, options: MLPParams = MLPParams()):
        super().__init__()
        self.options = options

        input_dim = num_channels*seq_length
        list_layers = []
        for i in range(self.options.num_layers):
            list_layers += [nn.Linear(input_dim, self.options.num_hidden), nn.ReLU()]
            input_dim = self.options.num_hidden
        list_layers += [nn.Linear(input_dim, output_dim)]
        self.network = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.network(x.view(len(x), -1))
