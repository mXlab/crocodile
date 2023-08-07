from dataclasses import dataclass
from typing import Tuple

from torch import optim


@dataclass
class OptimizerConfig:
    lr: float


@dataclass
class AdamConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


def load_optimizer(params, config: OptimizerConfig):
    match config:
        case AdamConfig(lr=lr, betas=betas, eps=eps):
            return optim.Adam(params, lr=lr, betas=betas, eps=eps)
        case _:
            raise ValueError("Optimizer not implemented.")
