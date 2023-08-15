from typing import Tuple
from dataclasses import dataclass
from torch.optim import Adam


@dataclass
class OptimizerConfig:
    lr: float = 1e-2


@dataclass
class AdamConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)


def load_optimizer(params, config: OptimizerConfig):
    match config:
        case AdamConfig():
            return Adam(params, lr=config.lr, betas=config.betas)
        case _:
            raise NotImplementedError()
