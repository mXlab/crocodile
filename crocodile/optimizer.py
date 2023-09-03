from typing import Tuple
from dataclasses import dataclass
from torch.optim import Adam, SGD


@dataclass
class OptimizerConfig:
    lr: float = 1e-2


@dataclass
class SGDConfig(OptimizerConfig):
    lr: float = 1e-2
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False


@dataclass
class AdamConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False


def load_optimizer(params, config: OptimizerConfig):
    match config:
        case SGDConfig():
            return SGD(
                params,
                lr=config.lr,
                momentum=config.momentum,
                dampening=config.dampening,
                weight_decay=config.weight_decay,
                nesterov=config.nesterov,
            )
        case AdamConfig():
            return Adam(
                params,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                weight_decay=config.weight_decay,
                amsgrad=config.amsgrad,
            )
        case _:
            raise NotImplementedError()
