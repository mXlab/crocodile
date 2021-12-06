from enum import Enum
from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from simple_parsing.helpers.serialization import encode, register_decoding_fn
import torch.optim as optim
from typing import Optional
import torch


class OptimizerType(Enum):
    SGD = "sgd"
    ADAM = "adam"


@dataclass
class OptimizerArgs(Serializable):
    optimizer: OptimizerType = OptimizerType.SGD
    lr: Optional[float] = None
    noise_scale: Optional[float] = None
    
    def __post_init__(self):
        if self.lr is None:
            if self.optimizer == OptimizerType.SGD:
                self.lr = 1e-2
            elif self.optimizer == OptimizerType.ADAM:
                self.lr = 1e-4


@encode.register
def encode_optimizer_type(obj: OptimizerType) -> str:
    """ We choose to encode a tensor as a list, for instance """
    return obj.name


def decode_optimizer_type(name: str) -> OptimizerType:
    return OptimizerType[name]


register_decoding_fn(OptimizerType, decode_optimizer_type)


def load_optimizer(parameters, args: OptimizerArgs = OptimizerArgs()) -> optim.Optimizer:
    if args.optimizer == OptimizerType.SGD:
        optimizer = optim.SGD(parameters, lr=args.lr)
    elif args.optimizer == OptimizerType.ADAM:
        optimizer = optim.Adam(parameters, lr=args.lr)

    if args.noise_scale is not None:
        optimizer = Langevin(optimizer, noise_scale=args.noise_scale)
    return optimizer


class Langevin(optim.Optimizer):
    def __init__(self, optimizer: optim.Optimizer, noise_scale: float = 1.):
        self.optimizer = optimizer
        self.noise_scale = noise_scale

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad()

    def step(self, closure):
        self.optimizer.step()
        for group in self.optimizer.param_groups:
            for p in group['params']:
                noise = torch.zeros_like(p).normal_()
                p.add_(noise, alpha=self.noise_scale)

