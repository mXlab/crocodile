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
    POLYAK = "polyak"


@dataclass
class OptimizerArgs(Serializable):
    optimizer: OptimizerType = OptimizerType.SGD
    lr: Optional[float] = None
    noise_scale: Optional[float] = None
    momentum: float = 0.

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
        optimizer = OptimizerWrapper(
            optim.SGD(parameters, lr=args.lr, momentum=args.momentum))
    elif args.optimizer == OptimizerType.ADAM:
        optimizer = OptimizerWrapper(optim.Adam(parameters, lr=args.lr))
    elif args.optimizer == OptimizerType.POLYAK:
        optimizer = PolyakStep(parameters)

    if args.noise_scale is not None:
        optimizer = Langevin(optimizer, noise_scale=args.noise_scale)
    return optimizer


class OptimizerWrapper:
    def __init__(self, optimizer: optim.Optimizer):
        self.optimizer = optimizer

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad()

    def step(self, closure=None, loss=None):
        return self.optimizer.step(closure=closure)


class Langevin(OptimizerWrapper):
    def __init__(self, optimizer: optim.Optimizer, noise_scale: float = 1.):
        super().__init__(optimizer)
        self.noise_scale = noise_scale

    def step(self, closure=None, loss=None):
        super().step(closure, loss)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                noise = torch.zeros_like(p).normal_()
                p.data.add_(noise, alpha=self.noise_scale)


class PolyakStep(optim.Optimizer):
    def __init__(self, params):
        defaults = dict()
        super().__init__(params, defaults)

    def step(self, closure=None, loss=None):
        if closure is None and loss is None:
            raise ValueError(
                "Please specify either a closure function or pass directly the loss.")
        if closure is not None and loss is not None:
            raise ValueError(
                "Please either specify a closure function or pass directly the loss. But you can't do both.")
        if loss is None:
            loss = closure()

        reduction = loss.numel() > 1
        grad_norm = compute_grad_norm(self.params, reduction)

        step_size = loss / grad_norm
        step_size[grad_norm < self.eps] = 0.

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                p.data = p.data - step_size*d_p

        return loss


def compute_grad_norm(params, reduction=True):
    grad_norm = 0.
    for p in params:
        g = p.grad
        if g is None:
            continue
        _norm = ((g.data)**2).view(len(g), -1).sum(dim=-1, keepdim=True)

        if reduction:
            _norm = _norm.sum()
        grad_norm += _norm

    return grad_norm
