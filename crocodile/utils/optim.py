from enum import Enum
from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from simple_parsing.helpers.serialization import encode, register_decoding_fn
import torch.optim as optim
from typing import Optional


class OptimizerType(Enum):
    SGD = "sgd"
    ADAM = "adam"


@dataclass
class OptimizerArgs(Serializable):
    optimizer: OptimizerType = OptimizerType.SGD
    lr: Optional[int] = None
    
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


def load_optimizer(parameters, args: OptimizerArgs = OptimizerArgs()):
    if args.optimizer == OptimizerType.SGD:
        return optim.SGD(parameters, lr=args.lr)
    elif args.optimizer == OptimizerType.ADAM:
        return optim.Adam(parameters, lr=args.lr)