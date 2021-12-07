from enum import Enum
from abc import ABC, abstractmethod
import torch.nn.functional as F
from dataclasses import dataclass
import FastGAN.lpips as lpips
import torch
from simple_parsing.helpers import Serializable
from simple_parsing.helpers.serialization import encode, register_decoding_fn


class LossType(Enum):
    EUCLIDEAN = "euclidean"
    PERCEPTUAL = "perceptual"


@dataclass
class PerceptualLossParams:
    mse_coeff: float = 0.2
    perceptual_model: str = 'net-lin'
    perceptual_net: str = 'vgg'


@dataclass
class LossParams(Serializable):
    loss: LossType = LossType.EUCLIDEAN
    perceptual_options: PerceptualLossParams = PerceptualLossParams()


@encode.register
def encode_loss_type(obj: LossType) -> str:
    """ We choose to encode a tensor as a list, for instance """
    return obj.name


def decode_loss_type(name: str) -> LossType:
    return LossType[name]


register_decoding_fn(LossType, decode_loss_type)


class Loss(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass


class PerceptualLoss(Loss):
    def __init__(self, args: PerceptualLossParams = PerceptualLossParams()):
        self.args = args
        self.percept = lpips.PerceptualLoss(model=args.perceptual_model, net=args.perceptual_net, use_gpu=True)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.percept(F.avg_pool2d(x, 2, 2), F.avg_pool2d(y, 2, 2)).sum() + self.args.mse_coeff*F.mse_loss(x, y)


class EuclideanLoss(Loss):
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, y)


def load_loss(loss_type: LossType = LossType.EUCLIDEAN, args: LossParams = LossParams()):
    if loss_type == LossType.EUCLIDEAN:
        return EuclideanLoss()
    elif loss_type == LossType.PERCEPTUAL:
        return PerceptualLoss(args.perceptual_options)
