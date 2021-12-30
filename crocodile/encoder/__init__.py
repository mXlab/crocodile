from enum import Enum
from .efficientnet import EfficientNetOptions, EfficientNet
from .regnet import RegNetOptions, RegNet
from .mlp import MLP, MLPParams
from .vgg import VGG, VGGOptions
from dataclasses import dataclass
from .encoder import Encoder
from simple_parsing.helpers import Serializable
from simple_parsing.helpers.serialization import encode, register_decoding_fn


class EncoderType(Enum):
    MLP = "mlp"
    VGG = "vgg"
    EFFICIENTNET = "efficientnet"
    REGNET = "regnet"


@dataclass
class EncoderParams(Serializable):
    encoder: EncoderType = EncoderType.MLP
    mlp_options: MLPParams = MLPParams()
    vgg_options: VGGOptions = VGGOptions()
    regnet_options: RegNetOptions = RegNetOptions()
    efficientnet_options: EfficientNetOptions = EfficientNetOptions()


def load_encoder(params: EncoderParams) -> Encoder:
    if params.encoder == EncoderType.MLP:
        return MLP(params.mlp_options)
    elif params.encoder == EncoderType.VGG:
        return VGG(params.vgg_options)
    elif params.encoder == EncoderType.REGNET:
        return RegNet(params.regnet_options)
    elif params.encoder == EncoderType.EFFICIENTNET:
        return EfficientNet(params.regnet_options)


@encode.register
def encode_encoder_type(obj: EncoderType) -> str:
    """ We choose to encode a tensor as a list, for instance """
    return obj.name


def decode_encoder_type(name: str) -> EncoderType:
    return EncoderType[name]


register_decoding_fn(EncoderType, decode_encoder_type)
