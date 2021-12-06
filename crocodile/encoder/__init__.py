from enum import Enum
from .mlp import MLP, MLPParams
from dataclasses import dataclass
from .encoder import Encoder
from simple_parsing.helpers import Serializable
from simple_parsing.helpers.serialization import encode, register_decoding_fn


class EncoderType(Enum):
    MLP = "mlp"


@dataclass
class EncoderParams(Serializable):
    encoder: EncoderType = EncoderType.MLP
    mlp_options: MLPParams = MLPParams()


def load_encoder(params: EncoderParams) -> Encoder:
    if params.encoder == EncoderType.MLP:
        return MLP(params.mlp_options)


@encode.register
def encode_encoder_type(obj: EncoderType) -> str:
    """ We choose to encode a tensor as a list, for instance """
    return obj.name


def decode_encoder_type(name: str) -> EncoderType:
    return EncoderType[name]


register_decoding_fn(EncoderType, decode_encoder_type)
