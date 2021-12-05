from enum import Enum
from .mlp import MLP, MLPParams
from dataclasses import dataclass
from .encoder import Encoder


class EncoderType(Enum):
    MLP = "mlp"


@dataclass
class EncoderParams:
    encoder: EncoderType = EncoderType.MLP
    mlp_options: MLPParams = MLPParams()


def load_encoder(params: EncoderParams) -> Encoder:
    if params.encoder == EncoderType.MLP:
        return MLP(params.mlp_options)
