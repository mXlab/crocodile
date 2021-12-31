from enum import Enum
from .efficientnet import EfficientNetOptions, load_efficientnet
from .regnet import RegNetOptions, load_regnet
from .mlp import MLPParams, load_mlp
from .vgg import VGGOptions, load_vgg
from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from simple_parsing.helpers.serialization import encode, register_decoding_fn
import torch
import torch.nn as nn

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


class Encoder(nn.Module):
    def __init__(self, num_channels: int, seq_length: int, output_dim: int, options: EncoderParams = EncoderParams()):
        super().__init__()
        self.options = options
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.output_dim = output_dim

        if options.encoder == EncoderType.MLP:
            self.network = load_mlp(num_channels, seq_length, output_dim, options.mlp_options)
        elif options.encoder == EncoderType.VGG:
            self.network = load_vgg(num_channels, seq_length, output_dim, options.vgg_options)
        elif options.encoder == EncoderType.REGNET:
            self.network = load_regnet(num_channels, seq_length, output_dim, options.regnet_options)
        elif options.encoder == EncoderType.EFFICIENTNET:
            self.network = load_efficientnet(num_channels, seq_length, output_dim, options.regnet_options)

    def forward(self, x):
        return self.network(x)

    def save(self, filename):
        torch.save({
            'state_dict': self.state_dict(), 
            'options': self.options, 
            'num_channels': self.num_channels, 
            'seq_length': self.seq_length,
            'output_dim': self.output_dim
            }, filename)

    @staticmethod
    def load(filename):
        checkpoint = torch.load(filename)
        encoder = Encoder(checkpoint["num_channels"], checkpoint["seq_length"], checkpoint["output_dim"], checkpoint=["options"])
        encoder.load_state_dict(checkpoint["state_dict"])
        return encoder


@encode.register
def encode_encoder_type(obj: EncoderType) -> str:
    """ We choose to encode a tensor as a list, for instance """
    return obj.name


def decode_encoder_type(name: str) -> EncoderType:
    return EncoderType[name]


register_decoding_fn(EncoderType, decode_encoder_type)
