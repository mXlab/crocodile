from enum import Enum
from .mlp import MLP


class EncoderType(Enum):
    MLP = "mlp"

    @classmethod
    def load(cls, encoder_type):
        if encoder_type == cls.MLP:
            return MLP

    @classmethod
    def load_params(cls, encoder_type):
        if encoder_type == cls.MLP:
            return MLP.Params