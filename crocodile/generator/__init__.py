from enum import Enum
from .styleformer import Styleformer
from .fastgan import FastGAN


class ModelType(Enum):
    STYLEFORMER = "styleformer"
    FASTGAN = "fastgan"

    @classmethod
    def load(cls, model_type):
        if model_type == ModelType.STYLEFORMER:
            return Styleformer
        elif model_type == ModelType.FASTGAN:
            return FastGAN