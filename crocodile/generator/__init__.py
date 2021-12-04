from .styleformer import Styleformer
from .fastgan import FastGAN
from .generator import Generator, GeneratorType, TrainParams
from pathlib import Path
from typing import Optional


def load_from_path(path: Path, epoch: Optional[int] = None, device=None) -> Generator:
    params = TrainParams.load(path, drop_extra_fields=False)
    generator = load_generator(params.generator)
    generator.load(params, epoch, device)
    return generator


def load_generator(generator: GeneratorType) -> Generator:
    if generator == GeneratorType.STYLEFORMER:
        return Styleformer
    elif generator == GeneratorType.FASTGAN:
        return FastGAN
    else:
        raise ValueError()