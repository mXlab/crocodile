from FastGAN.train import FastGAN, FastGANConfig
from crocodile.generator import GeneratorConfig


def load_generator(config: GeneratorConfig):
    match config:
        case FastGANConfig():
            return FastGAN.load_generator(config)
        case _:
            raise ValueError(f"Generator {config.__class__} is not yet supported.")
