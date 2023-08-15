"""Trainer module"""
from FastGAN import FastGANTrainer, FastGANTrainConfig
from .base import TrainConfig

def load_trainer(config: TrainConfig):
    """Load trainer by config"""
    match config:
        case FastGANTrainConfig():
            return FastGANTrainer(config)
        case _:
            raise NotImplementedError(f"Unknown trainer config: {config}")
        


