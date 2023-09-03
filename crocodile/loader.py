"""Trainer module"""
from pathlib import Path
import torch
from FastGAN.train import FastGANTrainer, FastGANTrainConfig

from .trainer import TrainConfig


trainer_subgroups = {"fastgan": FastGANTrainConfig}


def load_trainer(config: TrainConfig):
    """Load trainer by config"""
    match config:
        case FastGANTrainConfig():
            return FastGANTrainer(config)
        case _:
            raise NotImplementedError(f"Unknown trainer config: {config}")


def load_from_path(path: Path):
    ckpt = torch.load(path)
    config = ckpt["config"]
    params = ckpt["g"]
    match config:
        case FastGANTrainConfig():
            return FastGANTrainer.load_generator(config, params)
        case _:
            raise NotImplementedError("The loaded config is not supported.")
