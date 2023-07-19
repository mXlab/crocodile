import os
from typing import Optional
from dataclasses import dataclass
from simple_parsing import Serializable, subgroups

from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger

from FastGAN import FastGAN, FastGANConfig

from .dataset import LaurenceDataset
from .generator import GeneratorConfig


@dataclass
class TrainerConfig(Serializable):
    experiment_name: str = "crocodile-default"
    tracking_uri: str = "http://localhost:5000"
    max_epochs: int = 100
    batch_size: int = 32
    num_valid_samples: int = 1000
    use_gpu: bool = True
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    generator: GeneratorConfig = subgroups(
        {"fastgan": FastGANConfig},
        default="fastgan",
    )
    _dataloader_workers: Optional[int] = None

    @property
    def dataloader_workers(self):
        if self._dataloader_workers is None:
            cpu_count = os.cpu_count()
            if cpu_count is None:
                return 0
            else:
                return cpu_count
        else:
            return self._dataloader_workers


def load_generator(config: GeneratorConfig):
    match config:
        case FastGANConfig():
            return FastGAN(config)
        case _:
            raise ValueError(f"Generator {config.__class__} is not yet supported.")


class Trainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.generator = load_generator(config.generator)
        dataset = LaurenceDataset(config.dataset)

        validset, _ = random_split(
            dataset, [config.num_valid_samples, len(dataset) - config.num_valid_samples]
        )

        self.train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.dataloader_workers,
        )

        self.valid_loader = DataLoader(
            validset,
            batch_size=config.batch_size,
            num_workers=config.dataloader_workers,
        )

        mlf_logger = MLFlowLogger(
            experiment_name=config.experiment_name,
            tracking_uri=config.tracking_uri,
            log_model="all",
        )

        accelerator = "gpu" if config.use_gpu else "cpu"
        self.trainer = pl.Trainer(
            logger=mlf_logger, max_epochs=config.max_epochs, accelerator=accelerator
        )

    def train(self):
        self.trainer.fit(
            model=self.generator,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.valid_loader,
        )
