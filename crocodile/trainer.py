import os
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
import subprocess
from simple_parsing import Serializable, subgroups
import mlflow

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from FastGAN import FastGAN, FastGANConfig

from .dataset import LaurenceDataset
from .generator import GeneratorConfig


@dataclass
class TrainerConfig(Serializable):
    experiment_name: str = "crocodile-default"
    host: str = "http://localhost"
    port: int = 5000
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
    def dataloader_workers(self) -> int:
        if self._dataloader_workers is None:
            if os.environ["SLURM_CPUS_ON_NODE"] is not None:
                return int(os.environ["SLURM_CPUS_ON_NODE"])
            elif os.cpu_count() is not None:
                return os.cpu_count()  # TODO: fix error
            else:
                return 0
        else:
            return self._dataloader_workers


def load_gan(config: GeneratorConfig):
    match config:
        case FastGANConfig():
            return FastGAN(config)
        case _:
            raise ValueError(f"Generator {config.__class__} is not yet supported.")


class Trainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.generator = load_gan(config.generator)
        transform = transforms.ToTensor()
        dataset = LaurenceDataset(config.dataset, transform=transform)

        validset, _ = random_split(
            dataset, [config.num_valid_samples, len(dataset) - config.num_valid_samples]
        )

        print(f"Using {config.dataloader_workers} for dataloading...")

        self.train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.dataloader_workers,
            drop_last=True,
        )

        self.valid_loader = DataLoader(
            validset,
            batch_size=config.batch_size,
            num_workers=config.dataloader_workers,
            drop_last=True,
        )

        self.mlflow_server_host = os.environ["HOSTNAME"]

    def connect_to_mlflow_server(self):
        subprocess.run(
            f"ssh -N -f -L {self.config.port}:{self.mlflow_server_host}:{self.config.port} {self.mlflow_server_host}",
            shell=True,
            check=True,
        )

    def train(self):
        if self.mlflow_server_host != os.environ["HOSTNAME"]:
            print(f"Forwarding port {self.config.port} to {self.mlflow_server_host}...")
            self.connect_to_mlflow_server()

        accelerator = (
            "gpu" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        )

        print("Initializing Trainer...")
        trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator=accelerator,
        )

        print("Starting training...")
        mlflow.set_tracking_uri(f"{self.config.host}:{self.config.port}")
        mlflow.set_experiment(self.config.experiment_name)
        mlflow.pytorch.autolog()
        with mlflow.start_run() as run:
            trainer.fit(
                model=self.generator,
                train_dataloaders=self.train_loader,
                val_dataloaders=self.valid_loader,
            )
