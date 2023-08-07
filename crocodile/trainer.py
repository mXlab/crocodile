import os
from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass
import subprocess
from simple_parsing import Serializable, subgroups
import mlflow

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from FastGAN import FastGAN, FastGANConfig

from .dataset import LaurenceDataset
from .generator import GeneratorConfig


@dataclass
class TrainerConfig(Serializable):
    experiment_name: str = "crocodile-default-1"
    proxy_server: Optional[str] = None
    host: str = "http://localhost"
    port: int = 8080
    max_epochs: int = 100
    batch_size: int = 16
    num_valid_samples: int = 10000
    use_gpu: bool = True
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    generator: GeneratorConfig = subgroups(
        {"fastgan": FastGANConfig},
        default="fastgan",
    )
    dataloader_workers: Optional[int] = None
    root_dir: Optional[Path] = None
    limit_train_batches: Optional[Union[int, float]] = 1.0

    def __post_init__(self):
        scratch_dir = os.environ.get("SCRATCH")
        if self.root_dir is None and scratch_dir is not None:
            self.root_dir = Path(scratch_dir)


def load_gan(config: GeneratorConfig, im_size: int):
    match config:
        case FastGANConfig():
            return FastGAN(config, im_size)
        case _:
            raise ValueError(f"Generator {config.__class__} is not yet supported.")


class Trainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.generator = load_gan(config.generator, config.dataset.resolution)
        transform_list = [
            transforms.Resize(
                (int(config.dataset.resolution), int(config.dataset.resolution))
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        trans = transforms.Compose(transform_list)
        self.dataset = LaurenceDataset(config.dataset, transform=trans)

        self.validset, _ = random_split(
            self.dataset,
            [config.num_valid_samples, len(self.dataset) - config.num_valid_samples],
        )

    def connect_to_mlflow_server(self):
        if self.config.proxy_server is None:
            raise ValueError("Please specify a valid proxy_server.")
        subprocess.run(
            f"ssh -N -f -L {self.config.port}:{self.config.proxy_server}:{self.config.port} {self.config.proxy_server}",
            shell=True,
            check=True,
        )

    @property
    def num_workers(self):
        cpu_count = os.cpu_count()
        if self.config.dataloader_workers is None:
            if os.environ["SLURM_CPUS_ON_NODE"] is not None:
                return int(os.environ["SLURM_CPUS_ON_NODE"])
            elif cpu_count is not None:
                return cpu_count
            else:
                return 0
        return self.config.dataloader_workers

    def train(self):
        if self.config.proxy_server != os.environ["HOSTNAME"]:
            print(f"Forwarding port {self.config.port} to {self.config.proxy_server}...")
            self.connect_to_mlflow_server()

        print(f"Using {self.num_workers} for dataloading...")

        train_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )

        valid_loader = DataLoader(
            self.validset,
            batch_size=self.config.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

        accelerator = (
            "gpu" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        )

        early_stopping = EarlyStopping(
            monitor="fid_ema", check_on_train_epoch_end=False, patience=10
        )

        print("Starting training...")
        mlflow.set_tracking_uri(f"{self.config.host}:{self.config.port}")
        mlflow.set_experiment(self.config.experiment_name)
        mlflow.pytorch.autolog()
        with mlflow.start_run() as run:
            mlflow.log_params(self.config.to_dict())
            checkpoint_callback = ModelCheckpoint(
                save_top_k=5,
                verbose=True,
                monitor="fid_ema",
                mode="min",
                save_on_train_epoch_end=False,
            )

            print("Initializing Trainer...")
            trainer = pl.Trainer(
                default_root_dir=self.config.root_dir / run.info.run_id,
                max_epochs=self.config.max_epochs,
                accelerator=accelerator,
                callbacks=[checkpoint_callback, early_stopping],
                limit_train_batches=self.config.limit_train_batches,
            )

            trainer.fit(
                model=self.generator,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader,
            )
