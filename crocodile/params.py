from dataclasses import dataclass
from crocodile import generator
from simple_parsing.helpers import Serializable
from simple_parsing.helpers.serialization import register_decoding_fn
from pathlib import Path
from typing import Optional
from crocodile.dataset import LaurenceDataset
from crocodile.utils.optim import OptimizerArgs, OptimizerType
from crocodile.utils.loss import LossParams
from crocodile.encoder import EncoderParams
import os


register_decoding_fn(Path, Path)


@dataclass
class ComputeLatentParams(Serializable):
    generator_path: Path
    epoch: Optional[int] = None
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    batch_size: int = 64
    optimizer: OptimizerArgs = OptimizerArgs(momentum=0.9)
    loss: LossParams = LossParams()
    num_iter: int = 100
    log_dir: Path = Path("./results/latent")
    name: str = "test_1"
    num_test_samples: int = 8
    slurm_job_id: Optional[str] = None
    debug: bool = False

    def __post_init__(self):
        self.save_dir = self.log_dir / self.name
        if self.optimizer.lr is None:
            if self.optimizer.optimizer == OptimizerType.SGD:
                self.optimizer.lr = 20
            elif self.optimizer.optimizer == OptimizerType.ADAM:
                self.optimizer.lr = 2e-2

@dataclass
class TrainEncoderLatentParams(Serializable):
    latent_path: Path
    generator_path: Optional[Path] = None
    epoch: Optional[int] = None
    encoder: EncoderParams = EncoderParams()
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    batch_size: int = 64
    optimizer: OptimizerArgs = OptimizerArgs(momentum=0.9)
    loss: LossParams = LossParams()
    num_epochs: int = 100
    log_dir: Path = Path("./results/encoder_latent")
    name: str = "test_1"
    slurm_job_id: Optional[str] = None
    debug: bool = False
    num_test_samples: int = 8

    def __post_init__(self):
        self.save_dir = self.log_dir / self.name
        if self.optimizer.lr is None:
            if self.optimizer.optimizer == OptimizerType.SGD:
                self.optimizer.lr = 20
            elif self.optimizer.optimizer == OptimizerType.ADAM:
                self.optimizer.lr = 5e-4

@dataclass
class TrainEncoderParams(Serializable):
    generator_path: Optional[Path] = None
    epoch: Optional[int] = None
    encoder: EncoderParams = EncoderParams()
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    batch_size: int = 64
    optimizer: OptimizerArgs = OptimizerArgs(momentum=0.9)
    loss: LossParams = LossParams()
    num_epochs: int = 100
    log_dir: Path = Path("./results/encoder")
    name: str = "test_1"
    slurm_job_id: Optional[str] = None
    debug: bool = False
    num_test_samples: int = 8

    def __post_init__(self):
        self.save_dir = self.log_dir / self.name
        if self.optimizer.lr is None:
            if self.optimizer.optimizer == OptimizerType.SGD:
                self.optimizer.lr = 20
            elif self.optimizer.optimizer == OptimizerType.ADAM:
                self.optimizer.lr = 2e-2
