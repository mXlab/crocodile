from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from simple_parsing import Serializable, subgroups
from simple_parsing.helpers.serialization import register_decoding_fn

from crocodile.dataset import LaurenceDataset
from crocodile.optimizer import OptimizerConfig, SGDConfig, AdamConfig
from crocodile.utils.loss import LossParams
from crocodile.encoder import EncoderParams


register_decoding_fn(Path, Path)


@dataclass
class ComputeLatentParams(Serializable):
    generator_path: Path
    epoch: Optional[int] = None
    dataset: LaurenceDataset.Params = LaurenceDataset.Params(load_biodata=True)
    batch_size: int = 16
    optimizer: OptimizerConfig = subgroups(
        {"sgd": SGDConfig(lr=20, momentum=0.9), "adam": AdamConfig(lr=2e-2)},
        default="sgd",
    )
    loss: LossParams = LossParams()
    num_iter: int = 100
    log_dir: Path = Path("./results/latent")
    name: str = "test_1"
    num_test_samples: int = 8
    slurm_job_id: Optional[str] = None
    debug: bool = False

    def __post_init__(self):
        self.save_dir = self.log_dir / self.name


@dataclass
class TrainEncoderLatentParams(Serializable):
    latent_path: Path
    encoder: EncoderParams = EncoderParams()
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    batch_size: int = 64
    optimizer: OptimizerConfig = subgroups(
        {"sgd": SGDConfig(lr=20, momentum=0.9), "adam": AdamConfig(lr=5e-4)},
        default="sgd",
    )
    loss: LossParams = LossParams()
    num_epochs: int = 100
    log_dir: Path = Path("./results/encoder_latent")
    name: str = "test_1"
    slurm_job_id: Optional[str] = None
    debug: bool = False
    num_test_samples: int = 8

    def __post_init__(self):
        self.save_dir = self.log_dir / self.name


@dataclass
class TrainEncoderParams(Serializable):
    generator_path: Path
    latent_path: Optional[Path] = None
    encoder: EncoderParams = EncoderParams()
    dataset: LaurenceDataset.Params = LaurenceDataset.Params(load_biodata=True)
    batch_size: int = 16
    optimizer: OptimizerConfig = subgroups(
        {"sgd": SGDConfig(lr=20, momentum=0.9), "adam": AdamConfig(lr=2e-2)},
        default="sgd",
    )
    loss: LossParams = LossParams()
    num_epochs: int = 100
    log_dir: Path = Path("./results/encoder")
    name: str = "test_1"
    slurm_job_id: Optional[str] = None
    debug: bool = False
    num_test_samples: int = 8
    latent_regularization: float = 10.0
    decreasing_regularization: bool = False
    num_workers: int = 4

    def __post_init__(self):
        self.save_dir = self.log_dir / self.name
