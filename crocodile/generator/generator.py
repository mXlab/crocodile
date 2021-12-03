from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from crocodile.dataset import LaurenceDataset
from simple_parsing.helpers import Serializable
from typing import Optional
from torch.nn import Module
import torch


class GeneratorType(Enum):
    STYLEFORMER = "styleformer"
    FASTGAN = "fastgan"


@dataclass
class TrainParams(Serializable):
    output_dir: Path = Path("./results")
    generator: GeneratorType = GeneratorType.STYLEFORMER
    batch_size: int = 64
    name: str = "test_1"
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    log_dir: field(init=False)
    params_file: field(init=False)

    def __post_init__(self):
        self.log_dir = self.output_dir / self.name
        self.params_file = self.log_dir / "params.yaml"


class Generator(ABC):
    def __init__(self, model: Optional[Module] = None, latent_dim: Optional[int] = None, device=None):
        self.model = model
        self.device = device
        self.latent_dim = latent_dim

    @classmethod
    @abstractmethod
    def train(cls, args: TrainParams = TrainParams()):
        pass

    @staticmethod
    def set_dir(params: TrainParams = TrainParams()):
        params.log_dir.mkdir(exist_ok=True)
        params.save(params.params_file)

    @staticmethod
    @abstractmethod
    def load(self, params: TrainParams = TrainParams(), epoch: Optional[int] = None, device=None) -> Generator:
        pass

    @abstractmethod
    def sample_z(self, n_samples: int = 1) -> torch.Tensor:
        pass

    @abstractmethod
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        pass
