# from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from crocodile.dataset import LaurenceDataset
from simple_parsing import Serializable
from typing import Optional
from torch.nn import Module
import torch
import shutil
from simple_parsing.helpers.serialization import encode, register_decoding_fn


class GeneratorType(Enum):
    STYLEFORMER = "styleformer"
    FASTGAN = "fastgan"

    @classmethod
    def values(cls):
        return [e.value for e in cls]


@dataclass
class TrainParams(Serializable):
    output_dir: Path = Path("./results")
    generator: GeneratorType = GeneratorType.FASTGAN
    batch_size: int = 64
    exp_name: str = "default"
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    db_path: Path = Path("./crocodile.db")

    def __post_init__(self):
        self.log_dir = self.output_dir / self.exp_name
        self.params_file = self.log_dir / "params.yaml"

    def get_db_path(self):
        return self.db_path.resolve()


@encode.register
def encode_generator_type(obj: GeneratorType) -> str:
    """We choose to encode a tensor as a list, for instance"""
    return obj.name


def decode_generator_type(name: str) -> GeneratorType:
    return GeneratorType[name]


register_decoding_fn(GeneratorType, decode_generator_type)
register_decoding_fn(Path, Path)


class Generator(ABC):
    def __init__(
        self,
        model: Optional[Module] = None,
        latent_dim: Optional[int] = None,
        resolution: Optional[int] = None,
        device=None,
    ):
        self.model = model
        self.device = device
        self.latent_dim = latent_dim
        self.resolution = resolution

    @classmethod
    @abstractmethod
    def prepare(cls, args: TrainParams = TrainParams()):
        pass

    @classmethod
    @abstractmethod
    def train(cls, args: TrainParams = TrainParams()):
        pass

    @staticmethod
    def set_dir(params: TrainParams = TrainParams()):
        if params.log_dir.is_dir():
            shutil.rmtree(params.log_dir)
        params.log_dir.mkdir(exist_ok=True, parents=True)
        params.save(params.params_file)

    @staticmethod
    @abstractmethod
    def load(
        params: TrainParams = TrainParams(),
        epoch: Optional[int] = None,
        device=None,
    ):
        pass

    @abstractmethod
    def sample_z(self, n_samples: int = 1) -> torch.Tensor:
        pass

    @abstractmethod
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        pass
