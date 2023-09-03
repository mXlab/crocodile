"""Base trainer class."""
from dataclasses import dataclass
from typing import Protocol
from simple_parsing import Serializable

from .dataset import LaurenceDataset


@dataclass
class TrainConfig(Serializable):
    """Base trainer class."""

    dataset: LaurenceDataset.Params = LaurenceDataset.Params()


class Trainer(Protocol):
    """Base trainer class."""

    def train(self):
        """Train function of model"""
