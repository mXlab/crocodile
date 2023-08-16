"""Base trainer class."""
from dataclasses import dataclass
from typing import Protocol

from crocodile.dataset import LaurenceDataset


@dataclass
class TrainConfig:
    """Base trainer class."""

    dataset: LaurenceDataset.Params = LaurenceDataset.Params()


class Trainer(Protocol):
    """Base trainer class."""

    def train(self):
        """Train function of model"""
